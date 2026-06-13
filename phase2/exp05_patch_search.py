"""
Experiment 5: Tight-patch DINOv2 similarity with background subtraction.

Fix for Exp 4: instead of CLS on a large crop (dominated by background),
we use a very tight crop on the octopus blob (~56×56px) and subtract the
median frame before computing similarity. This isolates what's different
from the static background.

Also tries pixel-residual MAD as a simpler parallel signal.
"""

import sys, os, subprocess
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

PROJECT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DIR  = os.path.join(PROJECT, "data/aquarium/full/2026-02-20")
OUT_DIR   = os.path.join(PROJECT, "report/experiments/exp05_frames")
os.makedirs(OUT_DIR, exist_ok=True)

TIMESTAMPS = ["095420", "102421", "112421", "122421", "132421"]
CAMERA    = "Right_Front"
PROC_W    = 800
SAMPLE_FPS = 0.5

# Confirmed octopus center in normalised coords (from GDino, Exp 1)
OX, OY = 0.614, 0.640  # normalized cx, cy
# Tight crop: 7% of frame width around the blob center (~56px at 800px wide)
HALF = 0.04   # half-width in normalized units

transform_dino = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_frame(video_path, t_sec, width=PROC_W):
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    h = int(int(probe[1]) * width / int(probe[0]))
    if h % 2: h += 1
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", str(t_sec), "-i", video_path, "-vframes", "1",
           "-vf", f"scale={width}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read(); proc.wait()
    if len(raw) < width * h * 3:
        return None, h
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy(), h


def stream_frames(video_path, fps=0.5, width=PROC_W):
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    h = int(int(probe[1]) * width / int(probe[0]))
    if h % 2: h += 1
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", video_path, "-vf", f"fps={fps},scale={width}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * h * 3
    t, step = 0.0, 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes: break
            yield t, np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy(), h
            t += step
    finally:
        proc.kill(); proc.wait()


def tight_crop(frame, h):
    """Extract tight octopus-region crop."""
    cx = int(OX * PROC_W); cy = int(OY * h)
    hw = int(HALF * PROC_W); hh = int(HALF * h)
    x1, x2 = max(0, cx-hw), min(PROC_W, cx+hw)
    y1, y2 = max(0, cy-hh), min(h, cy+hh)
    return frame[y1:y2, x1:x2]


@torch.no_grad()
def dino_embed(model, bgr_patch):
    """DINOv2 CLS of a tight patch."""
    rgb = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform_dino(pil).unsqueeze(0)
    out = model.forward_features(x)
    v = out['x_norm_clstoken'][0].cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-8)


def cosine(a, b):
    return float(np.dot(a, b))


def build_background(ts_list, n_frames=20):
    """Build per-pixel median background from sparse frames across sessions."""
    samples = []
    for ts in ts_list:
        vid = os.path.join(FULL_DIR, ts, f"{CAMERA}.mp4")
        if not os.path.exists(vid): continue
        # Sample n_frames evenly spaced across the 30-min video
        for t in np.linspace(60, 1200, n_frames // len(ts_list)):
            frame, h = get_frame(vid, int(t))
            if frame is None: continue
            patch = tight_crop(frame, h).astype(np.float32)
            if patch.size > 0:
                samples.append(patch)
    if not samples: return None
    # Resize all to same shape
    target_h, target_w = samples[0].shape[:2]
    resized = [cv2.resize(s, (target_w, target_h)) for s in samples]
    return np.median(np.stack(resized, axis=0), axis=0).astype(np.float32)


def main():
    print("Loading DINOv2 ViT-B/14 ...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    model.eval()
    print("Loaded.\n")

    # ── build background median ───────────────────────────────────────────────
    print("Building background median from all sessions ...")
    bg = build_background(TIMESTAMPS, n_frames=40)
    if bg is None:
        print("  ERROR: couldn't build background"); return
    bg_gray = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    cv2.imwrite(os.path.join(OUT_DIR, "background_median.jpg"), bg.astype(np.uint8))
    print(f"  Background patch shape: {bg.shape}")

    # ── query embedding from confirmed octopus frame ──────────────────────────
    print("\nBuilding query from 095420 t=1024s ...")
    qvid = os.path.join(FULL_DIR, "095420", f"{CAMERA}.mp4")
    qframe, qh = get_frame(qvid, 1024)
    if qframe is None:
        print("ERROR: could not extract query frame"); return
    qpatch = tight_crop(qframe, qh)

    # Save query patch for reference
    cv2.imwrite(os.path.join(OUT_DIR, "query_tight_patch.jpg"),
                cv2.resize(qpatch, (qpatch.shape[1]*4, qpatch.shape[0]*4), interpolation=cv2.INTER_CUBIC))

    # Background-subtracted version
    qpatch_f = cv2.resize(qpatch.astype(np.float32), (bg.shape[1], bg.shape[0]))
    q_sub = np.clip(qpatch_f - bg + 128, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "query_bgsub_patch.jpg"),
                cv2.resize(q_sub, (q_sub.shape[1]*4, q_sub.shape[0]*4), interpolation=cv2.INTER_CUBIC))

    query_vec_raw = dino_embed(model, qpatch)
    query_vec_sub = dino_embed(model, q_sub)
    print(f"  Query patch: {qpatch.shape}")

    # ── empty-tank reference from 132421 t=300s ───────────────────────────────
    print("Computing empty-tank baseline ...")
    bvid = os.path.join(FULL_DIR, "132421", f"{CAMERA}.mp4")
    bframe, bh = get_frame(bvid, 300)
    bpatch = tight_crop(bframe, bh)
    bpatch_f = cv2.resize(bpatch.astype(np.float32), (bg.shape[1], bg.shape[0]))
    b_sub = np.clip(bpatch_f - bg + 128, 0, 255).astype(np.uint8)
    baseline_raw = cosine(query_vec_raw, dino_embed(model, bpatch))
    baseline_sub = cosine(query_vec_sub, dino_embed(model, b_sub))
    print(f"  Baseline raw: {baseline_raw:.4f}  baseline_sub: {baseline_sub:.4f}\n")

    # ── scan all timestamps ───────────────────────────────────────────────────
    all_results = {}

    for ts in TIMESTAMPS:
        vid = os.path.join(FULL_DIR, ts, f"{CAMERA}.mp4")
        if not os.path.exists(vid): continue
        print(f"[{ts}] Scanning ...")

        t_arr, sim_raw_arr, sim_sub_arr, mad_arr, frame_store = [], [], [], [], []

        for t, frame, h in stream_frames(vid, fps=SAMPLE_FPS, width=PROC_W):
            patch = tight_crop(frame, h)
            patch_f = cv2.resize(patch.astype(np.float32), (bg.shape[1], bg.shape[0]))
            patch_sub = np.clip(patch_f - bg + 128, 0, 255).astype(np.uint8)

            vec_raw = dino_embed(model, patch)
            vec_sub = dino_embed(model, patch_sub)
            sim_r = cosine(query_vec_raw, vec_raw)
            sim_s = cosine(query_vec_sub, vec_sub)

            # Pixel-level MAD of bg-subtracted patch vs query
            q_gray = cv2.cvtColor(q_sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
            p_gray = cv2.cvtColor(patch_sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mad = float(np.mean(np.abs(q_gray - p_gray)))

            t_arr.append(t); sim_raw_arr.append(sim_r)
            sim_sub_arr.append(sim_s); mad_arr.append(mad)
            frame_store.append((sim_s, t, frame.copy()))

        t_arr = np.array(t_arr)
        sim_raw_arr = np.array(sim_raw_arr)
        sim_sub_arr = np.array(sim_sub_arr)
        mad_arr     = np.array(mad_arr)

        print(f"  Done: {len(t_arr)} frames")
        print(f"  sim_raw: max={sim_raw_arr.max():.4f} mean={sim_raw_arr.mean():.4f}")
        print(f"  sim_sub: max={sim_sub_arr.max():.4f} mean={sim_sub_arr.mean():.4f}")
        print(f"  mad:     min={mad_arr.min():.2f} mean={mad_arr.mean():.2f}")

        all_results[ts] = (t_arr, sim_raw_arr, sim_sub_arr, mad_arr, frame_store)

        # Save top-3 by background-subtracted similarity
        threshold = baseline_sub + 0.03
        top = sorted(frame_store, key=lambda x: -x[0])
        saved = 0
        for _, (sim_s, t, fr) in enumerate(top):
            if sim_s < threshold or saved >= 3: break
            fname = os.path.join(OUT_DIR, f"{ts}_rank{saved+1:02d}_t{int(t):04d}s_simsub{sim_s:.4f}.jpg")
            cv2.imwrite(fname, fr)
            print(f"  Saved: {os.path.basename(fname)}")
            saved += 1

    # ── combined timeline ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(all_results), 1,
                             figsize=(16, 3*len(all_results)), squeeze=False)
    for i, (ts, (t_arr, _, sim_sub, mad, _)) in enumerate(all_results.items()):
        ax = axes[i][0]
        ax.plot(t_arr/60, sim_sub, color='darkorange', lw=1.5, label='sim_sub')
        ax.axhline(baseline_sub, color='gray', linestyle='--', lw=1, label=f'baseline={baseline_sub:.3f}')
        ax.axhline(baseline_sub + 0.03, color='red', linestyle='--', lw=1, label='threshold')
        ax.fill_between(t_arr/60, baseline_sub, sim_sub,
                        where=(sim_sub >= baseline_sub+0.03), color='red', alpha=0.25)
        ax.set_ylim(max(0, sim_sub.min()-0.02), min(1.0, sim_sub.max()+0.02))
        ax.set_title(f"{ts} — bgsub DINOv2 sim to Nity patch")
        ax.set_xlabel("time (min)"); ax.set_ylabel("cosine sim")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "patch_similarity_timeline.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)

    print("\n=== SUMMARY ===")
    print(f"Baselines — raw: {baseline_raw:.4f}  bgsub: {baseline_sub:.4f}")
    for ts, (t_arr, sim_raw, sim_sub, mad, _) in all_results.items():
        above = (sim_sub >= baseline_sub + 0.03).sum()
        print(f"  {ts}: sim_sub max={sim_sub.max():.4f}  above_thresh={above}")


if __name__ == "__main__":
    main()
