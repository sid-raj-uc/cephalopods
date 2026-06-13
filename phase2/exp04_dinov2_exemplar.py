"""
Experiment 4: DINOv2 exemplar search.

Use the confirmed Nity crop from 095420 Right_Front t=1024s as a query.
Compute its DINOv2 CLS embedding, then scan Right_Front across all timestamps
at 0.5fps, comparing each frame's CLS embedding via cosine similarity.

High similarity = frame looks like the confirmed octopus frame.
Bypasses GDino text queries entirely — no human/octopus confusion.

Memory: DINOv2-B/14 ~300MB, process 1 frame at a time.
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
OUT_DIR   = os.path.join(PROJECT, "report/experiments/exp04_frames")
os.makedirs(OUT_DIR, exist_ok=True)

TIMESTAMPS = ["095420", "102421", "112421", "122421", "132421"]
CAMERA     = "Right_Front"
SAMPLE_FPS = 0.5
PROC_W     = 518   # DINOv2 ViT-B/14 prefers multiples of 14; 518 = 37×14

# Confirmed octopus bbox in 095420 Right_Front at t=1024s
# GDino: cx=0.614, cy=0.640, w=0.080, h=0.137 at 800×450
# At 3840×2160 native: cx=2357, cy=1382, w=307, h=296
# Crop with padding
QUERY_VID = os.path.join(FULL_DIR, "095420", "Right_Front.mp4")
QUERY_T   = 1024  # seconds

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_frame(video_path, t_sec, width=PROC_W):
    """Extract single frame at t_sec, return BGR numpy array."""
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    orig_w, orig_h = int(probe[0]), int(probe[1])
    h = int(orig_h * width / orig_w)
    if h % 2: h += 1
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", str(t_sec), "-i", video_path, "-vframes", "1",
           "-vf", f"scale={width}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read()
    proc.wait()
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy()


def stream_frames(video_path, fps=0.5, width=PROC_W):
    """Yield (t_sec, bgr_frame) via ffmpeg pipe."""
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    orig_w, orig_h = int(probe[0]), int(probe[1])
    h = int(orig_h * width / orig_w)
    if h % 2: h += 1
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", video_path,
           "-vf", f"fps={fps},scale={width}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * h * 3
    t, step = 0.0, 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield t, np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy()
            t += step
    finally:
        proc.kill(); proc.wait()


def to_tensor(bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return transform(pil).unsqueeze(0)  # (1,3,224,224)


@torch.no_grad()
def get_cls(model, bgr_frame):
    x = to_tensor(bgr_frame)
    out = model.forward_features(x)
    return out['x_norm_clstoken'][0].cpu().numpy()  # (768,)


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def main():
    print("Loading DINOv2 ViT-B/14 ...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    model.eval()
    print("Model loaded.\n")

    # ── build query embedding from confirmed octopus frame ───────────────────
    print(f"Extracting query frame: {CAMERA} t={QUERY_T}s from 095420 ...")
    query_frame = extract_frame(QUERY_VID, QUERY_T)

    # Crop to the confirmed octopus region (bbox area + padding)
    # At PROC_W=518, 3840→518 scale: cx=0.614*518=318, cy=0.640*h
    h_q = query_frame.shape[0]
    cx_px = int(0.614 * PROC_W)
    cy_px = int(0.640 * h_q)
    half_w = int(0.080 * PROC_W / 2 * 3)   # 3× bbox width as context
    half_h = int(0.137 * h_q / 2 * 3)
    x1 = max(0, cx_px - half_w); x2 = min(PROC_W, cx_px + half_w)
    y1 = max(0, cy_px - half_h); y2 = min(h_q, cy_px + half_h)
    query_crop = query_frame[y1:y2, x1:x2]

    # Save the query crop for reference
    query_crop_big = cv2.resize(query_crop, (query_crop.shape[1]*3, query_crop.shape[0]*3),
                                interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT_DIR, "query_crop_nity.jpg"), query_crop_big)
    print(f"Query crop saved: {query_crop.shape[1]}×{query_crop.shape[0]}px")

    query_vec = get_cls(model, query_crop)
    print(f"Query embedding: norm={np.linalg.norm(query_vec):.3f}\n")

    # ── also compute a baseline "empty tank" vector ──────────────────────────
    # Use t=300s (5 min) from 132421 — no detections, known quiet period
    print("Computing empty-tank baseline embedding (132421 t=300s) ...")
    baseline_vid = os.path.join(FULL_DIR, "132421", "Right_Front.mp4")
    baseline_frame = extract_frame(baseline_vid, 300)
    # same crop region
    h_b = baseline_frame.shape[0]
    cx_px2 = int(0.614 * PROC_W)
    cy_px2 = int(0.640 * h_b)
    x1b = max(0, cx_px2 - half_w); x2b = min(PROC_W, cx_px2 + half_w)
    y1b = max(0, cy_px2 - half_h); y2b = min(h_b, cy_px2 + half_h)
    baseline_crop = baseline_frame[y1b:y2b, x1b:x2b]
    baseline_vec = get_cls(model, baseline_crop)
    baseline_sim = cosine_sim(query_vec, baseline_vec)
    print(f"Baseline similarity (empty tank): {baseline_sim:.4f}")
    print(f"(Anything >> {baseline_sim:.3f} is interesting)\n")

    # ── scan all timestamps ───────────────────────────────────────────────────
    all_results = {}

    for ts in TIMESTAMPS:
        vid = os.path.join(FULL_DIR, ts, f"{CAMERA}.mp4")
        if not os.path.exists(vid):
            print(f"  SKIP {ts}: not found"); continue

        print(f"[{ts}] Scanning ...")
        t_arr, sim_arr, frames = [], [], []

        for t, frame in stream_frames(vid, fps=SAMPLE_FPS, width=PROC_W):
            # Crop same region
            h_f = frame.shape[0]
            cx_ = int(0.614 * PROC_W); cy_ = int(0.640 * h_f)
            x1_ = max(0, cx_ - half_w); x2_ = min(PROC_W, cx_ + half_w)
            y1_ = max(0, cy_ - half_h); y2_ = min(h_f, cy_ + half_h)
            crop = frame[y1_:y2_, x1_:x2_]

            vec = get_cls(model, crop)
            sim = cosine_sim(query_vec, vec)
            t_arr.append(t); sim_arr.append(sim)
            frames.append((sim, t, frame.copy()))

        t_arr = np.array(t_arr); sim_arr = np.array(sim_arr)
        print(f"  Done: {len(t_arr)} frames | max_sim={sim_arr.max():.4f} mean={sim_arr.mean():.4f}")
        all_results[ts] = (t_arr, sim_arr, frames)

        # Save top-3 frames above baseline
        threshold = baseline_sim + 0.05
        top = sorted(frames, key=lambda x: -x[0])
        saved = 0
        for rank, (sim, t, frame) in enumerate(top[:6]):
            if sim < threshold or saved >= 3:
                break
            fname = os.path.join(OUT_DIR, f"{ts}_rank{saved+1:02d}_t{int(t):04d}s_sim{sim:.4f}.jpg")
            cv2.imwrite(fname, frame)
            print(f"  Saved: {os.path.basename(fname)}")
            saved += 1

    # ── combined timeline plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(len(all_results), 1,
                             figsize=(16, 3 * len(all_results)), squeeze=False)
    for i, (ts, (t_arr, sim_arr, _)) in enumerate(all_results.items()):
        ax = axes[i][0]
        ax.plot(t_arr / 60, sim_arr, color='darkorange', lw=1.5)
        ax.axhline(baseline_sim, color='gray', linestyle='--', lw=1, label=f'baseline={baseline_sim:.3f}')
        ax.axhline(baseline_sim + 0.05, color='red', linestyle='--', lw=1, label='threshold')
        ax.fill_between(t_arr / 60, baseline_sim, sim_arr,
                        where=(sim_arr >= baseline_sim + 0.05), color='red', alpha=0.25)
        ax.set_ylim(baseline_sim - 0.05, min(1.0, sim_arr.max() + 0.05))
        ax.set_ylabel("cosine sim")
        ax.set_title(f"{ts} — DINOv2 similarity to Nity exemplar")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dinov2_exemplar_timeline.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)

    print("\n=== SUMMARY ===")
    print(f"Baseline (empty tank): {baseline_sim:.4f}")
    for ts, (t_arr, sim_arr, _) in all_results.items():
        above = t_arr[sim_arr >= baseline_sim + 0.05]
        print(f"  {ts}: max={sim_arr.max():.4f}  frames_above_threshold={len(above)}")


if __name__ == "__main__":
    main()
