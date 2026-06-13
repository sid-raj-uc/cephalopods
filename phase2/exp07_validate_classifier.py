"""
Experiment 7: Validate exp06 classifier on unseen dates.

Downloads Right_Front only (saves disk) for 3 new sessions:
  - 2026-02-22 / 200003: TV Menu at 20:15 → event at t≈897s in video
  - 2026-02-27 / 093001: Kiss/bluetooth at 09:45 → event at t≈899s in video
  - 2026-03-12 / 060001: NO card, retires to den at 06:00 → mostly empty den session

Loads classifier.pkl from exp06, scans each video at 1fps, reports whether
the classifier fires at the right time and correctly handles the den-empty session.

NOTE: Video filenames encode start time (HHMMSS). Event offset in video =
  (event_hh*3600 + event_mm*60) - (start_hh*3600 + start_mm*60 + start_ss)
"""

import os, subprocess, pickle, re
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

PROJECT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(PROJECT, "data/aquarium/full")
FEAT_DIR  = os.path.join(PROJECT, "data/phase2/exp06_features")
OUT_DIR   = os.path.join(PROJECT, "report/experiments/exp07_frames")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://repo.octopus-intelligence.org/public"
SESSION  = "O-vulgaris-Nity-2026-2-20--"
USER, PASS = "octopus", "communication42"
CAMERA   = "Right Front"   # server-side name (space, not underscore)
CAM_TAG  = "Right_Front"   # local filename

PROC_W   = 800
OX, OY   = 0.614, 0.640
HALF     = 0.04

# Target sessions: (date, timestamp_6digit, event_description, event_t_in_video_approx)
# event_t is approximate seconds into the video where the CSV event occurs
TARGETS = [
    ("2026-02-22", "200003", "TV Menu choice",        897),
    ("2026-02-27", "093001", "Kiss / bluetooth",       899),
    ("2026-03-12", "060001", "NO card → retires to den", 5),
]

transform_dino = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── download ──────────────────────────────────────────────────────────────────

def find_video_url(date: str, hhmm: str) -> str | None:
    cam_enc = CAMERA.replace(" ", "%20")
    listing_url = f"{BASE_URL}/{SESSION}/{cam_enc}/Local/{date}/"
    auth_url = listing_url.replace("https://", f"https://{USER}:{PASS}@")
    result = subprocess.run(["curl", "-s", auth_url], capture_output=True, text=True)
    filenames = re.findall(r'href="([^"]+\.mp4)"', result.stdout)
    for fn in filenames:
        if fn[:4] == hhmm[:4]:
            return f"{listing_url}{fn}"
    return None


def download_right_front(date: str, ts6: str) -> str | None:
    out_dir = os.path.join(DATA_DIR, date, ts6)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{CAM_TAG}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"  [{date}/{ts6}] Already downloaded ({os.path.getsize(out_path)//1_000_000}MB)")
        return out_path

    video_url = find_video_url(date, ts6)
    if not video_url:
        print(f"  [{date}/{ts6}] ERROR: no matching mp4 found on server")
        return None
    print(f"  [{date}/{ts6}] Resolved → {video_url.split('/')[-1]}")

    auth_video = video_url.replace("https://", f"https://{USER}:{PASS}@")
    cmd = ["ffmpeg", "-loglevel", "error", "-stats", "-y",
           "-i", auth_video, "-c:v", "copy", "-c:a", "copy", out_path]
    print(f"  Downloading {CAM_TAG}.mp4 ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1_000_000:
        print(f"  ERROR downloading: {result.stderr[-300:]}")
        return None
    print(f"  ✔ {os.path.getsize(out_path)//1_000_000}MB saved to {out_path}")
    return out_path


# ── frame helpers ──────────────────────────────────────────────────────────────

def get_frame(video_path, t_sec, width=PROC_W):
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    h = int(int(probe[1]) * width / int(probe[0])); h += h % 2
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", str(t_sec), "-i", video_path, "-vframes", "1",
           "-vf", f"scale={width}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read(); proc.wait()
    if len(raw) < width * h * 3: return None, h
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy(), h


def stream_frames(video_path, fps=1.0, width=PROC_W):
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    h = int(int(probe[1]) * width / int(probe[0])); h += h % 2
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
    cx = int(OX * PROC_W); cy = int(OY * h)
    hw = int(HALF * PROC_W); hh = int(HALF * h)
    return frame[max(0,cy-hh):min(h,cy+hh), max(0,cx-hw):min(PROC_W,cx+hw)]


@torch.no_grad()
def dino_embed(model, bgr_patch):
    rgb = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform_dino(pil).unsqueeze(0)
    out = model.forward_features(x)
    v = out['x_norm_clstoken'][0].cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load exp06 classifier + background + query vector
    print("Loading classifier from exp06 ...")
    clf_path = os.path.join(FEAT_DIR, "classifier.pkl")
    with open(clf_path, 'rb') as f:
        saved = pickle.load(f)
    clf, scaler, bg, query_vec = saved['clf'], saved['scaler'], saved['bg'], saved['query_vec']
    print(f"  Classifier: {clf}")
    print(f"  Background patch: {bg.shape}")

    print("\nLoading DINOv2 ViT-B/14 ...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    model.eval()
    print("  Loaded.\n")

    results = {}

    for date, ts6, event_desc, event_t in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {date} / {ts6} — {event_desc}")
        print(f"  Expected event at t≈{event_t}s into video")
        print(f"{'='*60}")

        # 1. Download
        vid_path = download_right_front(date, ts6)
        if vid_path is None:
            print("  SKIP: download failed")
            continue

        # 2. Scan at 1fps
        print(f"  Scanning at 1fps ...")
        feats, probs_raw, t_arr = [], [], []

        for t, frame, h in stream_frames(vid_path, fps=1.0):
            patch = tight_crop(frame, h)
            patch_f = cv2.resize(patch.astype(np.float32), (int(bg.shape[1]), int(bg.shape[0])))
            patch_sub = np.clip(patch_f - bg + 128, 0, 255).astype(np.uint8)
            vec = dino_embed(model, patch_sub)
            feats.append(vec)
            t_arr.append(t)

        feats = np.array(feats, dtype=np.float32)
        t_arr = np.array(t_arr, dtype=np.float32)
        print(f"  {len(t_arr)} frames scanned")

        # 3. Classify
        X_s = scaler.transform(feats)
        probs = clf.predict_proba(X_s)[:, 1]
        peak_p = probs.max()
        peak_t = t_arr[probs.argmax()]
        n_detected = (probs >= 0.5).sum()

        print(f"  Peak P(octopus)={peak_p:.3f} at t={peak_t:.0f}s")
        print(f"  Frames above 0.5: {n_detected} / {len(probs)}")

        # Check if classifier fires near the event window (±120s slack for timing shift)
        window_lo = max(0, event_t - 120)
        window_hi = min(t_arr[-1], event_t + 120)
        mask_window = (t_arr >= window_lo) & (t_arr <= window_hi)
        p_at_event = probs[mask_window].max() if mask_window.any() else 0.0
        print(f"  P at event window [{window_lo:.0f}-{window_hi:.0f}s]: {p_at_event:.3f}")

        results[f"{date}/{ts6}"] = {
            'probs': probs, 't_arr': t_arr,
            'peak_p': peak_p, 'peak_t': peak_t,
            'event_t': event_t, 'p_at_event': p_at_event,
            'event_desc': event_desc, 'n_detected': int(n_detected),
        }

        # 4. Save top detection frames
        top_idx = np.argsort(-probs)[:3]
        for rank, idx in enumerate(top_idx):
            if probs[idx] < 0.3: break
            frame_out, h_out = get_frame(vid_path, int(t_arr[idx]), width=1280)
            if frame_out is None: continue
            cx = int(OX * 1280); cy = int(OY * h_out)
            hw = int(HALF * 1280); hh = int(HALF * h_out)
            cv2.rectangle(frame_out, (cx-hw*2, cy-hh*2), (cx+hw*2, cy+hh*2), (0,255,0), 2)
            cv2.putText(frame_out, f"P={probs[idx]:.3f} t={t_arr[idx]:.0f}s",
                        (cx-hw*2, cy-hh*2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            safe_date = date.replace("-","")
            fname = os.path.join(OUT_DIR, f"{safe_date}_{ts6}_rank{rank+1:02d}_t{int(t_arr[idx]):04d}s_p{probs[idx]:.3f}.jpg")
            cv2.imwrite(fname, frame_out)
            print(f"  Saved: {os.path.basename(fname)}")

    # 5. Combined timeline plot
    if results:
        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=(16, 3.5*n), squeeze=False)
        for i, (key, r) in enumerate(results.items()):
            ax = axes[i][0]
            ax.plot(r['t_arr']/60, r['probs'], color='steelblue', lw=1.2, label='P(octopus)')
            ax.axhline(0.5, color='red', ls='--', lw=1, label='threshold=0.5')
            ax.fill_between(r['t_arr']/60, 0, r['probs'], where=(r['probs']>=0.5),
                            color='red', alpha=0.25, label='detected')
            # mark expected event time
            ax.axvline(r['event_t']/60, color='green', ls=':', lw=1.5,
                       label=f"CSV event: {r['event_desc']} (±2min window)")
            ax.axvspan((r['event_t']-120)/60, (r['event_t']+120)/60,
                       color='green', alpha=0.08)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{key} | peak={r['peak_p']:.3f} @ t={r['peak_t']:.0f}s | "
                         f"detected={r['n_detected']} frames | P@event={r['p_at_event']:.3f}")
            ax.set_xlabel("time (min)"); ax.set_ylabel("P(octopus)")
            ax.legend(fontsize=8, loc='upper right')
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "exp07_timelines.png")
        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"\nTimeline plot saved: {plot_path}")

    # 6. Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for key, r in results.items():
        verdict = "✅ DETECTED" if r['peak_p'] >= 0.5 else "❌ NOT DETECTED"
        event_ok = "✅" if r['p_at_event'] >= 0.5 else "⚠️ "
        print(f"  {key:25s}  peak={r['peak_p']:.3f}  {verdict}  "
              f"P@event={r['p_at_event']:.3f} {event_ok}  ({r['event_desc']})")
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
