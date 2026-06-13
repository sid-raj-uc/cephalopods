"""
Experiment 2: MOG2 background subtraction + blob analysis on Left_Top full 30-min video.

Strategy:
- MOG2 learns static background (tank walls, substrate, equipment)
- Surviving foreground blobs are filtered by size: octopus-range 0.5%–12% of frame
- Score each second by (blob_area * blob_persistence)
- Find top windows, extract frames, visually inspect
"""

import sys, os, subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DIR = os.path.join(PROJECT, "data/aquarium/full/2026-02-20/095420")
OUT_DIR  = os.path.join(PROJECT, "report/experiments/exp02_frames")
os.makedirs(OUT_DIR, exist_ok=True)

CAMERA    = "Left_Top"
PROC_W    = 640    # width for processing (keeps memory low)
PROC_H    = 360
FPS       = 2.0    # sample rate
# Blob size bounds as fraction of total frame area (640*360 = 230400 px)
FRAME_PX  = PROC_W * PROC_H
MIN_BLOB  = 0.003  # >0.3% = min ~700px
MAX_BLOB  = 0.15   # <15% = max ~34560px (human would be bigger)

# Momentum smoothing: how many frames a detection persists
PERSIST_FRAMES = 6  # ~3 seconds at 2fps


def stream_frames(video_path, fps=2.0, width=640, height=360):
    """Yield (timestamp_sec, bgr_frame) via ffmpeg pipe at low res."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    t = 0.0
    step = 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            yield t, frame
            t += step
    finally:
        proc.kill()
        proc.wait()


def analyze_video(video_path, camera_name):
    print(f"\n[{camera_name}] Processing {os.path.basename(video_path)} ...")

    # MOG2 background subtractor
    # history=200 frames (~100s at 2fps) to learn stable background
    # varThreshold=40 — higher = less sensitive to subtle noise
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False
    )

    timestamps = []
    blob_scores = []   # area-based score per frame
    best_frames = []   # (score, t, frame_bgr, mask)

    # Morphological kernels
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    frame_count = 0
    for t, frame in stream_frames(video_path, fps=FPS, width=PROC_W, height=PROC_H):
        fgmask = fgbg.apply(frame)

        # Clean up mask: remove small noise, fill holes
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

        # Find blobs
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_score = 0.0
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            frac = area / FRAME_PX
            if MIN_BLOB <= frac <= MAX_BLOB:
                frame_score = max(frame_score, frac / MAX_BLOB)
                valid_contours.append((area, cnt))

        timestamps.append(t)
        blob_scores.append(frame_score)
        best_frames.append((frame_score, t, frame.copy(), fgmask.copy(), valid_contours))

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed_s = int(t)
            elapsed_m = elapsed_s // 60
            print(f"  t={elapsed_m:02d}:{elapsed_s%60:02d}  frames={frame_count}  "
                  f"max_score_so_far={max(blob_scores):.3f}")

    print(f"  Done: {frame_count} frames processed.")
    return np.array(timestamps), np.array(blob_scores), best_frames


def main():
    vid = os.path.join(FULL_DIR, f"{CAMERA}.mp4")
    if not os.path.exists(vid):
        print(f"ERROR: {vid} not found")
        sys.exit(1)

    timestamps, scores, frames = analyze_video(vid, CAMERA)

    # ── smooth scores with persistence window ────────────────────────────────
    kernel = np.ones(PERSIST_FRAMES) / PERSIST_FRAMES
    smooth_scores = np.convolve(scores, kernel, mode="same")

    # ── find top windows (non-overlapping, min gap 30s) ────────────────────────
    top_indices = []
    used = set()
    gap_frames = int(30 * FPS)
    sorted_idx = np.argsort(-smooth_scores)
    for idx in sorted_idx:
        if any(abs(idx - u) < gap_frames for u in used):
            continue
        top_indices.append(idx)
        used.add(idx)
        if len(top_indices) >= 6:
            break

    print("\nTop windows (after 30s gap dedup):")
    saved_paths = []
    for rank, idx in enumerate(top_indices):
        t = timestamps[idx]
        s = smooth_scores[idx]
        raw_s = scores[idx]
        score_data, _, frame_bgr, fgmask, contours = frames[idx]
        mins = int(t) // 60
        secs = int(t) % 60
        print(f"  #{rank+1}  t={mins:02d}:{secs:02d} ({t:.0f}s)  smooth_score={s:.3f}  raw={raw_s:.3f}")

        # Draw bounding boxes on frame
        vis = frame_bgr.copy()
        for area, cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, f"{area/FRAME_PX*100:.1f}%", (x, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Side-by-side: original + mask
        mask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_rgb])
        fname = os.path.join(OUT_DIR, f"rank{rank+1:02d}_{CAMERA}_t{int(t):04d}s_score{s:.3f}.jpg")
        cv2.imwrite(fname, combined)
        saved_paths.append(fname)
        print(f"     Saved: {os.path.basename(fname)}")

    # ── score timeline plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timestamps / 60, scores, alpha=0.4, color='steelblue', label='raw blob score')
    ax.plot(timestamps / 60, smooth_scores, color='steelblue', lw=2, label='smoothed')
    for idx in top_indices:
        ax.axvline(timestamps[idx]/60, color='red', alpha=0.6, lw=1.5)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("foreground blob score")
    ax.set_title(f"{CAMERA} — MOG2 blob score over 30-min session (095420)")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"{CAMERA}_blob_timeline.png")
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTimeline saved: {os.path.basename(plot_path)}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
