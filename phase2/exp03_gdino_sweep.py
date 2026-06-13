"""
Experiment 3: GroundingDINO sweep across all 5 timestamps on Right_Front.

Improvements over Exp 1:
- Scans all timestamps (095420, 102421, 112421, 122421, 132421) not just one clip
- Crops to tank interior ROI before running GDino (eliminates aquarist-edge FPs)
- Samples at 0.5fps (1 frame per 2s) — enough resolution to detect 60s events
- Saves top frames per timestamp + detection timeline plot
"""

import sys, os, subprocess
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GroundingDINO'))
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

PROJECT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DIR   = os.path.join(PROJECT, "data/aquarium/full/2026-02-20")
CONFIG     = os.path.join(PROJECT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS    = os.path.join(PROJECT, "weights/groundingdino_swint_ogc.pth")
OUT_DIR    = os.path.join(PROJECT, "report/experiments/exp03_frames")
os.makedirs(OUT_DIR, exist_ok=True)

TIMESTAMPS = ["112421", "122421", "132421"]  # 095420 and 102421 already done
CAMERA     = "Right_Front"
QUERY      = "octopus"
BOX_THRESH    = 0.30
TEXT_THRESH   = 0.25
SAMPLE_FPS    = 0.5    # 1 frame per 2 seconds
PROC_W        = 800

# Tank interior ROI in normalized coordinates (0-1) relative to 800px-wide frame
# Right_Front: tank glass occupies roughly x=0.08-0.88, y=0.25-0.82
# Exclude left edge (aquarist arm zone) and far right border
ROI_X1, ROI_X2 = 0.15, 0.85
ROI_Y1, ROI_Y2 = 0.28, 0.80

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def stream_frames(video_path, fps=0.5, width=800):
    """Yield (timestamp_sec, bgr_frame) at low fps."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale={width}:-2",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
    ]
    # Get height
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    orig_w, orig_h = int(probe[0]), int(probe[1])
    scale = width / orig_w
    h = int(orig_h * scale)
    if h % 2: h += 1

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * h * 3
    t = 0.0
    step = 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy()
            yield t, frame, h
            t += step
    finally:
        proc.kill(); proc.wait()


def infer(model, bgr_frame):
    """Run GDino on full frame, return boxes/scores filtered to ROI."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor, _ = transform(pil, None)
    with torch.no_grad():
        boxes, logits, _ = predict(
            model=model, image=tensor, caption=QUERY,
            box_threshold=BOX_THRESH, text_threshold=TEXT_THRESH, device="cpu",
        )
    if len(boxes) == 0:
        return [], []

    h, w = bgr_frame.shape[:2]
    # Filter to ROI: keep only boxes whose center falls inside the tank interior
    keep_boxes, keep_scores = [], []
    for box, score in zip(boxes.numpy(), logits.numpy()):
        cx, cy, bw, bh = box
        if ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2:
            keep_boxes.append(box)
            keep_scores.append(float(score))
    return keep_boxes, keep_scores


def scan_timestamp(model, ts_dir, timestamp):
    vid = os.path.join(ts_dir, f"{CAMERA}.mp4")
    if not os.path.exists(vid):
        print(f"  SKIP: {vid} not found")
        return [], []

    print(f"\n[{timestamp}] Scanning {CAMERA} at {SAMPLE_FPS}fps ...")
    t_arr, score_arr, frame_store = [], [], []

    for t, frame, h in stream_frames(vid, fps=SAMPLE_FPS, width=PROC_W):
        boxes, scores = infer(model, frame)
        best = max(scores) if scores else 0.0
        t_arr.append(t)
        score_arr.append(best)
        frame_store.append((best, t, frame.copy(), boxes, scores))
        if best >= BOX_THRESH:
            mins = int(t) // 60
            secs = int(t) % 60
            print(f"  t={mins:02d}:{secs:02d}  score={best:.3f}  boxes={len(boxes)}")

    print(f"  Done: {len(t_arr)} frames  |  max={max(score_arr) if score_arr else 0:.3f}")
    return (np.array(t_arr), np.array(score_arr), frame_store)


def save_top_frames(frame_store, timestamp, n=3):
    """Save top-n frames for a timestamp with bounding boxes."""
    sorted_frames = sorted(frame_store, key=lambda x: -x[0])
    paths = []
    for rank, (score, t, frame, boxes, scores) in enumerate(sorted_frames[:n]):
        if score < BOX_THRESH:
            break
        h, w = frame.shape[:2]
        vis = frame.copy()
        for box, s in zip(boxes, scores):
            cx, cy, bw, bh = box
            x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{s:.2f}", (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw ROI boundary
        rx1 = int(ROI_X1 * w); ry1 = int(ROI_Y1 * h)
        rx2 = int(ROI_X2 * w); ry2 = int(ROI_Y2 * h)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 200, 255), 1)

        fname = os.path.join(OUT_DIR, f"{timestamp}_rank{rank+1:02d}_t{int(t):04d}s_score{score:.3f}.jpg")
        cv2.imwrite(fname, vis)
        paths.append(fname)
        print(f"    Saved: {os.path.basename(fname)}")
    return paths


def main():
    print("Loading GroundingDINO ...")
    model = load_model(CONFIG, WEIGHTS, device="cpu")
    model.eval()
    print("Model loaded.\n")

    all_results = {}

    for ts in TIMESTAMPS:
        ts_dir = os.path.join(FULL_DIR, ts)
        result = scan_timestamp(model, ts_dir, ts)
        if result and len(result[0]) > 0:
            t_arr, score_arr, frame_store = result
            all_results[ts] = (t_arr, score_arr, frame_store)
            save_top_frames(frame_store, ts)

    # ── combined timeline plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(len(all_results), 1,
                             figsize=(16, 3 * len(all_results)), squeeze=False)
    for i, (ts, (t_arr, score_arr, _)) in enumerate(all_results.items()):
        ax = axes[i][0]
        ax.plot(t_arr / 60, score_arr, color='steelblue', lw=1.5)
        ax.axhline(y=BOX_THRESH, color='red', linestyle='--', alpha=0.6, lw=1)
        ax.fill_between(t_arr / 60, 0, score_arr,
                        where=(score_arr >= BOX_THRESH), color='red', alpha=0.25)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("GDino score")
        ax.set_title(f"{ts} — {CAMERA} (ROI-filtered, {SAMPLE_FPS}fps)")
        ax.set_xlabel("time (min)")
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "gdino_sweep_timeline.png")
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTimeline plot: {plot_path}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    for ts, (t_arr, score_arr, _) in all_results.items():
        peaks = t_arr[score_arr >= BOX_THRESH]
        if len(peaks):
            peak_mins = [f"{int(p)//60:02d}:{int(p)%60:02d}" for p in peaks[:5]]
            print(f"  {ts}: {len(peaks)} detection frames  | peaks at {', '.join(peak_mins)}"
                  f"  | max={score_arr.max():.3f}")
        else:
            print(f"  {ts}: 0 detections above threshold {BOX_THRESH}")


if __name__ == "__main__":
    main()
