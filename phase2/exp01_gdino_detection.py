"""
Experiment 1: GroundingDINO zero-shot detection on the 985-1045s candidate clips.
Cameras: Left_Top + Right_Front (low-bias cameras).
Samples 1 frame/sec, runs 'octopus' query, saves top-6 detections with bboxes.
"""

import sys, os, io, subprocess
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GroundingDINO'))
from groundingdino.util.inference import load_model, predict
from groundingdino.util.inference import annotate
import groundingdino.datasets.transforms as T

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(PROJECT, "data/aquarium/clips/2026-02-20/095420/985_1045")
CONFIG   = os.path.join(PROJECT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS  = os.path.join(PROJECT, "weights/groundingdino_swint_ogc.pth")
OUT_DIR  = os.path.join(PROJECT, "report/experiments/exp01_frames")
os.makedirs(OUT_DIR, exist_ok=True)

CAMERAS  = ["Left_Top", "Right_Front"]
QUERY    = "octopus"
BOX_THRESH  = 0.25   # lower threshold to catch partial views
TEXT_THRESH = 0.20
RESIZE_W    = 800    # resize before inference to save RAM

# ── GroundingDINO transform (matches training) ─────────────────────────────────
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_frame_tensor(bgr_frame):
    """Convert a BGR numpy frame to a normalized tensor for GDino."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor, _ = transform(pil, None)
    return tensor

def extract_frames_ffmpeg(video_path, fps=1.0, max_width=RESIZE_W):
    """Yield (timestamp_sec, bgr_array) at the given fps via ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale={max_width}:-2",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
    ]
    probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,duration",
                 "-of", "csv=p=0", video_path]
    info = subprocess.check_output(probe_cmd).decode().strip().split(",")
    orig_w, orig_h = int(info[0]), int(info[1])
    scale = max_width / orig_w
    w, h = max_width, int(orig_h * scale)
    if h % 2: h += 1

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    t = 0.0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            yield t, frame
            t += 1.0 / fps
    finally:
        proc.kill()
        proc.wait()

def draw_detection(bgr_frame, boxes, scores, timestamp, camera):
    """Return an annotated RGB matplotlib figure."""
    h, w = bgr_frame.shape[:2]
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(rgb)
    ax.set_title(f"{camera}  t={timestamp:.1f}s  best_score={max(scores):.3f}", fontsize=11)
    ax.axis("off")
    for box, score in zip(boxes, scores):
        cx, cy, bw, bh = box
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        rect = patches.Rectangle((x1, y1), bw*w, bh*h,
                                  linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{score:.2f}", color='lime',
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=1))
    plt.tight_layout(pad=0.5)
    return fig

def run_camera(model, video_path, camera_name):
    print(f"\n[{camera_name}] scanning {os.path.basename(video_path)} ...")
    detections = []  # list of (score, timestamp, frame_bgr, boxes, scores)

    for t, frame in extract_frames_ffmpeg(video_path, fps=1.0):
        tensor = load_frame_tensor(frame)
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=tensor,
                caption=QUERY,
                box_threshold=BOX_THRESH,
                text_threshold=TEXT_THRESH,
                device="cpu",
            )
        if len(boxes) > 0:
            best = float(logits.max())
            detections.append((best, t, frame.copy(), boxes.numpy(), logits.numpy()))
            print(f"  t={t:.0f}s  boxes={len(boxes)}  best_score={best:.3f}")

    detections.sort(key=lambda x: -x[0])
    return detections

def main():
    print("Loading GroundingDINO (SwinT) ...")
    model = load_model(CONFIG, WEIGHTS, device="cpu")
    model.eval()
    print("Model loaded.")

    all_detections = []  # (score, timestamp, camera, frame_bgr, boxes, logits)

    for cam in CAMERAS:
        vid = os.path.join(CLIP_DIR, f"{cam}.mp4")
        if not os.path.exists(vid):
            print(f"  SKIP {vid} — not found")
            continue
        dets = run_camera(model, vid, cam)
        for score, t, frame, boxes, logits in dets:
            all_detections.append((score, t, cam, frame, boxes, logits))

    # ── sort globally and save top 6 ──────────────────────────────────────────
    all_detections.sort(key=lambda x: -x[0])

    print(f"\nTotal detection frames: {len(all_detections)}")
    print("Top results:")
    for i, (score, t, cam, _, boxes, _) in enumerate(all_detections[:10]):
        print(f"  #{i+1}  {cam}  t={t:.0f}s  score={score:.3f}  boxes={len(boxes)}")

    # Save top 6
    saved_paths = []
    for rank, (score, t, cam, frame, boxes, logits) in enumerate(all_detections[:6]):
        fig = draw_detection(frame, boxes, logits, t, cam)
        fname = os.path.join(OUT_DIR, f"rank{rank+1:02d}_{cam}_t{int(t):03d}s_score{score:.3f}.jpg")
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(fname)
        print(f"  Saved {os.path.basename(fname)}")

    # Also save all detection timestamps as a summary plot
    if all_detections:
        cams = list(set(d[2] for d in all_detections))
        fig2, axes = plt.subplots(len(cams), 1, figsize=(12, 4 * len(cams)), squeeze=False)
        for i, cam in enumerate(cams):
            cam_dets = [(d[1], d[0]) for d in all_detections if d[2] == cam]
            if cam_dets:
                ts, scores = zip(*cam_dets)
                axes[i][0].bar(ts, scores, width=0.7, color='steelblue')
                axes[i][0].axhline(y=BOX_THRESH, color='r', linestyle='--', label=f'thresh={BOX_THRESH}')
                axes[i][0].set_ylim(0, 1.0)
                axes[i][0].set_xlabel("time (s)")
                axes[i][0].set_ylabel("GDino score")
                axes[i][0].set_title(f"{cam} — detections in 985-1045s clip")
                axes[i][0].legend()
        plt.tight_layout()
        summary_path = os.path.join(OUT_DIR, "detection_timeline.png")
        fig2.savefig(summary_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Timeline saved: detection_timeline.png")

    print(f"\nDone. Outputs in: {OUT_DIR}")
    return all_detections

if __name__ == "__main__":
    main()
