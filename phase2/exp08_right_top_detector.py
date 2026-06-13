"""
Experiment 8: Nity detection on Right_Top camera with brightness filtering.

Pipeline per video:
  1. Sample at 0.5fps
  2. Skip dark frames (mean tank brightness < BRIGHT_THRESH=100)
  3. On bright frames: run GroundingDINO on tank-interior crop
  4. Detection = at least one "octopus" box with conf >= CONF_THRESH in tank region
  5. Report time windows where Nity is detected

Only reports high-confidence detections. No detections on dark footage.
"""

import os, sys, subprocess, re
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'GroundingDINO'))
from groundingdino.util.inference import load_model, predict

PROJECT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(PROJECT, "report/experiments/exp08_frames")
os.makedirs(OUT_DIR, exist_ok=True)

# Camera settings
CAMERA   = "Right_Top"
PROC_W, PROC_H = 1280, 720
# Tank interior region in 1280x720 (excludes brackets, scissor handle, frame edges)
TANK = (130, 30, 870, 600)   # x1, y1, x2, y2
# Confirmed Nity zone (from manual verification 2026-02-27)
# Only count GDino detections whose centre falls inside this box
NITY_ZONE = (450, 100, 780, 360)  # x1, y1, x2, y2

SAMPLE_FPS   = 0.1      # 1 frame every 10 seconds
BRIGHT_THRESH = 100     # skip frames below this mean brightness
CONF_THRESH   = 0.30    # GDino confidence threshold


def stream_frames(video_path, fps=0.5):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", video_path,
           "-vf", f"fps={fps},scale={PROC_W}:{PROC_H}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = PROC_W * PROC_H * 3
    t, step = 0.0, 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield t, np.frombuffer(raw, dtype=np.uint8).reshape(PROC_H, PROC_W, 3).copy()
            t += step
    finally:
        proc.kill(); proc.wait()


def get_frame(video_path, t_sec):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", str(t_sec), "-i", video_path, "-vframes", "1",
           "-vf", f"scale={PROC_W}:{PROC_H}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read(); proc.wait()
    if len(raw) < PROC_W * PROC_H * 3:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(PROC_H, PROC_W, 3).copy()


def brightness(frame):
    x1, y1, x2, y2 = TANK
    tank = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(tank.mean())


def run_gdino(model, frame):
    """Run GDino on tank interior crop. Returns (max_conf, boxes_in_fullframe)."""
    x1, y1, x2, y2 = TANK
    crop = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    transform = T.Compose([
        T.Resize(800), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil)
    boxes, logits, phrases = predict(
        model=model, image=tensor, caption="octopus",
        box_threshold=CONF_THRESH, text_threshold=0.15, device="cpu"
    )
    if len(boxes) == 0:
        return 0.0, []

    CH, CW = crop.shape[:2]
    full_boxes = []
    for b, l in zip(boxes, logits):
        cx, cy, w, h = b.tolist()
        fx1 = int((cx - w / 2) * CW) + x1
        fy1 = int((cy - h / 2) * CH) + y1
        fx2 = int((cx + w / 2) * CW) + x1
        fy2 = int((cy + h / 2) * CH) + y1
        # Only keep detections whose centre falls inside the confirmed Nity zone
        fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        nx1, ny1, nx2, ny2 = NITY_ZONE
        if nx1 <= fcx <= nx2 and ny1 <= fcy <= ny2:
            full_boxes.append((float(l), fx1, fy1, fx2, fy2))

    if not full_boxes:
        return 0.0, []
    return max(l for l, *_ in full_boxes), full_boxes


def scan_video(model, video_path, label):
    print(f"\n  Scanning {label} ...")
    t_arr, bright_arr, conf_arr = [], [], []
    skipped_dark = 0

    for t, frame in stream_frames(video_path, fps=SAMPLE_FPS):
        b = brightness(frame)
        bright_arr.append(b)
        t_arr.append(t)

        if b < BRIGHT_THRESH:
            conf_arr.append(0.0)
            skipped_dark += 1
            continue

        max_conf, _ = run_gdino(model, frame)
        conf_arr.append(max_conf)

    t_arr = np.array(t_arr)
    bright_arr = np.array(bright_arr)
    conf_arr = np.array(conf_arr)

    n_bright = (bright_arr >= BRIGHT_THRESH).sum()
    n_detected = (conf_arr >= CONF_THRESH).sum()
    print(f"    {len(t_arr)} frames | {skipped_dark} dark (skipped) | "
          f"{n_bright} bright | {n_detected} detected (conf≥{CONF_THRESH})")
    if n_detected > 0:
        det_times = t_arr[conf_arr >= CONF_THRESH]
        print(f"    Detection windows: {det_times[0]:.0f}s – {det_times[-1]:.0f}s")
        print(f"    Peak conf={conf_arr.max():.3f} at t={t_arr[conf_arr.argmax()]:.0f}s")

    return t_arr, bright_arr, conf_arr


def save_top_frames(model, video_path, t_arr, conf_arr, label, n=3):
    top_idx = np.argsort(-conf_arr)
    saved = 0
    for idx in top_idx:
        if conf_arr[idx] < CONF_THRESH or saved >= n:
            break
        t = t_arr[idx]
        frame = get_frame(video_path, int(t))
        if frame is None:
            continue
        _, boxes = run_gdino(model, frame)
        out = frame.copy()
        # draw Nity zone in blue
        nx1, ny1, nx2, ny2 = NITY_ZONE
        cv2.rectangle(out, (nx1, ny1), (nx2, ny2), (255, 100, 0), 1)
        for conf, x1, y1, x2, y2 in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"octopus {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        safe = label.replace("/", "_").replace("-", "")
        fname = os.path.join(OUT_DIR, f"{safe}_t{int(t):04d}s_conf{conf_arr[idx]:.2f}.jpg")
        cv2.imwrite(fname, out)
        print(f"    Saved: {os.path.basename(fname)}")
        saved += 1


def main():
    print("Loading GroundingDINO ...")
    model = load_model(
        os.path.join(PROJECT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        os.path.join(PROJECT, "weights/groundingdino_swint_ogc.pth"),
    )
    print("  Loaded.\n")

    # Sessions to scan — add more as needed
    sessions = [
        # (video_path, label)
        ("/tmp/feb27_Right_Top.mp4", "Feb27/093001"),
    ]

    # Also scan any Feb 20 Right_Top sessions that exist locally
    for ts in ["095420", "102421", "112421", "122421", "132421"]:
        vp = os.path.join(PROJECT, f"data/aquarium/full/2026-02-20/{ts}/Right_Top.mp4")
        if os.path.exists(vp):
            sessions.append((vp, f"Feb20/{ts}"))

    all_results = {}
    for vid_path, label in sessions:
        t_arr, bright_arr, conf_arr = scan_video(model, vid_path, label)
        all_results[label] = (t_arr, bright_arr, conf_arr)
        save_top_frames(model, vid_path, t_arr, conf_arr, label)

    # Timeline plot
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), squeeze=False)
    for i, (label, (t_arr, bright_arr, conf_arr)) in enumerate(all_results.items()):
        ax = axes[i][0]
        ax2 = ax.twinx()
        ax.plot(t_arr / 60, conf_arr, color='steelblue', lw=1.2, label='GDino conf')
        ax.axhline(CONF_THRESH, color='red', ls='--', lw=1, label=f'threshold={CONF_THRESH}')
        ax.fill_between(t_arr / 60, 0, conf_arr, where=(conf_arr >= CONF_THRESH),
                        color='red', alpha=0.25, label='detected')
        ax2.plot(t_arr / 60, bright_arr, color='orange', lw=0.8, alpha=0.6, label='brightness')
        ax2.axhline(BRIGHT_THRESH, color='orange', ls=':', lw=1)
        ax.set_ylim(0, 1.05)
        ax2.set_ylim(0, 255)
        ax.set_title(f"{label} | bright_thresh={BRIGHT_THRESH} | conf_thresh={CONF_THRESH}")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("GDino conf", color='steelblue')
        ax2.set_ylabel("brightness", color='orange')
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc='upper right')
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "exp08_timelines.png")
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSummary:")
    for label, (t_arr, bright_arr, conf_arr) in all_results.items():
        n_bright = (bright_arr >= BRIGHT_THRESH).sum()
        n_det = (conf_arr >= CONF_THRESH).sum()
        status = "✅ DETECTED" if n_det > 0 else ("⬛ ALL DARK" if n_bright == 0 else "❌ NOT FOUND")
        print(f"  {label:25s}  bright={n_bright}/{len(t_arr)}  detected={n_det}  {status}")
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
