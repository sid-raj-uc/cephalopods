"""
Experiment 9: Motion-based Nity detection on Right_Top camera.

GDino failed because tank enrichment items (zebra toy) trigger false positives.
This experiment uses MOG2 background subtraction instead:
  - Static objects (toys, pipe, rocks) are absorbed into the background
  - Nity moving = large foreground blob = high-confidence detection

Pipeline per video:
  1. Sample at 1fps
  2. Skip dark frames (mean tank brightness < BRIGHT_THRESH=100)
  3. On bright frames: apply MOG2 background subtraction
  4. Filter foreground blobs by size (Nity is large — >1% of tank area)
  5. Report time windows with large moving blobs
"""

import os, sys, subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(PROJECT, "report/experiments/exp09_frames")
os.makedirs(OUT_DIR, exist_ok=True)

CAMERA        = "Right_Top"
PROC_W, PROC_H = 1280, 720
TANK          = (130, 30, 870, 600)    # tank interior x1,y1,x2,y2
SAMPLE_FPS    = 1.0
BRIGHT_THRESH = 100                    # skip dark frames
# Blob must cover at least this fraction of tank area to count as Nity
# Tank area = (870-130)*(600-30) = 740*570 = 421,800 px
# Nity ~47,500px = ~11% of tank. Use 2% as conservative lower bound
MIN_BLOB_FRAC = 0.02
TANK_AREA     = (TANK[2]-TANK[0]) * (TANK[3]-TANK[1])


def stream_frames(video_path, fps=1.0):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", video_path, "-vf", f"fps={fps},scale={PROC_W}:{PROC_H}",
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


def tank_brightness(frame):
    x1, y1, x2, y2 = TANK
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def scan_video(video_path, label):
    print(f"\n  Scanning {label} ...", flush=True)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=25, detectShadows=False
    )
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    t_arr, bright_arr, score_arr = [], [], []
    skipped_dark = 0
    x1, y1, x2, y2 = TANK

    for t, frame in stream_frames(video_path, fps=SAMPLE_FPS):
        b = tank_brightness(frame)
        t_arr.append(t)
        bright_arr.append(b)

        if b < BRIGHT_THRESH:
            score_arr.append(0.0)
            skipped_dark += 1
            # Still feed background subtractor so it learns
            fgbg.apply(frame[y1:y2, x1:x2])
            continue

        tank_roi = frame[y1:y2, x1:x2]
        fgmask = fgbg.apply(tank_roi)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

        # Find largest blob
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max((cv2.contourArea(c) for c in contours), default=0.0)
        score = max_area / TANK_AREA
        score_arr.append(score)

    t_arr     = np.array(t_arr)
    bright_arr = np.array(bright_arr)
    score_arr  = np.array(score_arr)

    n_bright   = (bright_arr >= BRIGHT_THRESH).sum()
    n_detected = (score_arr  >= MIN_BLOB_FRAC).sum()
    print(f"    {len(t_arr)} frames | {skipped_dark} dark (skipped) | "
          f"{n_bright} bright | {n_detected} motion events (blob≥{MIN_BLOB_FRAC*100:.0f}% tank)",
          flush=True)
    if n_detected > 0:
        det_t = t_arr[score_arr >= MIN_BLOB_FRAC]
        print(f"    Motion windows: {det_t[0]:.0f}s – {det_t[-1]:.0f}s", flush=True)
        print(f"    Peak blob={score_arr.max()*100:.1f}% at t={t_arr[score_arr.argmax()]:.0f}s",
              flush=True)

    return t_arr, bright_arr, score_arr


def save_top_frames(video_path, t_arr, score_arr, label, n=3):
    top_idx = np.argsort(-score_arr)
    saved = 0
    x1, y1, x2, y2 = TANK

    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    for idx in top_idx:
        if score_arr[idx] < MIN_BLOB_FRAC or saved >= n:
            break
        t = t_arr[idx]
        frame = get_frame(video_path, int(t))
        if frame is None:
            continue
        tank_roi = frame[y1:y2, x1:x2]
        fgmask = fgbg.apply(tank_roi)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

        out = frame.copy()
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) / TANK_AREA >= MIN_BLOB_FRAC:
                bx, by, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(out, (bx+x1, by+y1), (bx+bw+x1, by+bh+y1), (0,255,0), 2)
                pct = cv2.contourArea(c) / TANK_AREA * 100
                cv2.putText(out, f"blob {pct:.1f}%", (bx+x1, by+y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.rectangle(out, (x1,y1), (x2,y2), (255,100,0), 1)

        safe = label.replace("/","_").replace("-","")
        fname = os.path.join(OUT_DIR, f"{safe}_t{int(t):04d}s_blob{score_arr[idx]*100:.1f}pct.jpg")
        cv2.imwrite(fname, out)
        print(f"    Saved: {os.path.basename(fname)}", flush=True)
        saved += 1


def main():
    sessions = [("/tmp/feb27_Right_Top.mp4", "Feb27/093001")]
    for ts in ["095420","102421","112421","122421","132421"]:
        vp = os.path.join(PROJECT, f"data/aquarium/full/2026-02-20/{ts}/Right_Top.mp4")
        if os.path.exists(vp):
            sessions.append((vp, f"Feb20/{ts}"))

    all_results = {}
    for vid_path, label in sessions:
        t_arr, bright_arr, score_arr = scan_video(vid_path, label)
        all_results[label] = (t_arr, bright_arr, score_arr)
        save_top_frames(vid_path, t_arr, score_arr, label)

    # Timeline plot
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5*n), squeeze=False)
    for i, (label, (t_arr, bright_arr, score_arr)) in enumerate(all_results.items()):
        ax = axes[i][0]
        ax2 = ax.twinx()
        ax.plot(t_arr/60, score_arr*100, color='steelblue', lw=1.2, label='Blob % tank')
        ax.axhline(MIN_BLOB_FRAC*100, color='red', ls='--', lw=1, label=f'threshold={MIN_BLOB_FRAC*100:.0f}%')
        ax.fill_between(t_arr/60, 0, score_arr*100,
                        where=(score_arr>=MIN_BLOB_FRAC), color='red', alpha=0.25, label='motion event')
        ax2.plot(t_arr/60, bright_arr, color='orange', lw=0.8, alpha=0.5, label='brightness')
        ax2.axhline(BRIGHT_THRESH, color='orange', ls=':', lw=1)
        ax.set_title(f"{label} | bright_thresh={BRIGHT_THRESH} | blob_thresh={MIN_BLOB_FRAC*100:.0f}% tank")
        ax.set_xlabel("time (min)"); ax.set_ylabel("Blob size (% tank)", color='steelblue')
        ax2.set_ylabel("Brightness", color='orange')
        ax2.set_ylim(0, 255)
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, lab1+lab2, fontsize=8, loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp09_timelines.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)

    print("\nSummary:")
    for label, (t_arr, bright_arr, score_arr) in all_results.items():
        n_bright = (bright_arr >= BRIGHT_THRESH).sum()
        n_det    = (score_arr  >= MIN_BLOB_FRAC).sum()
        if n_bright == 0:
            status = "⬛ ALL DARK"
        elif n_det > 0:
            status = f"✅ MOTION DETECTED  peak={score_arr.max()*100:.1f}%"
        else:
            status = "❌ NO MOTION"
        print(f"  {label:25s}  bright={n_bright}/{len(t_arr)}  events={n_det}  {status}")
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
