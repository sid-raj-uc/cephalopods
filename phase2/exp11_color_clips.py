"""
Experiment 11: Extract color clips from April 7 2026 Right_Top session.

Apr 7 has color footage from ~12:42 through 18:30+.
We pick one segment per hour, scan for the peak motion window,
and extract a 15-second color clip.
"""

import os, subprocess
import numpy as np
import cv2

PROJECT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(PROJECT, "data/aquarium/full")
CLIPS_DIR = os.path.join(PROJECT, "data/clips/color")
os.makedirs(CLIPS_DIR, exist_ok=True)

SESSION  = "O-vulgaris-Nity-2026-2-20--"
BASE_URL = "https://repo.octopus-intelligence.org/public"
USER, PASS = "octopus", "communication42"

PROC_W, PROC_H = 1280, 720
TANK       = (130, 30, 870, 600)
TANK_AREA  = (TANK[2]-TANK[0]) * (TANK[3]-TANK[1])
CLIP_DUR   = 15
WARM_UP    = 60    # seconds to skip at start before scoring (MOG2 init)
BRIGHT_MIN = 100
SAT_MIN    = 5     # saturation threshold — below this = B&W/IR frame

# Color segments on Apr 7 2026 — one per ~hour slot to get variety
# Format: (segment_file_base, label, scan_start, scan_end)
# scan_start/end = seconds within the segment to scan (None = full segment)
TARGET_SEGMENTS = [
    ("2026-04-07", "190003", "19:00_activity"),
    ("2026-04-07", "193003", "19:30_activity"),
]


def video_duration(path):
    r = subprocess.run(
        ["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0",path],
        capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except: return 0.0


def stream_window(path, start, dur, fps=2.0):
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss",str(start),"-i",path,"-t",str(dur),
           "-vf",f"fps={fps},scale={PROC_W}:{PROC_H}",
           "-f","rawvideo","-pix_fmt","bgr24","-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fb = PROC_W * PROC_H * 3
    t = start
    step = 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb: break
            yield t, np.frombuffer(raw, dtype=np.uint8).reshape(PROC_H, PROC_W, 3).copy()
            t += step
    finally:
        proc.kill(); proc.wait()


def mean_brightness(frame):
    x1,y1,x2,y2 = TANK
    return float(cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY).mean())


def mean_saturation(frame):
    x1,y1,x2,y2 = TANK
    hsv = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2HSV)
    return float(hsv[:,:,1].mean())


def score_motion(path, win_start, win_dur):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    x1,y1,x2,y2 = TANK
    t_arr, scores, sat_arr = [], [], []
    for t, frame in stream_window(path, win_start, win_dur):
        t_arr.append(t)
        sat_arr.append(mean_saturation(frame))
        if mean_brightness(frame) < BRIGHT_MIN:
            scores.append(0.0); continue
        roi = frame[y1:y2,x1:x2]
        fg = fgbg.apply(roi)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  ko)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kc)
        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = max((cv2.contourArea(c) for c in cnts), default=0.0)
        scores.append(best / TANK_AREA)
    return np.array(t_arr), np.array(scores), np.array(sat_arr)


def best_clip_start(t_arr, scores, clip_dur=CLIP_DUR):
    best_t, best_s = None, -1.0
    for t in t_arr:
        mask = (t_arr >= t) & (t_arr < t + clip_dur)
        s = scores[mask].sum()
        if s > best_s:
            best_s = s; best_t = float(t)
    return best_t, best_s


def download(date, seg):
    out_dir  = os.path.join(DATA_DIR, date, seg)
    out_path = os.path.join(out_dir, "Right_Top.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"    Already cached.", flush=True)
        return out_path
    url = f"{BASE_URL}/{SESSION}/Right%20Top/Local/{date}/{seg}--vv-1.mp4"
    auth_url = url.replace("https://", f"https://{USER}:{PASS}@")
    os.makedirs(out_dir, exist_ok=True)
    print(f"    Downloading ...", flush=True)
    r = subprocess.run(
        ["ffmpeg","-loglevel","error","-y","-i",auth_url,"-c:v","copy","-c:a","copy",out_path],
        capture_output=True)
    if r.returncode == 0 and os.path.getsize(out_path) > 1_000_000:
        print(f"    ✔ {os.path.getsize(out_path)//1_000_000}MB", flush=True)
        return out_path
    print(f"    ✗ download failed", flush=True)
    return None


def main():
    for date, seg, label in TARGET_SEGMENTS:
        print(f"\n{'='*60}", flush=True)
        print(f"  {date}  {seg}  [{label}]", flush=True)

        vid = download(date, seg)
        if vid is None: continue

        dur = video_duration(vid)
        print(f"  Duration: {dur:.0f}s", flush=True)
        if dur < 30:
            print(f"  SKIP: video too short or unreadable ({dur:.0f}s)", flush=True)
            continue

        # For Apr 1 narrow color window, limit scan to t=480–720s
        if date == "2026-04-01" and seg == "120002":
            win_start, win_dur = 480, 240
        else:
            win_start, win_dur = 0, dur

        print(f"  Scanning [{win_start:.0f}s – {win_start+win_dur:.0f}s] ...", flush=True)
        t_arr, scores, sat_arr = score_motion(vid, win_start, win_dur)

        # Check color coverage
        color_frac = (sat_arr > SAT_MIN).mean()
        print(f"  Color frames: {color_frac*100:.0f}%", flush=True)
        if color_frac < 0.3:
            print(f"  SKIP: <30% color frames in window", flush=True)
            continue

        # Zero out warm-up and B&W frames
        scores[t_arr < win_start + WARM_UP] = 0.0
        scores[sat_arr <= SAT_MIN] = 0.0

        if scores.max() == 0:
            print(f"  SKIP: no motion in color window", flush=True)
            continue

        peak_t = t_arr[scores.argmax()]
        print(f"  Peak motion: {scores.max()*100:.1f}% at t={peak_t:.0f}s", flush=True)

        clip_start, clip_score = best_clip_start(t_arr, scores)
        clip_start = max(win_start, clip_start - 2)
        print(f"  Best clip: t={clip_start:.0f}s  (score={clip_score:.3f})", flush=True)

        out_name = f"{date.replace('-','')}_{seg}_{label}.mp4"
        out_path = os.path.join(CLIPS_DIR, out_name)
        r = subprocess.run([
            "ffmpeg","-loglevel","error","-y",
            "-ss",str(clip_start),"-i",vid,
            "-t",str(CLIP_DUR),
            "-c:v","libx264","-crf","22","-preset","fast",
            out_path], capture_output=True)
        if r.returncode == 0 and os.path.exists(out_path):
            mb = os.path.getsize(out_path)/1e6
            print(f"  ✔ Saved: {out_name}  ({mb:.1f}MB)", flush=True)
        else:
            print(f"  ✗ ffmpeg failed", flush=True)

    print(f"\nDone. Color clips in: {CLIPS_DIR}", flush=True)


if __name__ == "__main__":
    main()
