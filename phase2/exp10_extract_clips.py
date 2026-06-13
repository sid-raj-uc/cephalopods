"""
Experiment 10: Extract confirmed Nity activity clips using CSV events.

For each CSV event:
  1. Map event time → offset in the Right_Top video segment
  2. Extract a wide window (±SLACK minutes) around that offset
  3. Run MOG2 motion detection at 2fps on that window
  4. Find the peak activity 15-second segment
  5. Check brightness — skip if dark
  6. Save the best 15s clip as MP4

This handles CSV timing drift automatically — we don't need exact times,
just a rough anchor and let motion detection find the real peak.
"""

import os, sys, subprocess, csv
import numpy as np
import cv2

PROJECT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(PROJECT, "data/aquarium/full")
CLIPS_DIR = os.path.join(PROJECT, "data/clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

SESSIONS     = [
    "O-vulgaris-Nity-2026-2-20--",   # Feb 20 2026 onwards
    "O-vulgaris-Nity-2025-9-17--",   # fallback for earlier dates
]
BASE_URL     = "https://repo.octopus-intelligence.org/public"
USER, PASS   = "octopus", "communication42"
YEAR_FILTER  = 2026   # only process events from this year

SLACK_SEC    = 150    # ±2.5 min window around CSV event time
WARM_UP_SEC  = 60    # discard first N seconds of MOG2 scores (initialization artifact)
CLIP_DURATION = 15   # final clip length in seconds
BRIGHT_THRESH = 100  # skip dark footage
PROC_W, PROC_H = 1280, 720
TANK          = (130, 30, 870, 600)
TANK_AREA     = (TANK[2]-TANK[0]) * (TANK[3]-TANK[1])


# ── video helpers ─────────────────────────────────────────────────────────────

def video_duration(path):
    r = subprocess.run(
        ["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", path],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except:
        return 0.0


def get_frame(path, t_sec):
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", str(t_sec), "-i", path, "-vframes","1",
           "-vf", f"scale={PROC_W}:{PROC_H}",
           "-f","rawvideo","-pix_fmt","bgr24","-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read(); proc.wait()
    if len(raw) < PROC_W*PROC_H*3:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(PROC_H, PROC_W, 3).copy()


def stream_window(path, start_sec, duration_sec, fps=2.0):
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", str(start_sec), "-i", path,
           "-t", str(duration_sec),
           "-vf", f"fps={fps},scale={PROC_W}:{PROC_H}",
           "-f","rawvideo","-pix_fmt","bgr24","-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = PROC_W * PROC_H * 3
    t = start_sec
    step = 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield t, np.frombuffer(raw, dtype=np.uint8).reshape(PROC_H, PROC_W, 3).copy()
            t += step
    finally:
        proc.kill(); proc.wait()


def mean_brightness(frame):
    x1,y1,x2,y2 = TANK
    return float(cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY).mean())


# ── motion scoring ─────────────────────────────────────────────────────────────

def score_window(video_path, window_start, window_dur):
    """Score each frame in [window_start, window_start+window_dur] by motion blob size."""
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    x1,y1,x2,y2 = TANK

    t_arr, scores, bright_arr = [], [], []
    for t, frame in stream_window(video_path, window_start, window_dur, fps=2.0):
        b = mean_brightness(frame)
        bright_arr.append(b)
        t_arr.append(t)
        if b < BRIGHT_THRESH:
            scores.append(0.0)
            continue
        roi = frame[y1:y2, x1:x2]
        fg = fgbg.apply(roi)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_blob = max((cv2.contourArea(c) for c in cnts), default=0.0)
        scores.append(max_blob / TANK_AREA)

    return np.array(t_arr), np.array(scores), np.array(bright_arr)


def find_best_clip_start(t_arr, scores, clip_dur=CLIP_DURATION):
    """Find the clip_dur window with highest total motion score."""
    if len(scores) == 0:
        return None
    best_start, best_sum = None, -1.0
    for i, t in enumerate(t_arr):
        # sum scores within [t, t+clip_dur]
        mask = (t_arr >= t) & (t_arr < t + clip_dur)
        s = scores[mask].sum()
        if s > best_sum:
            best_sum = s
            best_start = float(t)
    return best_start, best_sum


# ── download helper ───────────────────────────────────────────────────────────

def find_and_download_right_top(date, ts6, session):
    """Download Right_Top for date/ts6 if not already present. Returns local path or None."""
    import re
    out_dir  = os.path.join(DATA_DIR, date, ts6)
    out_path = os.path.join(out_dir, "Right_Top.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        return out_path

    listing = f"{BASE_URL}/{session}/Right%20Top/Local/{date}/"
    auth    = listing.replace("https://", f"https://{USER}:{PASS}@")
    r = subprocess.run(["curl","-s",auth], capture_output=True, text=True)
    files = re.findall(r'href="([^"]+\.mp4)"', r.stdout)
    hhmm  = ts6[:4]
    match = [f for f in files if f[:4] == hhmm]
    if not match:
        print(f"    No Right_Top file for {date}/{ts6} in {session}", flush=True)
        return None

    url      = f"{listing}{match[0]}"
    auth_url = url.replace("https://", f"https://{USER}:{PASS}@")
    os.makedirs(out_dir, exist_ok=True)
    print(f"    Downloading Right_Top for {date}/{ts6} ...", flush=True)
    r2 = subprocess.run(
        ["ffmpeg","-loglevel","error","-y","-i",auth_url,"-c:v","copy","-c:a","copy",out_path],
        capture_output=True, text=True
    )
    if r2.returncode == 0 and os.path.getsize(out_path) > 1_000_000:
        print(f"    ✔ {os.path.getsize(out_path)//1_000_000}MB", flush=True)
        return out_path
    print(f"    ✗ download failed: {r2.stderr[-100:]}", flush=True)
    return None


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_time_hms(t_str):
    """Parse 'HH:MM' or 'HH:MM:SS' → total seconds. Returns None if not parseable."""
    t_str = t_str.strip()
    try:
        parts = [int(x) for x in t_str.split(":")]
        if len(parts) == 2:
            return parts[0]*3600 + parts[1]*60
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
    except:
        pass
    return None


def parse_date_ymd(d_str):
    """Parse '2026-3-7' or '2026-03-07' → 'YYYY-MM-DD'."""
    parts = d_str.strip().split("-")
    if len(parts) != 3:
        return None
    return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def find_segment_for_event(date_str, event_sec):
    """
    Find which 30-min Right_Top segment on the server contains event_sec (wall clock).
    Tries all sessions in order. Returns (ts6, offset_in_video, session) or (None, None, None).
    """
    import re
    for session in SESSIONS:
        listing = f"{BASE_URL}/{session}/Right%20Top/Local/{date_str}/"
        auth    = listing.replace("https://", f"https://{USER}:{PASS}@")
        r = subprocess.run(["curl","-s",auth], capture_output=True, text=True)
        files = re.findall(r'href="([^"]+\.mp4)"', r.stdout)
        if not files:
            continue

        best_ts6 = None
        best_offset = None
        for fn in files:
            base = fn.split("--")[0]
            if len(base) != 6:
                continue
            try:
                hh, mm, ss = int(base[0:2]), int(base[2:4]), int(base[4:6])
                seg_start = hh*3600 + mm*60 + ss
            except:
                continue
            if seg_start <= event_sec <= seg_start + 1800:
                offset = event_sec - seg_start
                if best_ts6 is None or abs(offset - 900) < abs(best_offset - 900):
                    best_ts6    = base
                    best_offset = offset

        if best_ts6 is not None:
            return best_ts6, best_offset, session

    return None, None, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    csv_path = os.path.join(PROJECT, "data/Nity events.csv")

    # Parse CSV — only use rows with Right camera and parseable time
    events = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            date_str = parse_date_ymd(row["Date"])
            t_sec    = parse_time_hms(row["Time"])
            camera   = row.get("Cameras","").strip().lower()
            if date_str is None or t_sec is None:
                continue
            if "right" not in camera:
                continue
            events.append({
                "date": date_str,
                "event_sec": t_sec,
                "event": row["Event"].strip(),
                "details": row["Details"].strip(),
            })

    # Filter to target year
    events = [e for e in events if e["date"].startswith(str(YEAR_FILTER))]
    print(f"Found {len(events)} events in {YEAR_FILTER} with Right camera and parseable time\n", flush=True)

    for ev in events:
        date_str  = ev["date"]
        event_sec = ev["event_sec"]
        label     = ev["event"] or "activity"
        print(f"{'='*60}", flush=True)
        print(f"  {date_str}  {event_sec//3600:02d}:{(event_sec%3600)//60:02d}  {label}", flush=True)

        # Find which video segment covers this event
        ts6, offset, session = find_segment_for_event(date_str, event_sec)
        if ts6 is None:
            print(f"  SKIP: no matching segment on server", flush=True)
            continue
        print(f"  Segment: {ts6}  offset≈{offset:.0f}s  session={session}", flush=True)

        # Download Right_Top if needed
        vid_path = find_and_download_right_top(date_str, ts6, session)
        if vid_path is None:
            continue

        dur = video_duration(vid_path)
        if dur < 10:
            print(f"  SKIP: video too short ({dur:.0f}s)", flush=True)
            continue

        # Quick brightness check at event offset
        t_check = min(offset, dur - 5)
        frame_check = get_frame(vid_path, t_check)
        if frame_check is not None and mean_brightness(frame_check) < BRIGHT_THRESH:
            print(f"  SKIP: dark footage (brightness={mean_brightness(frame_check):.0f})", flush=True)
            continue

        # Score motion in ±SLACK window around offset
        win_start = max(0, offset - SLACK_SEC)
        win_end   = min(dur, offset + SLACK_SEC)
        win_dur   = win_end - win_start
        print(f"  Scanning motion in [{win_start:.0f}s – {win_end:.0f}s] ...", flush=True)

        t_arr, scores, bright_arr = score_window(vid_path, win_start, win_dur)

        if (bright_arr >= BRIGHT_THRESH).sum() == 0:
            print(f"  SKIP: all dark in window", flush=True)
            continue

        # Discard dark frames and MOG2 warm-up period from scoring
        scores[bright_arr < BRIGHT_THRESH] = 0.0
        warm_mask = t_arr < (win_start + WARM_UP_SEC)
        scores[warm_mask] = 0.0

        peak_score = scores.max()
        print(f"  Peak motion: {peak_score*100:.1f}% tank at t={t_arr[scores.argmax()]:.0f}s", flush=True)

        result = find_best_clip_start(t_arr, scores, clip_dur=CLIP_DURATION)
        if result is None or result[1] <= 0:
            print(f"  SKIP: no motion detected in window", flush=True)
            continue

        clip_start, clip_score = result
        clip_start = max(0, clip_start - 2)   # 2s lead-in
        print(f"  Best clip window: {clip_start:.0f}s  (score={clip_score:.3f})", flush=True)

        # Extract final clip
        safe_label = label[:30].replace(" ","_").replace("/","").replace(",","")
        out_name = f"{date_str.replace('-','')}_{ts6}_{safe_label}.mp4"
        out_path = os.path.join(CLIPS_DIR, out_name)

        r = subprocess.run([
            "ffmpeg","-loglevel","error","-y",
            "-ss", str(clip_start), "-i", vid_path,
            "-t", str(CLIP_DURATION),
            "-c:v","libx264","-crf","22","-preset","fast",
            out_path
        ], capture_output=True, text=True)

        if r.returncode == 0 and os.path.exists(out_path):
            mb = os.path.getsize(out_path) / 1e6
            print(f"  ✔ Saved: {out_name}  ({mb:.1f}MB)", flush=True)
        else:
            print(f"  ✗ ffmpeg failed: {r.stderr[-200:]}", flush=True)

    print(f"\nDone. Clips in: {CLIPS_DIR}", flush=True)


if __name__ == "__main__":
    main()
