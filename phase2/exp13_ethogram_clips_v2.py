"""
Experiment 13: Ethogram clip re-extraction with tight motion window.

Fixes from exp12:
  - "all day" / "morning and afternoon" events are SKIPPED (no anchor, not anchored to 10:00)
  - Scan window tightened to ±120s (was ±150s) to avoid drifting to adjacent events
  - Everything else identical to exp12

For every anchored event in 'data/Nity events.csv':
  1. Parse date + time (skip if "all day", unparseable, etc.)
  2. Find the Right_Top video segment on the server
  3. Download it
  4. Scan ±2 min around event time for peak motion (MOG2 with warm-up skip)
  5. Extract 15-second clip at peak
  6. DELETE the raw video immediately
  7. Record result in data/ethogram_clips_v2.json (saved after every event)

Fully resumable: events already in JSON are skipped.
"""

import os, csv, json, subprocess, re
import numpy as np
import cv2

PROJECT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR  = os.path.join(PROJECT, "data/clips/ethogram_v2")
JSON_OUT   = os.path.join(PROJECT, "data/ethogram_clips_v2.json")
TEMP_DIR   = os.path.join(PROJECT, "data/_tmp_download")
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

SESSIONS = [
    "O-vulgaris-Nity-2026-2-20--",
    "O-vulgaris-Nity-2025-9-17--",
]
BASE_URL   = "https://repo.octopus-intelligence.org/public"
USER, PASS = "octopus", "communication42"

SLACK_SEC     = 120    # ±2 min scan window (tighter than exp12's 150s)
WARM_UP_SEC   = 60     # skip first N sec of MOG2 (init artifact)
CLIP_DUR      = 15
BRIGHT_THRESH = 100
PROC_W, PROC_H = 1280, 720
TANK          = (130, 30, 870, 600)
TANK_AREA     = (TANK[2]-TANK[0]) * (TANK[3]-TANK[1])

SKIP_TIME_VALS = {"all day", "morning and afternoon"}


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_date(d_str):
    parts = d_str.strip().split("-")
    if len(parts) != 3: return None
    try: return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    except: return None


def parse_time(t_str):
    """Returns (seconds_since_midnight, note_str). note_str is None if exact."""
    t = t_str.strip()
    if not t:
        return None, "empty_time"
    if t.lower() in SKIP_TIME_VALS:
        return None, f"skipped_no_anchor:{t!r}"
    t = t.lstrip("~").strip()
    t = re.sub(r'\s*\(?[\?!]\)?\s*$', '', t).strip()
    if re.search(r'[?!+]', t):
        return None, f"unparseable:{t_str!r}"
    try:
        parts = [int(x) for x in t.split(":")]
        if len(parts) == 2: return parts[0]*3600 + parts[1]*60, None
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2], None
    except: pass
    try:
        return int(t) * 3600, None
    except: pass
    return None, f"unparseable:{t_str!r}"


# ── video helpers ─────────────────────────────────────────────────────────────

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
    fb = PROC_W*PROC_H*3; t = start; step = 1.0/fps
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb: break
            yield t, np.frombuffer(raw,dtype=np.uint8).reshape(PROC_H,PROC_W,3).copy()
            t += step
    finally:
        proc.kill(); proc.wait()


def get_frame(path, t):
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss",str(t),"-i",path,"-vframes","1",
           "-vf",f"scale={PROC_W}:{PROC_H}","-f","rawvideo","-pix_fmt","bgr24","-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read(); proc.wait()
    if len(raw) < PROC_W*PROC_H*3: return None
    return np.frombuffer(raw,dtype=np.uint8).reshape(PROC_H,PROC_W,3).copy()


def mean_brightness(frame):
    x1,y1,x2,y2 = TANK
    return float(cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY).mean())


# ── motion scoring ────────────────────────────────────────────────────────────

def score_motion(path, win_start, win_dur):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    x1,y1,x2,y2 = TANK
    t_arr, scores, bright_arr = [], [], []
    for t, frame in stream_window(path, win_start, win_dur):
        b = mean_brightness(frame)
        t_arr.append(t); bright_arr.append(b)
        if b < BRIGHT_THRESH:
            scores.append(0.0); continue
        roi = frame[y1:y2,x1:x2]
        fg  = fgbg.apply(roi)
        fg  = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  ko)
        fg  = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kc)
        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = max((cv2.contourArea(c) for c in cnts), default=0.0)
        scores.append(best / TANK_AREA)
    if not t_arr:
        return np.array([]), np.array([]), np.array([])
    return np.array(t_arr), np.array(scores), np.array(bright_arr)


def best_clip_start(t_arr, scores):
    if len(scores) == 0: return None, 0.0
    best_t, best_s = None, -1.0
    for t in t_arr:
        s = scores[(t_arr >= t) & (t_arr < t+CLIP_DUR)].sum()
        if s > best_s: best_s = s; best_t = float(t)
    return best_t, best_s


# ── server / download ─────────────────────────────────────────────────────────

def find_segment(date_str, event_sec):
    for session in SESSIONS:
        listing = f"{BASE_URL}/{session}/Right%20Top/Local/{date_str}/"
        auth    = listing.replace("https://", f"https://{USER}:{PASS}@")
        r = subprocess.run(["curl","-s",auth], capture_output=True, text=True)
        files = re.findall(r'href="([^"]+\.mp4)"', r.stdout)
        best_ts6 = best_off = None
        for fn in files:
            base = fn.split("--")[0]
            if len(base) != 6: continue
            try:
                hh,mm,ss = int(base[0:2]),int(base[2:4]),int(base[4:6])
                seg_start = hh*3600+mm*60+ss
            except: continue
            if seg_start <= event_sec <= seg_start+1800:
                off = event_sec - seg_start
                if best_ts6 is None or abs(off-900) < abs(best_off-900):
                    best_ts6 = base; best_off = off
        if best_ts6:
            return best_ts6, best_off, session
    return None, None, None


def download_video(date, seg, session):
    out_path = os.path.join(TEMP_DIR, f"{date}_{seg}_Right_Top.mp4")
    if os.path.exists(out_path) and video_duration(out_path) > 30:
        print(f"    Already in temp ({os.path.getsize(out_path)//1_000_000}MB)", flush=True)
        return out_path
    if os.path.exists(out_path):
        os.remove(out_path)
    url = f"{BASE_URL}/{session}/Right%20Top/Local/{date}/{seg}--vv-1.mp4"
    auth_url = url.replace("https://", f"https://{USER}:{PASS}@")
    print(f"    Downloading ...", flush=True)
    r = subprocess.run(
        ["ffmpeg","-loglevel","error","-y","-i",auth_url,
         "-c:v","copy","-c:a","copy",out_path],
        capture_output=True)
    if r.returncode == 0 and os.path.exists(out_path) and video_duration(out_path) > 30:
        print(f"    ✔ {os.path.getsize(out_path)//1_000_000}MB", flush=True)
        return out_path
    print(f"    ✗ download failed", flush=True)
    if os.path.exists(out_path): os.remove(out_path)
    return None


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json():
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT) as f: return json.load(f)
    return []


def save_json(results):
    with open(JSON_OUT, "w") as f: json.dump(results, f, indent=2)


def already_done(results, date, event):
    return any(r["date"] == date and r["event"] == event for r in results)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = load_json()
    print(f"Resuming: {len(results)} events already in JSON\n", flush=True)

    with open(os.path.join(PROJECT, "data/Nity events.csv")) as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        date_str = parse_date(row["Date"])
        event    = row["Event"].strip()
        cameras  = row.get("Cameras","").strip()
        details  = row.get("Details","").strip()
        time_raw = row.get("Time","").strip()

        print(f"\n[{idx+1}/{len(rows)}] {row['Date']}  {time_raw!r}  {event[:50]}", flush=True)

        key_date = date_str or row["Date"]
        if already_done(results, key_date, event):
            print(f"  Already done.", flush=True)
            continue

        def skip(reason):
            results.append({"date": key_date, "time": time_raw, "event": event,
                            "cameras": cameras, "details": details,
                            "status": "skipped", "skip_reason": reason})
            save_json(results)

        if date_str is None:
            skip("unparseable_date"); continue

        event_sec, time_note = parse_time(time_raw)
        if event_sec is None:
            skip(time_note); continue

        ts6, offset, session = find_segment(date_str, event_sec)
        if ts6 is None:
            skip("no_segment_on_server"); continue

        print(f"  Segment {ts6}  offset≈{offset:.0f}s  [{session[:30]}]", flush=True)

        safe = re.sub(r'[^\w]','_', event)[:30]
        clip_name = f"{date_str.replace('-','')}_{ts6}_{safe}.mp4"
        clip_path = os.path.join(CLIPS_DIR, clip_name)

        vid = download_video(date_str, ts6, session)
        if vid is None:
            skip("download_failed"); continue

        try:
            dur = video_duration(vid)

            t_check = min(max(offset, 5), dur-5)
            frame = get_frame(vid, t_check)
            if frame is not None and mean_brightness(frame) < BRIGHT_THRESH:
                print(f"  SKIP: dark (brightness={mean_brightness(frame):.0f})", flush=True)
                skip("dark_footage")
                continue

            win_start = max(0, offset - SLACK_SEC)
            win_end   = min(dur, offset + SLACK_SEC)
            print(f"  Scanning [{win_start:.0f}s – {win_end:.0f}s] ...", flush=True)
            t_arr, scores, bright_arr = score_motion(vid, win_start, win_end - win_start)

            if len(scores) == 0 or (bright_arr >= BRIGHT_THRESH).sum() == 0:
                skip("no_bright_frames"); continue

            scores[t_arr < win_start + WARM_UP_SEC] = 0.0
            scores[bright_arr < BRIGHT_THRESH]      = 0.0

            peak_score = float(scores.max()) if len(scores) > 0 else 0.0
            peak_t     = float(t_arr[scores.argmax()]) if peak_score > 0 else offset
            print(f"  Peak motion: {peak_score*100:.1f}% at t={peak_t:.0f}s", flush=True)

            clip_start, _ = best_clip_start(t_arr, scores)
            if clip_start is None: clip_start = max(0, offset - 7)
            clip_start = max(0, clip_start - 2)

            r = subprocess.run([
                "ffmpeg","-loglevel","error","-y",
                "-ss",str(clip_start),"-i",vid,
                "-t",str(CLIP_DUR),
                "-c:v","libx264","-crf","22","-preset","fast",
                clip_path], capture_output=True)

            if r.returncode == 0 and os.path.exists(clip_path) and os.path.getsize(clip_path) > 10_000:
                mb = os.path.getsize(clip_path)/1e6
                print(f"  ✔ {clip_name}  ({mb:.1f}MB)", flush=True)
                results.append({
                    "date": date_str, "time": time_raw, "event": event,
                    "cameras": cameras, "details": details,
                    "session": session, "segment": ts6,
                    "clip_path": f"data/clips/ethogram_v2/{clip_name}",
                    "clip_start_in_video_sec": round(clip_start, 1),
                    "motion_score": round(peak_score, 4),
                    "status": "extracted",
                    "time_note": time_note,
                })
            else:
                skip("ffmpeg_failed")
        finally:
            if os.path.exists(vid):
                os.remove(vid)
                print(f"  Deleted raw video.", flush=True)

        save_json(results)

    extracted = [r for r in results if r.get("status") == "extracted"]
    skipped   = [r for r in results if r.get("status") == "skipped"]
    reasons   = {}
    for r in skipped: reasons[r.get("skip_reason","?")] = reasons.get(r.get("skip_reason","?"),0)+1
    print(f"\n{'='*60}", flush=True)
    print(f"Done.  {len(extracted)} clips extracted,  {len(skipped)} skipped", flush=True)
    print(f"Skip reasons: {reasons}", flush=True)
    print(f"JSON:  {JSON_OUT}", flush=True)
    print(f"Clips: {CLIPS_DIR}", flush=True)


if __name__ == "__main__":
    main()
