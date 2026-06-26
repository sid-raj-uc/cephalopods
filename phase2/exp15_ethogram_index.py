"""
Experiment 15: Multi-camera ethogram video index.

Reads data/Nity events.csv and produces data/ethogram_index.json —
one entry per ethogram row mapping the event to its video URLs across
all available cameras, with the event's offset within that video.

No downloading, no clip extraction, no VLM.
confidence_score and caption fields are null placeholders for future runs.

Fully resumable: events already in the JSON are skipped on re-run.
"""

import csv, json, re, os
import requests
from urllib.parse import quote
from pathlib import Path

PROJECT  = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT / "data" / "Nity events.csv"
OUT_PATH = PROJECT / "data" / "ethogram_index.json"

BASE_URL = "https://repo.octopus-intelligence.org/public"
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from server_creds import USER, PASS  # creds from env / .env, not hardcoded
SESSION_2025 = "O-vulgaris-Nity-2025-9-17--"
SESSION_2026 = "O-vulgaris-Nity-2026-2-20--"

# Cameras confirmed to have data per session (Left Left / Left Right / Left Top
# are present as folders in the 2025 session but are empty — excluded).
CAMERAS = {
    SESSION_2025: ["Left Back", "Left Front", "Right Back", "Right Front",
                   "Right Left", "Right Right", "Right Top"],
    SESSION_2026: ["Left Top", "Right Back", "Right Front",
                   "Right Left", "Right Right", "Right Top"],
}

SKIP_TIME_VALS = {"all day", "morning and afternoon"}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_date(raw: str):
    """'2026-4-7' → '2026-04-07', or None on failure."""
    raw = raw.strip()
    parts = raw.split("-")
    if len(parts) != 3:
        return None
    try:
        return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    except ValueError:
        return None


def parse_time(raw: str):
    """
    Returns seconds-since-midnight (int) or None if unparseable.
    Handles: 'HH:MM', '~HH:MM', 'HH:MM (?)', 'HH:MM:SS'.
    Skips: 'all day', 'morning and afternoon', '17??++', empty.
    """
    t = raw.strip().lower()
    if not t or t in SKIP_TIME_VALS:
        return None
    t = t.lstrip("~").strip()
    t = re.sub(r"\s*\(\?\)\s*$", "", t).strip()
    if re.search(r"[?]", t):
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", t)
    if not m:
        return None
    h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return h * 3600 + mn * 60 + s


def session_for_date(date_str: str) -> str:
    """'2026-04-07' → session name."""
    return SESSION_2026 if date_str >= "2026-02-20" else SESSION_2025


def hhmmss_to_sec(s: str) -> int:
    """'123002' → 45602 (seconds since midnight)."""
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def list_directory(session: str, camera: str, date: str) -> dict:
    """
    Fetch the camera/date directory once.
    Returns {"filenames": [...full mp4 names...], "segments": [...HHMMSS...]}
    or {"filenames": [], "segments": []} on failure.
    """
    url = f"{BASE_URL}/{quote(session)}/{quote(camera)}/Local/{date}/"
    try:
        r = requests.get(url, auth=(USER, PASS), timeout=15)
        if r.status_code != 200:
            return {"filenames": [], "segments": []}
        filenames = re.findall(r"(\d{6}--[av]v-1\.mp4)", r.text)
        segments  = sorted(set(fn[:6] for fn in filenames))
        return {"filenames": sorted(filenames), "segments": segments}
    except Exception:
        return {"filenames": [], "segments": []}


def find_segment(segments: list[str], event_sec: int):
    """
    Find the segment whose 30-min window contains event_sec.
    Returns (hhmmss_str, offset_sec) or (None, None).
    """
    for seg in reversed(segments):          # descending → first one ≤ event
        seg_sec = hhmmss_to_sec(seg)
        if seg_sec <= event_sec:
            offset = event_sec - seg_sec
            if offset < 1800:               # must be within the 30-min window
                return seg, offset
    return None, None


def build_url(session: str, camera: str, date: str, seg: str, filenames: list[str]) -> str:
    """Pick the matching filename for this segment and return the full URL."""
    matches = [fn for fn in filenames if fn.startswith(seg)]
    filename = matches[0] if matches else f"{seg}--vv-1.mp4"
    return f"{BASE_URL}/{quote(session)}/{quote(camera)}/Local/{date}/{filename}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load existing index for resumability
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            index = json.load(f)
    else:
        index = []

    done_keys = {(e["date"], e["time"], e.get("event", "")) for e in index}

    session = requests.Session()
    session.auth = (USER, PASS)

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} ethogram rows …\n")

    new_entries = 0

    for row in rows:
        date_raw  = row.get("Date", "").strip()
        time_raw  = row.get("Time", "").strip()
        event     = row.get("Event", "").strip()
        cameras_col = row.get("Cameras", "").strip()
        details   = row.get("Details", "").strip()

        date = parse_date(date_raw)
        if date is None:
            print(f"  SKIP (bad date): {date_raw!r}")
            continue

        key = (date, time_raw, event)
        if key in done_keys:
            print(f"  already indexed: {date} {time_raw} — {event[:40]}")
            continue

        event_sec = parse_time(time_raw)

        if event_sec is None:
            entry = {
                "date": date, "time": time_raw, "event": event,
                "cameras_col": cameras_col, "details": details,
                "status": "skipped", "skip_reason": "no_timestamp",
                "cameras": [],
            }
            print(f"  SKIPPED (no timestamp): {date} {time_raw!r} — {event[:40]}")
        else:
            sess = session_for_date(date)
            cam_list = CAMERAS[sess]

            camera_entries = []
            for cam in cam_list:
                listing = list_directory(sess, cam, date)
                if not listing["segments"]:
                    camera_entries.append({
                        "name": cam, "available": False,
                        "reason": "date_not_found",
                    })
                    continue

                seg, offset = find_segment(listing["segments"], event_sec)
                if seg is None:
                    camera_entries.append({
                        "name": cam, "available": False,
                        "reason": "segment_not_found",
                    })
                    continue

                video_url = build_url(sess, cam, date, seg, listing["filenames"])
                camera_entries.append({
                    "name": cam,
                    "available": True,
                    "video_url": video_url,
                    "event_offset_sec": offset,
                    "confidence_score": None,
                    "caption": None,
                })

            # Also mark cameras not in this session
            all_known = set(CAMERAS[SESSION_2025]) | set(CAMERAS[SESSION_2026])
            for cam in sorted(all_known - set(cam_list)):
                camera_entries.append({
                    "name": cam, "available": False,
                    "reason": "not_in_session",
                })

            available = [c for c in camera_entries if c.get("available")]
            entry = {
                "date": date, "time": time_raw, "event": event,
                "cameras_col": cameras_col, "details": details,
                "status": "indexed",
                "session": sess,
                "cameras": sorted(camera_entries, key=lambda c: c["name"]),
            }
            print(f"  indexed: {date} {time_raw} — {event[:40]} "
                  f"({len(available)}/{len(cam_list)} cameras available)")

        index.append(entry)
        done_keys.add(key)
        new_entries += 1

        # Save after every entry (resumable)
        with open(OUT_PATH, "w") as f:
            json.dump(index, f, indent=2)

    print(f"\nDone. {new_entries} new entries. Total: {len(index)}. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
