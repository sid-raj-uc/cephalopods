#!/usr/bin/env python3
"""
Clip extractor — cuts time slots from locally downloaded 30-min aquarium videos
and saves one folder per time slot containing all camera clips.

Modes:
  --slots   : explicit start-end pairs, e.g. --slots 950,1100 1300,1400
  --auto    : detect windows from saved .npz score files (CLIP + motion peaks)
  --cameras : which cameras to include (default: all 6)

Output structure:
    data/aquarium/clips/<date>/<video_timestamp>/<start>_<end>/
        Left_Top.mp4
        Right_Back.mp4
        ...

Usage:
    # Extract a specific window from all cameras
    python phase2/extract_clips.py --date 2026-02-20 --timestamp 095420 --slots 950,1100

    # Auto-detect interesting windows from score files
    python phase2/extract_clips.py --date 2026-02-20 --timestamp 095420 --auto

    # Auto-detect, reliable cameras only
    python phase2/extract_clips.py --date 2026-02-20 --timestamp 095420 --auto --cameras "Left Top,Right Front"
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_clips")

ALL_CAMERAS = [
    "Left Top", "Right Back", "Right Front",
    "Right Left", "Right Right", "Right Top",
]

VIDEO_DIR  = Path("data/aquarium/full")
SCORES_DIR = Path("data/aquarium/full")   # same base — results are next to videos
CLIPS_DIR  = Path("data/aquarium/clips")


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_right_right(camera: str) -> bool:
    return camera.lower().replace(" ", "") == "rightright"


def _video_path(date: str, timestamp: str, camera: str) -> Path:
    return VIDEO_DIR / date / timestamp / f"{camera.replace(' ', '_')}.mp4"


def _results_path(date: str, timestamp: str, camera: str) -> Path:
    return SCORES_DIR / date / timestamp / f"{camera.replace(' ', '_')}_results.npz"


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# ── auto-detection ─────────────────────────────────────────────────────────────

def _find_peaks(scores: np.ndarray, timestamps: np.ndarray, threshold: float, min_gap: float) -> list[float]:
    """
    Return timestamps of local peaks above threshold, separated by at least min_gap seconds.
    Picks the highest score in each cluster rather than every crossing.
    """
    above = np.where(scores >= threshold)[0]
    if len(above) == 0:
        return []

    peaks = []
    cluster_start = above[0]
    cluster_end   = above[0]

    for idx in above[1:]:
        if timestamps[idx] - timestamps[cluster_end] <= min_gap:
            cluster_end = idx
        else:
            best = cluster_start + np.argmax(scores[cluster_start:cluster_end + 1])
            peaks.append(float(timestamps[best]))
            cluster_start = cluster_end = idx

    best = cluster_start + np.argmax(scores[cluster_start:cluster_end + 1])
    peaks.append(float(timestamps[best]))
    return peaks


def detect_slots(
    date: str,
    timestamp: str,
    cameras: list[str],
    clip_threshold: float = 0.65,
    motion_threshold: float = 0.40,
    clip_weight: float = 0.7,
    motion_weight: float = 0.3,
    pad_sec: float = 20.0,
    min_gap_sec: float = 120.0,
) -> list[tuple[float, float]]:
    """
    Find tight windows around score peaks across cameras.

    Strategy:
      1. For each camera, find local CLIP peaks above clip_threshold.
      2. Optionally boost peaks where motion is also high at the same time.
      3. Merge peak times that are within min_gap_sec of each other.
      4. Pad each merged peak by pad_sec on each side → final (start, end) slot.
    """
    all_peak_times = []

    for camera in cameras:
        path = _results_path(date, timestamp, camera)
        if not path.exists():
            log.warning("No results file for %s — skipping in auto-detect", camera)
            continue

        data = np.load(path)
        clip_ts     = data["clip_timestamps"]
        clip_scores = data["clip_scores"]
        mot_ts      = data["motion_timestamps"]
        mot_scores  = data["motion_scores"]

        # interpolate motion onto clip timestamps for combined scoring
        if len(mot_ts) > 0:
            mot_interp = np.interp(clip_ts, mot_ts, mot_scores)
            combined   = clip_weight * clip_scores + motion_weight * mot_interp
        else:
            combined = clip_scores

        peaks = _find_peaks(combined, clip_ts, clip_threshold * clip_weight, min_gap_sec)
        if peaks:
            log.info("  %-14s  %d peak(s): %s", camera, len(peaks),
                     ", ".join(f"t={_fmt(p)}" for p in peaks))
        all_peak_times.extend(peaks)

    if not all_peak_times:
        log.warning("No peaks found above thresholds — nothing to extract")
        return []

    # merge peak times that are close together across cameras
    all_peak_times.sort()
    merged = []
    cluster = [all_peak_times[0]]

    for t in all_peak_times[1:]:
        if t - cluster[-1] <= min_gap_sec:
            cluster.append(t)
        else:
            merged.append(np.mean(cluster))
            cluster = [t]
    merged.append(np.mean(cluster))

    slots = [(max(0, t - pad_sec), t + pad_sec) for t in merged]

    log.info("Auto-detected %d slot(s):", len(slots))
    for s, e in slots:
        log.info("  t=%s – %s  (%.0f–%.0fs, %.0fs long)", _fmt(s), _fmt(e), s, e, e - s)

    return slots


# ── extraction ─────────────────────────────────────────────────────────────────

def extract_clip(
    src: Path,
    start: float,
    end: float,
    out: Path,
) -> bool:
    """Cut [start, end] from src and write to out. Returns True on success."""
    if out.exists() and out.stat().st_size > 10_000:
        log.info("  ✔ exists: %s", out.name)
        return True

    out.parent.mkdir(parents=True, exist_ok=True)

    audio_flag = ["-c:a", "aac"] if _is_right_right(src.stem) else ["-c:a", "copy"]
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(src),
        "-c:v", "copy",
        *audio_flag,
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        log.error("  ✗ %s: %s", out.name, result.stderr[-300:])
        return False

    mb = out.stat().st_size / 1e6
    log.info("  ✔ %s  %.1fMB", out.name, mb)
    return True


def extract_slot(
    date: str,
    timestamp: str,
    start: float,
    end: float,
    cameras: list[str],
    clips_base: Path,
) -> Path:
    """
    Extract [start, end] from all cameras and save into one folder.
    Returns the output folder path.
    """
    slot_tag = f"{int(start)}_{int(end)}"
    out_dir  = clips_base / date / timestamp / slot_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Extracting t=%s–%s  →  %s", _fmt(start), _fmt(end), out_dir)

    ok = 0
    for camera in cameras:
        src = _video_path(date, timestamp, camera)
        if not src.exists():
            log.warning("  ✗ %s: source video not found", camera)
            continue

        out = out_dir / f"{camera.replace(' ', '_')}.mp4"
        if extract_clip(src, start, end, out):
            ok += 1

    log.info("Saved %d / %d camera clips → %s", ok, len(cameras), out_dir)
    return out_dir


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",      default="2026-02-20")
    parser.add_argument("--timestamp", default="095420")
    parser.add_argument("--out-dir",   default=str(CLIPS_DIR))
    parser.add_argument(
        "--cameras",
        default=",".join(ALL_CAMERAS),
        help="Comma-separated camera names, e.g. 'Left Top,Right Front'",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--slots", nargs="+", metavar="START,END",
        help="Explicit time slots as start,end pairs in seconds, e.g. 950,1100 1300,1400",
    )
    mode.add_argument(
        "--auto", action="store_true",
        help="Auto-detect interesting windows from saved score files",
    )

    # auto-detect tuning
    parser.add_argument("--clip-threshold",   type=float, default=0.50)
    parser.add_argument("--motion-threshold", type=float, default=0.20)
    parser.add_argument("--pad",              type=float, default=30.0,
                        help="Padding in seconds around each detected hit")
    parser.add_argument("--min-gap",          type=float, default=60.0,
                        help="Minimum gap to keep slots separate (seconds)")

    args = parser.parse_args()

    cameras   = [c.strip() for c in args.cameras.split(",")]
    clips_base = Path(args.out_dir)

    # ── resolve time slots ────────────────────────────────────────────
    if args.auto:
        slots = detect_slots(
            args.date, args.timestamp, cameras,
            clip_threshold=args.clip_threshold,
            motion_threshold=args.motion_threshold,
            pad_sec=args.pad,
            min_gap_sec=args.min_gap,
        )
    else:
        slots = []
        for s in args.slots:
            parts = s.split(",")
            if len(parts) != 2:
                log.error("Bad slot format (expected start,end): %s", s)
                sys.exit(1)
            slots.append((float(parts[0]), float(parts[1])))

    if not slots:
        log.error("No slots to extract.")
        sys.exit(1)

    # ── extract ───────────────────────────────────────────────────────
    out_dirs = []
    for start, end in slots:
        d = extract_slot(args.date, args.timestamp, start, end, cameras, clips_base)
        out_dirs.append(d)

    log.info("=" * 60)
    log.info("Done — %d slot(s) extracted", len(out_dirs))
    for d in out_dirs:
        files = sorted(d.glob("*.mp4"))
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        log.info("  %s  (%d cameras, %.1f MB)", d, len(files), total_mb)


if __name__ == "__main__":
    main()
