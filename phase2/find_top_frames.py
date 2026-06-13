#!/usr/bin/env python3
"""
Run CLIP detection on one camera across multiple timestamps,
find the top-N frames with highest octopus confidence, and save them as images.

Usage:
    python phase2/find_top_frames.py --camera "Right Top" --top 20
    python phase2/find_top_frames.py --camera "Left Top" --top 20 --timestamps 095420,102421
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
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("find_top_frames")

VIDEO_BASE = Path("data/aquarium/full")
OUT_BASE   = Path("data/aquarium/top_frames")

ALL_TIMESTAMPS = ["095420", "102421", "112421", "122421", "132421"]


def extract_frame(video_path: Path, time_sec: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-ss", str(time_sec),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ], capture_output=True, text=True)
    return result.returncode == 0 and out_path.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",     default="Right Top")
    parser.add_argument("--date",       default="2026-02-20")
    parser.add_argument("--timestamps", default=",".join(ALL_TIMESTAMPS))
    parser.add_argument("--top",        type=int, default=20)
    parser.add_argument("--scan-fps",   type=float, default=0.2)
    args = parser.parse_args()

    timestamps = [t.strip() for t in args.timestamps.split(",")]
    cam_tag    = args.camera.replace(" ", "_")

    # ── CLIP scan across all timestamps ──────────────────────────────
    log.info("Loading CLIP …")
    from phase2.scanner import load_clip
    from phase2.remote_scanner import scan_url

    model, processor, text_features, device = load_clip()

    all_hits = []  # list of (score, timestamp_str, frame_sec, video_path)

    for ts in timestamps:
        video_path = VIDEO_BASE / args.date / ts / f"{cam_tag}.mp4"
        if not video_path.exists():
            log.warning("Missing: %s — skipping", video_path)
            continue

        log.info("Scanning %s / %s …", ts, args.camera)
        frame_ts, scores = scan_url(
            str(video_path), model, processor, text_features, device,
            username="", password="",
            scan_fps=args.scan_fps, size=224, batch_size=64,
        )

        log.info("  %s  frames=%d  max=%.3f @ t=%.0fs  mean=%.3f",
                 ts, len(scores), scores.max(),
                 frame_ts[scores.argmax()], scores.mean())

        for t, s in zip(frame_ts, scores):
            all_hits.append((float(s), ts, float(t), video_path))

    if not all_hits:
        log.error("No frames found.")
        sys.exit(1)

    # ── pick top-N across all videos ─────────────────────────────────
    all_hits.sort(key=lambda x: x[0], reverse=True)
    top = all_hits[:args.top]

    log.info("")
    log.info("Top %d frames across all %d timestamps:", args.top, len(timestamps))
    log.info("  %-6s  %-10s  %-8s  %s", "Rank", "Timestamp", "t (sec)", "Score")
    log.info("  " + "-" * 40)
    for i, (score, ts, t, _) in enumerate(top, 1):
        m, s = divmod(int(t), 60)
        log.info("  #%-5d  %-10s  %02d:%02d      %.4f", i, ts, m, s, score)

    # ── extract and save frames ───────────────────────────────────────
    out_dir = OUT_BASE / args.date / cam_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("Extracting frames → %s", out_dir)
    saved = 0
    for i, (score, ts, t, video_path) in enumerate(top, 1):
        m, s = divmod(int(t), 60)
        fname = f"rank{i:02d}_{ts}_t{int(t):04d}s_score{score:.3f}.jpg"
        out_path = out_dir / fname
        if extract_frame(video_path, t, out_path):
            log.info("  ✔ %s", fname)
            saved += 1
        else:
            log.warning("  ✗ failed: %s", fname)

    log.info("")
    log.info("Done — %d / %d frames saved → %s", saved, args.top, out_dir.resolve())


if __name__ == "__main__":
    main()
