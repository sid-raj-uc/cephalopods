#!/usr/bin/env python3
"""
Batch remote scanner — scan + download octopus segments across multiple
aquarium videos without downloading full files.

Usage:
    # single date, one camera
    python phase2/run_remote.py --date 2026-02-20 --camera "Left Top"

    # single date, all cameras
    python phase2/run_remote.py --date 2026-02-20

    # multiple dates, one camera
    python phase2/run_remote.py --dates 2026-02-20,2026-02-21,2026-02-22 --camera "Left Top"

    # limit videos per camera per day (useful for testing)
    python phase2/run_remote.py --date 2026-02-20 --camera "Left Top" --max-videos 3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_remote")

ALL_CAMERAS = [
    "Left Top",
    "Right Back",
    "Right Front",
    "Right Left",
    "Right Right",
    "Right Top",
]


def main():
    parser = argparse.ArgumentParser(description="Batch remote aquarium scanner")
    parser.add_argument("--date",          type=str, help="Single date  e.g. 2026-02-20")
    parser.add_argument("--dates",         type=str, help="Comma-separated dates  e.g. 2026-02-20,2026-02-21")
    parser.add_argument("--camera",        type=str, default=None, help="Camera name (default: all cameras)")
    parser.add_argument("--threshold",     type=float, default=0.6,  help="CLIP score threshold")
    parser.add_argument("--min-duration",  type=float, default=5.0,  help="Min segment length (s)")
    parser.add_argument("--scan-fps",      type=float, default=0.2,  help="Frames/sec to scan (default 0.2 = 1 frame/5s)")
    parser.add_argument("--max-videos",    type=int,   default=None, help="Max videos per camera per day")
    parser.add_argument("--out-dir",       type=str,   default="data/aquarium", help="Output directory")
    args = parser.parse_args()

    # resolve dates
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
    elif args.date:
        dates = [args.date]
    else:
        parser.error("Provide --date or --dates")

    cameras = [args.camera] if args.camera else ALL_CAMERAS

    log.info("Dates   : %s", dates)
    log.info("Cameras : %s", cameras)
    log.info("Output  : %s", args.out_dir)

    # load CLIP once
    from phase2.scanner import load_clip, detect_segments, save_scores, load_scores
    from phase2.remote_scanner import list_camera_urls, scan_url, download_segment
    from pathlib import Path as P

    SCORES_DIR = P("data/phase2/scores")
    out_dir    = P(args.out_dir)

    model, processor, text_features, device = load_clip()

    t_pipeline = time.perf_counter()
    summary    = []  # (date, camera, video, n_segments, dl_mb, status)

    for date in dates:
        log.info("=" * 60)
        log.info("Date: %s", date)

        camera_urls = list_camera_urls(date)

        for camera in cameras:
            urls = camera_urls.get(camera, [])
            if not urls:
                log.warning("No videos found for %s / %s", camera, date)
                continue

            if args.max_videos:
                urls = urls[:args.max_videos]

            log.info("-" * 40)
            log.info("Camera: %s  (%d videos)", camera, len(urls))

            for url in urls:
                video_name = Path(url).stem
                score_key  = f"remote_{date}_{camera.replace(' ', '_')}_{video_name}"
                scores_path = SCORES_DIR / f"{score_key}.npz"

                log.info("--- %s ---", video_name)
                t_vid = time.perf_counter()

                # ── Scan ──────────────────────────────────────────
                if scores_path.exists():
                    log.info("[CLIP]  Cached — loading scores")
                    from phase2.scanner import load_scores
                    timestamps, scores = load_scores(score_key)
                    t_scan = 0.0
                else:
                    t0 = time.perf_counter()
                    try:
                        timestamps, scores = scan_url(
                            url, model, processor, text_features, device,
                            scan_fps=args.scan_fps,
                        )
                        save_scores(score_key, timestamps, scores)
                        t_scan = time.perf_counter() - t0
                        log.info("[CLIP]  Scan done in %.1fs", t_scan)
                    except Exception as e:
                        log.warning("[CLIP]  Failed: %s", e)
                        summary.append((date, camera, video_name, 0, 0, f"scan failed: {e}"))
                        continue

                # ── Detect segments ───────────────────────────────
                segments = detect_segments(
                    timestamps, scores, args.threshold, args.min_duration
                )
                log.info("[CLIP]  %d segments detected", len(segments))

                if not segments:
                    summary.append((date, camera, video_name, 0, 0, "no octopus"))
                    continue

                # ── Download segments ─────────────────────────────
                cam_dir = out_dir / date / camera.replace(" ", "_")
                dl_mb   = 0.0
                n_dl    = 0

                for i, (start, end) in enumerate(segments):
                    out_path = cam_dir / f"{video_name}_{start:.0f}_{end:.0f}.mp4"
                    if out_path.exists():
                        dl_mb += out_path.stat().st_size / 1e6
                        n_dl  += 1
                        continue
                    try:
                        p = download_segment(url, start, end, out_path)
                        dl_mb += p.stat().st_size / 1e6
                        n_dl  += 1
                    except Exception as e:
                        log.warning("[DL]  Segment %d failed: %s", i, e)

                t_vid_total = time.perf_counter() - t_vid
                log.info(
                    "[VIDEO]  Done in %.1fs  —  %d segments  %.1fMB downloaded",
                    t_vid_total, n_dl, dl_mb,
                )
                summary.append((date, camera, video_name, n_dl, dl_mb, "done"))

    # ── Summary ───────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_pipeline
    log.info("=" * 60)
    log.info("COMPLETE  (%.1fs total)", total_elapsed)
    log.info("")
    log.info("%-12s  %-15s  %-22s  %8s  %8s  %s",
             "Date", "Camera", "Video", "Segments", "MB", "Status")
    log.info("-" * 85)

    total_segs = total_mb = 0
    for date, cam, vid, n_seg, mb, status in summary:
        log.info("%-12s  %-15s  %-22s  %8d  %8.1f  %s",
                 date, cam, vid, n_seg, mb, status)
        total_segs += n_seg
        total_mb   += mb

    log.info("-" * 85)
    log.info("%-12s  %-15s  %-22s  %8d  %8.1f",
             "TOTAL", "", "", total_segs, total_mb)


if __name__ == "__main__":
    main()
