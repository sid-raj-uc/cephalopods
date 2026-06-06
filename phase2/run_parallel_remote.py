#!/usr/bin/env python3
"""
Parallel remote scanner — scans all cameras for a video timestamp in parallel,
applies majority vote, downloads confirmed segments into one folder per timestamp.

Folder structure:
    data/aquarium/<date>/<video_name>/
        Left_Top_<start>_<end>.mp4
        Right_Back_<start>_<end>.mp4
        ...

Usage:
    python phase2/run_parallel_remote.py --dates 2026-02-20,2026-02-21 --target 15
    python phase2/run_parallel_remote.py --date 2026-02-20 --max-videos 5
"""

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_parallel")

ALL_CAMERAS = [
    "Left Top", "Right Back", "Right Front",
    "Right Left", "Right Right", "Right Top",
]
MAJORITY = 2  # cameras that must detect octopus to keep a video


def scan_one_camera(args_tuple):
    """
    Worker function: scan one camera for one video.
    Returns (camera_name, segments_list).
    Serialises CLIP inference via a shared lock.
    """
    url, camera, model, processor, text_features, device, threshold, min_dur, scan_fps, clip_lock = args_tuple

    from phase2.remote_scanner import scan_url
    from phase2.scanner import detect_segments

    try:
        # clip_lock passed into scan_url — only inference is serialized,
        # ffmpeg streaming for all cameras runs in true parallel
        timestamps, scores = scan_url(
            url, model, processor, text_features, device,
            scan_fps=scan_fps, size=224, batch_size=64,
            clip_lock=clip_lock,
        )
        segments = detect_segments(timestamps, scores, threshold, min_dur)
        log.info("  %-15s  %d segments", camera, len(segments))
        return camera, segments, timestamps, scores
    except Exception as e:
        log.warning("  %-15s  FAILED: %s", camera, e)
        return camera, [], None, None


def process_video_timestamp(
    date: str,
    video_name: str,
    camera_urls: dict[str, str],
    model, processor, text_features, device,
    threshold: float,
    min_dur: float,
    out_base: Path,
    clip_lock: threading.Lock,
    scan_fps: float = 0.2,
) -> int:
    """
    Scan all cameras in parallel, majority-vote, download confirmed segments.
    Returns number of segment files saved.
    """
    from phase2.remote_scanner import download_segment
    from phase2.scanner import save_scores

    log.info("▶  %s / %s", date, video_name)
    t0 = time.perf_counter()

    # ── Parallel scan across all cameras ──────────────────────────
    scan_args = [
        (url, cam, model, processor, text_features, device,
         threshold, min_dur, scan_fps, clip_lock)
        for cam, url in camera_urls.items()
        if url
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=len(scan_args)) as ex:
        futures = {ex.submit(scan_one_camera, a): a[1] for a in scan_args}
        for fut in as_completed(futures):
            cam, segs, ts, sc = fut.result()
            results[cam] = (segs, ts, sc)

    # ── Majority vote ──────────────────────────────────────────────
    cams_with_octopus = [c for c, (segs, _, _) in results.items() if segs]
    log.info(
        "  Majority vote: %d/%d cameras detected octopus  (%s)",
        len(cams_with_octopus), len(results),
        ", ".join(cams_with_octopus) or "none",
    )

    if len(cams_with_octopus) < MAJORITY:
        log.info("  ✗ Skipped — below majority threshold (%d)", MAJORITY)
        return 0

    # ── Save scores ────────────────────────────────────────────────
    for cam, (segs, ts, sc) in results.items():
        if ts is not None:
            score_key = f"remote_{date}_{cam.replace(' ', '_')}_{video_name}"
            save_scores(score_key, ts, sc)

    # ── Download segments from all cameras ─────────────────────────
    out_dir = out_base / date / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_jobs = []
    for cam, (segs, _, _) in results.items():
        url = camera_urls.get(cam)
        if not url or not segs:
            continue
        for start, end in segs:
            cam_tag = cam.replace(" ", "_")
            out_path = out_dir / f"{cam_tag}_{start:.0f}_{end:.0f}.mp4"
            dl_jobs.append((url, start, end, out_path))

    saved = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {
            ex.submit(download_segment, url, s, e, p): p
            for url, s, e, p in dl_jobs
            if not p.exists()
        }
        for fut in as_completed(futures):
            try:
                p = fut.result()
                saved += 1
                log.info("  ✓ %s  (%.1fMB)", p.name, p.stat().st_size / 1e6)
            except Exception as e:
                log.warning("  Download failed: %s", e)

    # count already-existing files
    saved += sum(1 for _, _, _, p in dl_jobs if p.exists() and p not in
                 [futures.get(f) for f in futures])

    log.info(
        "  ✔  %s done — %d files saved in %.1fs",
        video_name, saved, time.perf_counter() - t0,
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description="Parallel aquarium scanner with majority vote")
    parser.add_argument("--date",         type=str, help="Single date e.g. 2026-02-20")
    parser.add_argument("--dates",        type=str, help="Comma-separated dates")
    parser.add_argument("--threshold",    type=float, default=0.70)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--scan-fps",     type=float, default=0.2,  help="Frames/sec for scan (0.1 = 1 frame/10s, faster but coarser)")
    parser.add_argument("--max-videos",   type=int,   default=None, help="Max timestamps per day")
    parser.add_argument("--target",       type=int,   default=15,   help="Stop after N confirmed videos")
    parser.add_argument("--out-dir",      type=str,   default="data/aquarium")
    args = parser.parse_args()

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
    elif args.date:
        dates = [args.date]
    else:
        # default: scan a range of dates
        dates = [
            "2026-02-20", "2026-02-21", "2026-02-22",
            "2026-02-23", "2026-02-24", "2026-02-25",
        ]

    from phase2.scanner import load_clip
    from phase2.remote_scanner import list_camera_urls

    log.info("Loading CLIP ...")
    model, processor, text_features, device = load_clip()
    clip_lock = threading.Lock()

    out_base = Path(args.out_dir)
    confirmed_videos = 0
    t_total = time.perf_counter()

    for date in dates:
        if confirmed_videos >= args.target:
            break

        log.info("=" * 60)
        log.info("DATE: %s", date)

        camera_urls = list_camera_urls(date)

        # get video names from first available camera
        video_names = []
        for cam in ALL_CAMERAS:
            urls = camera_urls.get(cam, [])
            if urls:
                video_names = [Path(u).stem for u in urls]
                break

        if not video_names:
            log.warning("No videos found for %s", date)
            continue

        # skip early morning (before 09:00) — octopus typically inactive
        video_names = [v for v in video_names if v[:2] >= "09"]

        if args.max_videos:
            video_names = video_names[:args.max_videos]

        log.info("Found %d video timestamps for %s", len(video_names), date)

        for video_name in video_names:
            if confirmed_videos >= args.target:
                break

            # checkpointing — skip if folder already has mp4 segments
            out_dir = out_base / date / video_name
            if out_dir.exists():
                existing = list(out_dir.glob("*.mp4"))
                if existing:
                    log.info("  ✔ %s / %s  (%d files) — skipping", date, video_name, len(existing))
                    confirmed_videos += 1
                    continue
                if (out_dir / "_no_octopus.txt").exists():
                    log.info("  ✗ %s / %s  (no octopus marker) — skipping", date, video_name)
                    continue

            # match by HHMM prefix — cameras use different suffixes (--vv-1, --av-1)
            # and may differ by 1-2 seconds, so match on hour+minute only
            hhmm = video_name[:4]
            cam_urls = {}
            for cam in ALL_CAMERAS:
                urls = camera_urls.get(cam, [])
                match = [u for u in urls if Path(u).stem[:4] == hhmm]
                cam_urls[cam] = match[0] if match else None

            n_saved = process_video_timestamp(
                date, video_name, cam_urls,
                model, processor, text_features, device,
                args.threshold, args.min_duration,
                out_base, clip_lock,
                scan_fps=args.scan_fps,
            )

            if n_saved == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "_no_octopus.txt").write_text(f"{date}/{video_name}\n")

            if n_saved > 0:
                confirmed_videos += 1
                log.info(
                    "Progress: %d / %d confirmed videos",
                    confirmed_videos, args.target,
                )

    log.info("=" * 60)
    log.info(
        "DONE — %d confirmed videos saved in %.1fs",
        confirmed_videos, time.perf_counter() - t_total,
    )
    log.info("Output: %s", out_base.resolve())

    # print folder summary
    log.info("\nSaved folders:")
    for folder in sorted(out_base.rglob("*/*/"))[:30]:
        files = list(folder.glob("*.mp4"))
        if files:
            total_mb = sum(f.stat().st_size for f in files) / 1e6
            log.info("  %-50s  %d files  %.1fMB", str(folder), len(files), total_mb)


if __name__ == "__main__":
    main()
