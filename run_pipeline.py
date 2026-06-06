#!/usr/bin/env python3
"""
Cephalopod segmentation pipeline CLI.

Usage:
  python run_pipeline.py scrape   [--query Q] [--max-videos N] [--max-duration S]
  python run_pipeline.py extract  [--clip-duration S] [--fps N] [--size N]
  python run_pipeline.py segment  [--text-prompt T] [--box-thresh F] [--device D]
  python run_pipeline.py status
  python run_pipeline.py run      (scrape → extract → segment in one shot)
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def cmd_scrape(args):
    from pipeline.manifest import init_db
    from pipeline.scraper import download_pending, search_and_queue

    init_db()
    n = search_and_queue(args.query, max_videos=args.max_videos)
    print(f"Queued {n} videos.")
    download_pending(max_duration=args.max_duration)


def cmd_extract(args):
    from pipeline.extractor import extract_clips
    from pipeline.manifest import init_db

    init_db()
    extract_clips(
        clip_duration=args.clip_duration,
        fps=args.fps,
        size=args.size,
    )


def cmd_segment(args):
    from pipeline.manifest import init_db
    from pipeline.segmenter import segment_clips

    init_db()
    segment_clips(
        text_prompt=args.text_prompt,
        box_thresh=args.box_thresh,
        text_thresh=args.text_thresh,
        device=args.device,
    )


def cmd_status(_args):
    from pipeline.manifest import init_db, summary

    init_db()
    s = summary()
    print("\n--- Videos ---")
    for status, n in sorted(s["videos"].items()):
        print(f"  {status:<12} {n}")
    print("\n--- Clips ---")
    for status, n in sorted(s["clips"].items()):
        print(f"  {status:<12} {n}")
    print()


def cmd_run(args):
    cmd_scrape(args)
    cmd_extract(args)
    cmd_segment(args)


def main():
    parser = argparse.ArgumentParser(
        description="Cephalopod segmentation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- scrape ---
    p_scrape = sub.add_parser("scrape", help="Search YouTube and download videos")
    p_scrape.add_argument("--query",        default="octopus in ocean")
    p_scrape.add_argument("--max-videos",   type=int,   default=20)
    p_scrape.add_argument("--max-duration", type=float, default=600.0,
                          help="Skip videos longer than this many seconds")

    # --- extract ---
    p_extract = sub.add_parser("extract", help="Slice videos into frame clips")
    p_extract.add_argument("--clip-duration", type=float, default=5.0)
    p_extract.add_argument("--fps",           type=int,   default=8)
    p_extract.add_argument("--size",          type=int,   default=224)

    # --- segment ---
    p_seg = sub.add_parser("segment", help="Run Grounded SAM 2 on all clips")
    p_seg.add_argument("--text-prompt", default="octopus")
    p_seg.add_argument("--box-thresh",  type=float, default=0.3)
    p_seg.add_argument("--text-thresh", type=float, default=0.25)
    p_seg.add_argument("--device",      default=None,
                       help="cuda / cpu (auto-detected if omitted)")

    # --- status ---
    sub.add_parser("status", help="Print manifest counts")

    # --- run (all) ---
    p_run = sub.add_parser("run", help="Run all stages end-to-end")
    p_run.add_argument("--query",         default="octopus in ocean")
    p_run.add_argument("--max-videos",    type=int,   default=20)
    p_run.add_argument("--max-duration",  type=float, default=600.0)
    p_run.add_argument("--clip-duration", type=float, default=5.0)
    p_run.add_argument("--fps",           type=int,   default=8)
    p_run.add_argument("--size",          type=int,   default=224)
    p_run.add_argument("--text-prompt",   default="octopus")
    p_run.add_argument("--box-thresh",    type=float, default=0.3)
    p_run.add_argument("--text-thresh",   type=float, default=0.25)
    p_run.add_argument("--device",        default=None)

    args = parser.parse_args()
    {
        "scrape":  cmd_scrape,
        "extract": cmd_extract,
        "segment": cmd_segment,
        "status":  cmd_status,
        "run":     cmd_run,
    }[args.command](args)


if __name__ == "__main__":
    main()
