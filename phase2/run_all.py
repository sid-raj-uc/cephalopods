#!/usr/bin/env python3
"""
Run the full phase2 pipeline (scan → extract) on all videos.

Usage:
    python phase2/run_all.py
    python phase2/run_all.py --threshold 0.65 --min-duration 5
    python phase2/run_all.py --rescan     # re-run scanner even if scores exist
    python phase2/run_all.py --reextract  # re-run extractor even if features exist
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ensure project root is on the path regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")

VIDEOS_DIR  = Path("data/videos")
SCORES_DIR  = Path("data/phase2/scores")
FEATURES_DIR = Path("data/phase2/features")


def main():
    parser = argparse.ArgumentParser(description="Phase2 pipeline: scan + extract all videos")
    parser.add_argument("--threshold",    type=float, default=0.6,  help="CLIP score threshold")
    parser.add_argument("--min-duration", type=float, default=5.0,  help="Min segment length (s)")
    parser.add_argument("--n-frames",     type=int,   default=16,   help="Frames sampled per segment")
    parser.add_argument("--motion-threshold", type=float, default=0.02, help="Min motion score to keep a segment")
    parser.add_argument("--rescan",       action="store_true",      help="Re-run scanner even if scores exist")
    parser.add_argument("--reextract",    action="store_true",      help="Re-run extractor even if features exist")
    args = parser.parse_args()

    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        log.error("No .mp4 files found in %s", VIDEOS_DIR)
        return

    log.info("Found %d videos", len(videos))

    # lazy-load models only when needed
    clip_model = clip_processor = clip_text_features = clip_device = None
    dino_model = dino_transform = dino_device = None

    t_pipeline = time.perf_counter()
    summary = []

    for video_path in videos:
        video_id = video_path.stem
        log.info("=" * 60)
        log.info("Processing: %s", video_id)
        t_video = time.perf_counter()

        scores_path   = SCORES_DIR   / f"{video_id}.npz"
        features_path = FEATURES_DIR / f"{video_id}.npz"

        # ── Scan ──────────────────────────────────────────────────
        if scores_path.exists() and not args.rescan:
            log.info("[CLIP]  Already scanned — loading cached scores")
            from phase2.scanner import load_scores
            timestamps, scores = load_scores(video_id)
            t_scan = 0.0
        else:
            if clip_model is None:
                from phase2.scanner import load_clip
                clip_model, clip_processor, clip_text_features, clip_device = load_clip()

            from phase2.scanner import scan_video, save_scores
            t0 = time.perf_counter()
            timestamps, scores = scan_video(
                str(video_path), clip_model, clip_processor, clip_text_features, clip_device
            )
            save_scores(video_id, timestamps, scores)
            t_scan = time.perf_counter() - t0
            log.info("[CLIP]  Scan done in %.1fs", t_scan)

        # ── Detect segments ───────────────────────────────────────
        from phase2.scanner import detect_segments
        segments = detect_segments(timestamps, scores, args.threshold, args.min_duration)
        log.info("[CLIP]  %d segments detected  (threshold=%.2f)", len(segments), args.threshold)

        if not segments:
            log.warning("No segments detected for %s — skipping", video_id)
            summary.append((video_id, 0, 0, 0, t_scan, 0.0, 0.0, "no segments"))
            continue

        n_before_motion = len(segments)

        # ── Motion filter ─────────────────────────────────────────
        from phase2.extractor import filter_by_motion
        t0 = time.perf_counter()
        segments, motion_scores_list = filter_by_motion(
            segments, str(video_path),
            threshold=args.motion_threshold,
            n_frames=args.n_frames,
        )
        t_motion = time.perf_counter() - t0
        log.info(
            "[MOTION]  %d / %d segments kept  (%.1fs, threshold=%.3f)",
            len(segments), n_before_motion, t_motion, args.motion_threshold,
        )

        if not segments:
            log.warning("All segments were still — skipping %s", video_id)
            summary.append((video_id, n_before_motion, 0, 0, t_scan, t_motion, 0.0, "all still"))
            continue

        # ── Extract ───────────────────────────────────────────────
        if features_path.exists() and not args.reextract:
            log.info("[DINO]  Features already exist — skipping extraction")
            from phase2.extractor import load_features
            feats, _, _ = load_features(video_id)
            t_video_total = time.perf_counter() - t_video
            log.info("[VIDEO TOTAL]  %.1fs  (scan=cached, motion=%.1fs, extract=cached)", t_motion)
            summary.append((video_id, n_before_motion, len(segments), len(feats), t_scan, t_motion, 0.0, "cached"))
            continue

        if dino_model is None:
            from phase2.extractor import load_dinov2
            dino_model, dino_transform, dino_device = load_dinov2()

        from phase2.extractor import extract_video
        t0 = time.perf_counter()
        extract_video(
            str(video_path), video_id, segments,
            dino_model, dino_transform, dino_device,
            n_frames=args.n_frames,
        )
        t_extract = time.perf_counter() - t0
        t_video_total = time.perf_counter() - t_video

        log.info("[DINO]    Extraction done in %.1fs", t_extract)
        log.info(
            "[VIDEO TOTAL]  %.1fs  —  scan=%.1fs  motion=%.1fs  extract=%.1fs",
            t_video_total, t_scan, t_motion, t_extract,
        )
        summary.append((video_id, n_before_motion, len(segments), len(segments), t_scan, t_motion, t_extract, "done"))

    # ── Summary ───────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_pipeline
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE  (%.1fs total)", total_elapsed)
    log.info("")
    log.info(
        "%-25s  %8s  %8s  %8s  %8s  %8s  %8s  %s",
        "Video", "Segs", "Motion", "Vectors", "Scan(s)", "Motion(s)", "Extract(s)", "Status",
    )
    log.info("-" * 95)
    total_vectors = 0
    for vid, n_before, n_after, n_vec, t_sc, t_mo, t_ex, status in summary:
        log.info(
            "%-25s  %8d  %8d  %8d  %8.1f  %9.1f  %10.1f  %s",
            vid, n_before, n_after, n_vec, t_sc, t_mo, t_ex, status,
        )
        total_vectors += n_vec
    log.info("-" * 95)
    log.info(
        "%-25s  %8d  %8d  %8d  %8.1f  %9.1f  %10.1f",
        "TOTAL",
        sum(s[1] for s in summary),
        sum(s[2] for s in summary),
        total_vectors,
        sum(s[4] for s in summary),
        sum(s[5] for s in summary),
        sum(s[6] for s in summary),
    )


if __name__ == "__main__":
    main()
