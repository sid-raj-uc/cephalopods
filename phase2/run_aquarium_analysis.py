#!/usr/bin/env python3
"""
Download one full 30-min aquarium recording (all 6 cameras) and run
CLIP octopus detection + motion detection on each camera.

Steps:
  1. Download full 30-min MP4 from all 6 cameras
  2. CLIP scan at 0.2 fps  → octopus probability over time
  3. Motion scan at 1 fps  → activity / movement over time
  4. Save .npz results + PNG plot per camera + combined summary plot

Usage:
    python phase2/run_aquarium_analysis.py
    python phase2/run_aquarium_analysis.py --date 2026-02-20 --timestamp 095420
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aquarium_analysis")

BASE_URL  = "https://repo.octopus-intelligence.org/public"
USER      = "octopus"
PASS      = "communication42"
SESSION   = "O-vulgaris-Nity-2026-2-20--"

ALL_CAMERAS = [
    "Left Top", "Right Back", "Right Front",
    "Right Left", "Right Right", "Right Top",
]

CLIP_THRESHOLD = 0.70
MOTION_HIGH    = 0.40   # motion score considered "active"


# ── helpers ───────────────────────────────────────────────────────────────────

def _auth_url(url: str) -> str:
    return url.replace("https://", f"https://{USER}:{PASS}@")


def _find_camera_video(camera: str, date: str, hhmm: str) -> str | None:
    """
    List the camera's directory on the server and return the full URL of
    the video whose filename starts with `hhmm`. Handles --vv-1 / --av-1
    and the ±1 second offset between cameras.
    """
    import re
    import urllib.parse

    cam_enc    = urllib.parse.quote(camera)
    listing_url = f"{BASE_URL}/{SESSION}/{cam_enc}/Local/{date}/"
    auth_listing = listing_url.replace("https://", f"https://{USER}:{PASS}@")

    result = subprocess.run(
        ["curl", "-s", auth_listing],
        capture_output=True, text=True,
    )
    filenames = re.findall(r'href="([^"]+\.mp4)"', result.stdout)
    for fn in filenames:
        if fn[:4] == hhmm:
            return f"{listing_url}{fn}"
    return None


def _is_right_right(camera: str) -> bool:
    return camera.lower().replace(" ", "") == "rightright"


# ── step 1: download ──────────────────────────────────────────────────────────

def download_full_video(camera: str, date: str, hhmm: str, out_dir: Path) -> Path | None:
    """Download the full 30-min MP4 for one camera. Returns path or None on failure."""
    cam_tag  = camera.replace(" ", "_")
    out_path = out_dir / f"{cam_tag}.mp4"

    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        log.info("  ✔ %s already downloaded (%.0fMB)", cam_tag, out_path.stat().st_size / 1e6)
        return out_path

    video_url = _find_camera_video(camera, date, hhmm)
    if video_url is None:
        log.error("  ✗ %s: no file matching HHMM=%s on %s", cam_tag, hhmm, date)
        return None
    log.info("  Resolved %s → %s", cam_tag, video_url.split("/")[-1])
    url = _auth_url(video_url)

    # Right_Right uses pcm_alaw audio — copy video, re-encode audio to aac
    audio_flag = ["-c:a", "aac"] if _is_right_right(camera) else ["-c:a", "copy"]

    cmd = [
        "ffmpeg", "-loglevel", "error", "-stats", "-y",
        "-i", url,
        "-c:v", "copy",
        *audio_flag,
        str(out_path),
    ]

    log.info("  Downloading %s …", cam_tag)
    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1_000_000:
        log.error("  ✗ %s download failed:\n%s", cam_tag, result.stderr[-500:])
        return None

    mb  = out_path.stat().st_size / 1e6
    log.info("  ✔ %s  %.0fMB in %.0fs", cam_tag, mb, time.perf_counter() - t0)
    return out_path


def download_all_cameras(date: str, hhmm: str, out_dir: Path) -> dict[str, Path]:
    """Download all 6 cameras in parallel. Returns {camera: path}."""
    log.info("Downloading 6 cameras for %s / %s …", date, hhmm)
    results: dict[str, Path | None] = {}
    lock = threading.Lock()

    def _dl(camera):
        p = download_full_video(camera, date, hhmm, out_dir)
        with lock:
            results[camera] = p

    threads = [threading.Thread(target=_dl, args=(c,), daemon=True) for c in ALL_CAMERAS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok  = {c: p for c, p in results.items() if p is not None}
    log.info("Downloads complete: %d / %d cameras", len(ok), len(ALL_CAMERAS))
    return ok


# ── step 2 & 3: scan ──────────────────────────────────────────────────────────

def run_clip_scan(video_path: Path, model, processor, text_features, device) -> tuple[np.ndarray, np.ndarray]:
    # Use ffmpeg pipe at 0.2fps (same as remote scanner) — 30x faster than cv2 seeking
    from phase2.remote_scanner import scan_url
    return scan_url(
        str(video_path), model, processor, text_features, device,
        username="", password="",  # no auth for local files
        scan_fps=0.2, size=224, batch_size=64,
    )


def run_motion_scan(video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from phase2.motion_detector import scan_motion
    return scan_motion(str(video_path), fps=1.0, smooth_window=5)


# ── step 4: save + plot ───────────────────────────────────────────────────────

def save_results(
    out_dir: Path,
    camera: str,
    clip_ts: np.ndarray, clip_scores: np.ndarray,
    mot_ts: np.ndarray,  mot_scores: np.ndarray,
):
    cam_tag = camera.replace(" ", "_")
    np.savez_compressed(
        out_dir / f"{cam_tag}_results.npz",
        clip_timestamps=clip_ts, clip_scores=clip_scores,
        motion_timestamps=mot_ts, motion_scores=mot_scores,
    )
    log.info("  Saved results → %s", out_dir / f"{cam_tag}_results.npz")


def plot_camera(
    ax,
    camera: str,
    clip_ts: np.ndarray, clip_scores: np.ndarray,
    mot_ts: np.ndarray,  mot_scores: np.ndarray,
):
    ax2 = ax.twinx()
    ax.plot(clip_ts / 60, clip_scores, color="steelblue", lw=1.2, label="CLIP (octopus)")
    ax.axhline(CLIP_THRESHOLD, color="steelblue", ls="--", lw=0.8, alpha=0.6)
    ax2.plot(mot_ts / 60, mot_scores, color="darkorange", lw=1.0, alpha=0.8, label="Motion")
    ax2.axhline(MOTION_HIGH, color="darkorange", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax.set_ylabel("CLIP P(octopus)", color="steelblue", fontsize=8)
    ax2.set_ylabel("Motion score", color="darkorange", fontsize=8)
    ax.set_title(camera, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (min)", fontsize=8)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")


def plot_all_cameras(all_results: dict, date: str, video_name: str, out_path: Path):
    cameras = list(all_results.keys())
    n = len(cameras)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()

    for i, camera in enumerate(cameras):
        clip_ts, clip_scores, mot_ts, mot_scores = all_results[camera]
        plot_camera(axes[i], camera, clip_ts, clip_scores, mot_ts, mot_scores)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Aquarium Analysis — {date} / {video_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved combined plot → %s", out_path)


def plot_overlap_summary(all_results: dict, date: str, video_name: str, out_path: Path):
    """Plot per-second P(octopus AND motion) across cameras."""
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.tab10.colors

    for i, (camera, (clip_ts, clip_scores, mot_ts, mot_scores)) in enumerate(all_results.items()):
        # align motion to clip timestamps via nearest-neighbour
        if len(mot_ts) == 0:
            continue
        mot_interp = np.interp(clip_ts, mot_ts, mot_scores)
        combined = clip_scores * mot_interp
        ax.plot(clip_ts / 60, combined, lw=1.0, color=colors[i % 10],
                alpha=0.7, label=camera)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("CLIP × Motion")
    ax.set_title(f"CLIP × Motion score per camera — {date} / {video_name}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved overlap plot → %s", out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",      default="2026-02-20")
    parser.add_argument("--timestamp", default="095420",
                        help="HHMMSS prefix, e.g. 095420")
    parser.add_argument("--out-dir",       default="data/aquarium/full")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download videos, skip CLIP and motion scans")
    args = parser.parse_args()

    date     = args.date
    hhmm     = args.timestamp[:4]
    out_base = Path(args.out_dir) / date / args.timestamp
    out_base.mkdir(parents=True, exist_ok=True)

    # ── step 1: download ──────────────────────────────────────────────
    cam_paths = download_all_cameras(date, hhmm, out_base)
    if not cam_paths:
        log.error("No cameras downloaded — aborting.")
        sys.exit(1)

    if args.download_only:
        log.info("Download-only mode — done.")
        log.info("Output: %s", out_base.resolve())
        return

    # ── step 2 & 3: scan each camera ─────────────────────────────────
    log.info("Loading CLIP …")
    from phase2.scanner import load_clip
    model, processor, text_features, device = load_clip()

    all_results: dict[str, tuple] = {}

    for camera, video_path in cam_paths.items():
        log.info("Analysing %s …", camera)

        log.info("  CLIP scan …")
        clip_ts, clip_scores = run_clip_scan(video_path, model, processor, text_features, device)

        log.info("  Motion scan …")
        mot_ts, mot_scores = run_motion_scan(video_path)

        save_results(out_base, camera, clip_ts, clip_scores, mot_ts, mot_scores)
        all_results[camera] = (clip_ts, clip_scores, mot_ts, mot_scores)

        # quick per-camera summary
        clip_hits = int((clip_scores >= CLIP_THRESHOLD).sum())
        mot_hits  = int((mot_scores  >= MOTION_HIGH).sum())
        log.info(
            "  %-12s  CLIP hits: %d/%d (≥%.2f)  |  Motion hits: %d/%d (≥%.2f)",
            camera, clip_hits, len(clip_scores), CLIP_THRESHOLD,
            mot_hits,  len(mot_scores),  MOTION_HIGH,
        )

    # ── step 4: plots ─────────────────────────────────────────────────
    plot_all_cameras(all_results, date, args.timestamp, out_base / "analysis.png")
    plot_overlap_summary(all_results, date, args.timestamp, out_base / "overlap_summary.png")

    # ── terminal summary ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SUMMARY — %s / %s", date, args.timestamp)
    log.info("=" * 60)
    for camera, (clip_ts, clip_scores, mot_ts, mot_scores) in all_results.items():
        clip_max_t = clip_ts[clip_scores.argmax()] if len(clip_scores) else 0
        mot_max_t  = mot_ts[mot_scores.argmax()]   if len(mot_scores)  else 0
        log.info(
            "  %-14s  CLIP max=%.3f @ t=%.0fs  |  Motion max=%.3f @ t=%.0fs",
            camera, clip_scores.max() if len(clip_scores) else 0, clip_max_t,
            mot_scores.max() if len(mot_scores) else 0, mot_max_t,
        )
    log.info("Output: %s", out_base.resolve())


if __name__ == "__main__":
    main()
