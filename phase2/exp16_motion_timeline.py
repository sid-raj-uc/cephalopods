"""
Experiment 16: Multi-camera motion timeline with voting.

For each indexed event in ethogram_index.json:
  1. Scan all available cameras in parallel for motion (full 30-min video)
  2. Align scores to a 1-second grid
  3. For each second, count how many cameras detect motion (vote)
  4. Merge active seconds into windows; sort by vote_count descending
  5. Write motion_timeline array back into ethogram_index.json

No VLM, no CLIP. vote_count is the confidence signal — higher means
more cameras agreed that something was happening at that moment.

Usage:
    python3 phase2/exp16_motion_timeline.py          # first 5 indexed entries
    python3 phase2/exp16_motion_timeline.py --n 20
    python3 phase2/exp16_motion_timeline.py --all    # all 64 indexed entries
"""

import json, sys, argparse
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

sys.path.insert(0, str(Path(__file__).parent))
from motion_detector import scan_motion

PROJECT    = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT / "data" / "ethogram_index.json"

USER, PASS    = "octopus", "communication42"
MOTION_PERCENTILE = 80  # per-camera: active if score ≥ this percentile of own distribution
MERGE_GAP     = 15     # seconds — merge windows separated by less than this
MIN_WINDOW    = 10     # seconds — drop windows shorter than this
MAX_WORKERS   = 4      # parallel ffmpeg processes


# ── helpers ───────────────────────────────────────────────────────────────────

def embed_auth(url: str) -> str:
    p = urlparse(url)
    netloc = f"{USER}:{PASS}@{p.hostname}"
    if p.port:
        netloc += f":{p.port}"
    return urlunparse(p._replace(netloc=netloc))


def fmt(sec: int) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def scan_camera(cam_name: str, url: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Scan one camera; returns (name, timestamps, scores). Safe for threading."""
    try:
        ts, scores = scan_motion(embed_auth(url), fps=1.0)
        return cam_name, ts, scores
    except Exception as e:
        print(f"    [{cam_name}] scan failed: {e}", flush=True)
        return cam_name, np.array([]), np.array([])


def build_vote_grid(camera_results: dict[str, tuple]) -> tuple[np.ndarray, dict]:
    """
    Align per-camera motion scores onto a common integer-second grid.

    Returns:
        vote_grid  : int array [0..1799], count of cameras active each second
        cam_grids  : dict cam_name → float array on same grid
    """
    # Determine grid length from longest scan
    max_sec = 0
    for ts, scores in camera_results.values():
        if len(ts):
            max_sec = max(max_sec, int(ts[-1]) + 1)
    max_sec = max(max_sec, 1800)  # at least 30 min

    vote_grid = np.zeros(max_sec, dtype=np.int32)
    cam_grids = {}

    for cam, (ts, scores) in camera_results.items():
        if len(ts) == 0:
            cam_grids[cam] = np.zeros(max_sec, dtype=np.float32)
            continue

        grid = np.zeros(max_sec, dtype=np.float32)
        indices = np.round(ts).astype(int)
        indices = np.clip(indices, 0, max_sec - 1)
        # where multiple frames map to same second, take the max
        for i, sc in zip(indices, scores):
            grid[i] = max(grid[i], sc)

        thresh = float(np.percentile(grid[grid > 0], MOTION_PERCENTILE)) if (grid > 0).any() else 1.0
        active = grid >= thresh
        vote_grid += active.astype(np.int32)
        cam_grids[cam] = grid

    return vote_grid, cam_grids


def build_windows(vote_grid: np.ndarray, cam_grids: dict) -> list[dict]:
    """
    Convert vote_grid into merged motion windows, sorted by vote_count desc.
    """
    active = vote_grid > 0  # at least 1 camera active
    windows = []
    in_window = False
    w_start = 0

    for t in range(len(active)):
        if active[t] and not in_window:
            w_start = t
            in_window = True
        elif not active[t] and in_window:
            windows.append((w_start, t - 1))
            in_window = False
    if in_window:
        windows.append((w_start, len(active) - 1))

    # Merge windows separated by ≤ MERGE_GAP
    merged = []
    for start, end in windows:
        if merged and start - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Drop short windows and build result dicts
    result = []
    for start, end in merged:
        if end - start < MIN_WINDOW:
            continue

        window_votes = vote_grid[start:end+1]
        window_sec   = np.argmax(window_votes) + start  # peak second
        peak_votes   = int(vote_grid[window_sec])

        # Which cameras are active at the peak second?
        cameras_active = [
            cam for cam, grid in cam_grids.items()
            if grid[window_sec] > 0
        ]
        peak_scores = {
            cam: round(float(cam_grids[cam][window_sec]), 3)
            for cam in cameras_active
        }

        result.append({
            "start":          fmt(start),
            "end":            fmt(end),
            "peak":           fmt(window_sec),
            "duration_sec":   end - start,
            "vote_count":     peak_votes,
            "cameras_active": sorted(cameras_active),
            "peak_scores":    peak_scores,
        })

    # Sort: highest vote_count first, then earliest start
    result.sort(key=lambda w: (-w["vote_count"], w["start"]))
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def process_entry(entry: dict) -> dict:
    avail = {
        c["name"]: c["video_url"]
        for c in entry.get("cameras", [])
        if c.get("available")
    }
    if not avail:
        entry["motion_timeline"] = []
        return entry

    print(f"  Scanning {len(avail)} cameras in parallel …", flush=True)

    camera_results = {}
    CAM_TIMEOUT = 360  # seconds — kill whole batch if any camera stalls beyond this
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(avail))) as pool:
        futures = {
            pool.submit(scan_camera, name, url): name
            for name, url in avail.items()
        }
        try:
            for fut in as_completed(futures, timeout=CAM_TIMEOUT):
                name, ts, scores = fut.result()
                camera_results[name] = (ts, scores)
                status = f"{len(scores)}s" if len(scores) else "failed"
                print(f"    {name}: {status}", flush=True)
        except FuturesTimeout:
            for fut, name in futures.items():
                if not fut.done():
                    print(f"    {name}: timeout (>{CAM_TIMEOUT}s) — skipped", flush=True)
                    camera_results[name] = (np.array([]), np.array([]))

    vote_grid, cam_grids = build_vote_grid(camera_results)
    windows = build_windows(vote_grid, cam_grids)

    print(f"  → {len(windows)} motion windows found", flush=True)
    for w in windows[:5]:  # print top 5
        print(f"    {w['start']}–{w['end']} (peak {w['peak']}, "
              f"votes={w['vote_count']}, cams={len(w['cameras_active'])})", flush=True)

    entry["motion_timeline"] = windows
    return entry


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n",   type=int, default=5, help="Process first N indexed entries")
    group.add_argument("--all", action="store_true",  help="Process all indexed entries")
    args = parser.parse_args()

    with open(INDEX_PATH) as f:
        index = json.load(f)

    limit = None if args.all else args.n
    done  = 0

    for i, entry in enumerate(index):
        if entry.get("status") != "indexed":
            continue
        if "motion_timeline" in entry:
            print(f"  already done: {entry['date']} {entry['time']} — skip")
            continue
        if limit is not None and done >= limit:
            break

        print(f"\n[{done+1}] {entry['date']} {entry['time']} — {entry['event'][:55]}")
        index[i] = process_entry(entry)
        done += 1

        with open(INDEX_PATH, "w") as f:
            json.dump(index, f, indent=2)
        print("  saved.", flush=True)

    print(f"\nDone. Processed {done} entries.")


if __name__ == "__main__":
    main()
