"""
Extract labeled frames from octopus_patches.json at 1fps.

Output:
  data/frames/visible/  — octopus present
  data/frames/hidden/   — octopus not visible (negatives)
  data/frames/manifest.csv — path,label for each frame

Usage:
  python3 phase2/extract_frames.py
"""
import csv, json, subprocess, sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

PROJECT      = Path(__file__).resolve().parent.parent
PATCHES_PATH = PROJECT / "data" / "octopus_patches.json"
FRAMES_DIR   = PROJECT / "data" / "frames"
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from server_creds import USER, PASS  # creds from env / .env, not hardcoded
def embed_auth(url: str) -> str:
    p = urlparse(url)
    netloc = f"{USER}:{PASS}@{p.hostname}"
    if p.port:
        netloc += f":{p.port}"
    return urlunparse(p._replace(netloc=netloc))


def parse_sec(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def safe(s: str) -> str:
    return s.replace(":", "").replace(" ", "_").replace("/", "-")


def extract_patch(patch: dict) -> list[Path]:
    start_sec = parse_sec(patch["start"])
    end_sec   = parse_sec(patch["end"])
    duration  = end_sec - start_sec
    if duration <= 0:
        return []

    label   = patch["label"]
    out_dir = FRAMES_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "_".join([
        safe(patch["date"]), safe(patch["time"]),
        safe(patch["camera"]),
        f"{safe(patch['start'])}-{safe(patch['end'])}",
    ])
    pattern = str(out_dir / f"{prefix}_%04d.jpg")

    # Check if already extracted (resumable)
    existing = list(out_dir.glob(f"{prefix}_*.jpg"))
    if existing:
        print(f"  already extracted ({len(existing)} frames) — skip")
        return existing

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", embed_auth(patch["video_url"]),
        "-t", str(duration),
        "-vf", "fps=1",
        "-q:v", "3",
        pattern,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return []

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
        return []

    return sorted(out_dir.glob(f"{prefix}_*.jpg"))


def main():
    if not PATCHES_PATH.exists():
        print("No patches file found — label patches in the UI first (http://localhost:8001)")
        sys.exit(1)

    patches = json.load(open(PATCHES_PATH))["patches"]
    print(f"{len(patches)} patches to extract …\n")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    manifest  = []
    total     = 0

    for i, p in enumerate(patches):
        dur = parse_sec(p["end"]) - parse_sec(p["start"])
        print(f"[{i+1}/{len(patches)}] {p['date']} {p['time']} | {p['camera']} | "
              f"{p['start']}–{p['end']} ({dur}s) [{p['label']}]", flush=True)

        frames = extract_patch(p)
        print(f"  → {len(frames)} frames", flush=True)
        total += len(frames)
        for f in frames:
            manifest.append({"path": str(f.relative_to(PROJECT)), "label": p["label"]})

    # Write manifest
    manifest_path = FRAMES_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\n{'─'*50}")
    print(f"Done. {total} frames total → data/frames/")
    counts = {}
    for m in manifest:
        counts[m["label"]] = counts.get(m["label"], 0) + 1
    for label, count in sorted(counts.items()):
        print(f"  {label:10s}: {count:4d} frames")
    print(f"  manifest  : {manifest_path}")


if __name__ == "__main__":
    main()
