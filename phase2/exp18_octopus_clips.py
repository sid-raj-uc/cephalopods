"""
Exp 18 — Extract 20-second clips where octopus is present.

For each motion window in ethogram_index.json:
  1. Sample frames at 1fps across the window (up to 300s around the peak)
  2. Run CLIP linear classifier on each frame → P(octopus visible)
  3. Slide a 20s window to find the position with highest octopus presence
  4. If >= 60% of frames in that window are "visible", extract a 20s video clip

Best camera per window is chosen by highest visible fraction.

Output:
  data/octopus_clips/          — mp4 video clips
  data/octopus_clips/manifest.json — metadata for each clip

Usage:
  python3 phase2/exp18_octopus_clips.py
  python3 phase2/exp18_octopus_clips.py --max-windows 20  # quick test
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Allow importing from phase2/
sys.path.insert(0, str(Path(__file__).parent))

PROJECT    = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT / "data" / "ethogram_index.json"
CKPT_PATH  = PROJECT / "weights" / "clip_mlp_best.pt"
OUT_DIR    = PROJECT / "data" / "octopus_clips"

USER, PASS = "octopus", "communication42"

VISIBLE_THRESHOLD = 0.70  # P(visible) must exceed this to count a frame as "visible"
MIN_VISIBLE_FRAC  = 0.60  # at least 60% of frames must be "visible" to extract a clip
CLIP_DURATION    = 20     # seconds per output clip
MAX_SCAN_SEC     = 300    # scan at most 5 min around each window's peak
BATCH_SIZE       = 32
SAMPLE_FPS       = 1
TOP_WINDOWS      = 3      # only check top N motion windows per entry


# ── Auth ──────────────────────────────────────────────────────────────────────

def embed_auth(url: str) -> str:
    p = urlparse(url)
    netloc = f"{USER}:{PASS}@{p.hostname}"
    if p.port:
        netloc += f":{p.port}"
    return urlunparse(p._replace(netloc=netloc))


def parse_sec(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def fmt_sec(sec: int) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def safe(s: str) -> str:
    return s.replace(":", "").replace(" ", "_").replace("/", "-")


# ── Load model ────────────────────────────────────────────────────────────────

def build_classifier(ckpt: dict) -> nn.Module:
    feat_dim = ckpt["feat_dim"]
    arch     = ckpt.get("arch", "linear")
    if arch == "linear":
        return nn.Linear(feat_dim, 2)
    # Parse "mlp_256_64" → hidden dims [256, 64]
    hidden = [int(x) for x in arch.replace("mlp_", "").split("_")]
    dims   = [feat_dim] + hidden + [2]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(0.3)]
    return nn.Sequential(*layers)


def load_model(device):
    try:
        import pkg_resources
        import packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib

    ckpt = torch.load(CKPT_PATH, map_location=device)

    clip_model, preprocess = clip_lib.load(ckpt["clip_model"], device=device)
    clip_model.eval()

    classifier = build_classifier(ckpt).to(device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()

    print(f"Model loaded — CLIP {ckpt['clip_model']}  arch={ckpt.get('arch','linear')}  test_acc={ckpt.get('test_acc', 0):.1%}")
    return clip_model, preprocess, classifier


# ── Frame classification ──────────────────────────────────────────────────────

def classify_window(url: str, start_sec: int, end_sec: int,
                    clip_model, preprocess, classifier, device) -> np.ndarray:
    """Extract frames at 1fps from [start_sec, end_sec], return P(visible) array."""
    duration = end_sec - start_sec
    if duration < CLIP_DURATION:
        return np.array([])

    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = str(Path(tmpdir) / "frame_%04d.jpg")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start_sec),
            "-i", embed_auth(url),
            "-t", str(duration),
            "-vf", f"fps={SAMPLE_FPS}",
            "-q:v", "3",
            pattern,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            return np.array([])
        if r.returncode != 0:
            return np.array([])

        frames = sorted(Path(tmpdir).glob("frame_*.jpg"))
        if not frames:
            return np.array([])

        probs = []
        for i in range(0, len(frames), BATCH_SIZE):
            batch_imgs = []
            for f in frames[i : i + BATCH_SIZE]:
                try:
                    batch_imgs.append(preprocess(Image.open(f).convert("RGB")))
                except Exception:
                    batch_imgs.append(torch.zeros(3, 224, 224))
            batch_t = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                feats = clip_model.encode_image(batch_t).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = classifier(feats)
                p = torch.softmax(logits, dim=1)[:, 1]  # P(visible)
            probs.extend(p.cpu().tolist())

    return np.array(probs)


def best_20s_window(probs: np.ndarray) -> tuple[int, float]:
    """Return (offset_frames, visible_fraction) for the best 20s sub-window."""
    w = CLIP_DURATION * SAMPLE_FPS
    if len(probs) < w:
        frac = float((probs > 0.5).mean()) if len(probs) else 0.0
        return 0, frac
    visible = (probs > VISIBLE_THRESHOLD).astype(float)
    scores  = np.convolve(visible, np.ones(w), mode="valid") / w
    best_i  = int(np.argmax(scores))
    return best_i, float(scores[best_i])


# ── Clip extraction ───────────────────────────────────────────────────────────

def extract_clip(url: str, start_sec: int, out_path: Path) -> bool:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", embed_auth(url),
        "-t", str(CLIP_DURATION),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an",
        str(out_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


# ── Camera URL lookup ─────────────────────────────────────────────────────────

def cam_url_map(entry: dict) -> dict[str, str]:
    return {c["name"]: c["video_url"]
            for c in entry.get("cameras", [])
            if c.get("available")}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Stop after processing this many motion windows (for testing)")
    args = parser.parse_args()

    if not CKPT_PATH.exists():
        print(f"Checkpoint not found: {CKPT_PATH}")
        print("Run exp17_clip_classifier.ipynb to completion first.")
        sys.exit(1)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"Device: {device}")
    clip_model, preprocess, classifier = load_model(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index    = json.load(open(INDEX_PATH))
    manifest = []
    n_processed = 0
    n_extracted = 0

    for entry in index:
        if entry.get("status") not in ("indexed", "annotated"):
            continue
        windows = entry.get("motion_timeline", [])
        if not windows:
            continue

        cams = cam_url_map(entry)
        tag  = f"{safe(entry['date'])}_{safe(entry['time'])}"

        for wi, window in enumerate(windows[:TOP_WINDOWS]):
            if args.max_windows and n_processed >= args.max_windows:
                break

            peak_sec  = parse_sec(window["peak"])
            start_sec = parse_sec(window["start"])
            end_sec   = parse_sec(window["end"])

            # Limit scan to MAX_SCAN_SEC around peak
            scan_start = max(start_sec, peak_sec - MAX_SCAN_SEC // 2)
            scan_end   = min(end_sec,   peak_sec + MAX_SCAN_SEC // 2)
            if scan_end - scan_start < CLIP_DURATION:
                continue

            print(f"\n{tag} | window {wi+1} | {window['start']}–{window['end']} "
                  f"(peak {window['peak']}, votes={window['vote_count']})", flush=True)

            best_cam    = None
            best_frac   = 0.0
            best_offset = 0

            for cam_name in [c for c in window.get("cameras_active", []) if "Left" not in c][:3]:
                url = cams.get(cam_name)
                if not url:
                    continue
                print(f"  [{cam_name}] classifying {scan_end-scan_start}s …", flush=True)
                probs = classify_window(url, scan_start, scan_end,
                                        clip_model, preprocess, classifier, device)
                if len(probs) < CLIP_DURATION:
                    print(f"    → too few frames ({len(probs)}), skip")
                    continue

                offset, frac = best_20s_window(probs)
                print(f"    → best 20s: {frac:.0%} visible  (offset +{offset}s)")

                if frac > best_frac:
                    best_frac   = frac
                    best_cam    = cam_name
                    best_offset = offset

            n_processed += 1

            if best_frac < MIN_VISIBLE_FRAC:
                print(f"  → skip (best frac {best_frac:.0%} < {MIN_VISIBLE_FRAC:.0%})")
                continue

            clip_start = scan_start + best_offset
            clip_end   = clip_start + CLIP_DURATION
            fname      = f"{tag}_w{wi+1}_{safe(best_cam)}_{fmt_sec(clip_start)}-{fmt_sec(clip_end)}.mp4"
            out_path   = OUT_DIR / fname

            if out_path.exists():
                print(f"  → already extracted: {fname}")
            else:
                print(f"  → extracting {fmt_sec(clip_start)}–{fmt_sec(clip_end)} "
                      f"from {best_cam} ({best_frac:.0%} visible) …", flush=True)
                ok = extract_clip(cams[best_cam], clip_start, out_path)
                if not ok:
                    print("  → ffmpeg failed, skip")
                    continue

            n_extracted += 1
            manifest.append({
                "file":          fname,
                "date":          entry["date"],
                "time":          entry["time"],
                "event":         entry.get("event", ""),
                "camera":        best_cam,
                "clip_start":    fmt_sec(clip_start),
                "clip_end":      fmt_sec(clip_end),
                "visible_frac":  round(best_frac, 3),
                "vote_count":    window["vote_count"],
            })

        if args.max_windows and n_processed >= args.max_windows:
            break

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"Done. {n_extracted} clips extracted → {OUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
