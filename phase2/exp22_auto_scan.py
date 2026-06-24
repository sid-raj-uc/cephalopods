"""
Exp 22 — Auto-scan all Nity videos on the server for octopus clips.

Crawls the remote server, runs motion detection + CLIP classifier,
extracts 20s clips where octopus is present (P > 0.7), saves to
data/auto_clips/. Fully resumable — skips already-processed videos.

Collections scanned (Right-side cameras only):
  - O-vulgaris-Nity-2025-9-17--   (155+ dates)
  - O-vulgaris-Nity-2026-2-20--   (dates not already in ethogram_index.json)

Usage:
  python3 phase2/exp22_auto_scan.py
  python3 phase2/exp22_auto_scan.py --limit 10  # test first 10 videos
"""
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse, urljoin, unquote
from urllib.request import urlopen, Request

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT      = Path(__file__).resolve().parent.parent
CKPT_PATH    = PROJECT / "weights" / "clip_mlp_best.pt"
OUT_DIR      = PROJECT / "data" / "auto_clips"
PROGRESS_PATH = OUT_DIR / "progress.json"

USER, PASS   = "octopus", "communication42"
BASE_URL     = "https://repo.octopus-intelligence.org/public/"

# Only right-side cameras (left cameras miss Nity's den)
RIGHT_CAMS   = ["Right Back", "Right Front", "Right Right", "Right Top"]

VISIBLE_THRESHOLD = 0.70
MIN_VISIBLE_FRAC  = 0.60
CLIP_DURATION     = 20
MAX_SCAN_SEC      = 300
BATCH_SIZE        = 32
TOP_WINDOWS       = 3

COLLECTIONS = [
    "O-vulgaris-Nity-2025-9-17--/",
    "O-vulgaris-Nity-2026-2-20--/",
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def auth_url(url: str) -> str:
    p = urlparse(url)
    netloc = f"{USER}:{PASS}@{p.hostname}"
    if p.port:
        netloc += f":{p.port}"
    return urlunparse(p._replace(netloc=netloc))


def list_dir(url: str) -> list[str]:
    """Return href values from an lighttpd directory listing."""
    req = Request(auth_url(url))
    with urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="ignore")
    hrefs = re.findall(r'href="([^"]+)"', html)
    return [unquote(h) for h in hrefs if h not in ("../", "./", "..")]


def discover_videos() -> list[dict]:
    """
    Crawl server and return list of {date, time_str, camera, url} for
    all right-side cameras across both collections.
    """
    videos = []
    for collection in COLLECTIONS:
        col_url = urljoin(BASE_URL, collection)
        print(f"\nCrawling {collection} …")
        try:
            cameras = list_dir(col_url)
        except Exception as e:
            print(f"  Failed: {e}"); continue

        for cam_entry in cameras:
            cam_name = cam_entry.rstrip("/")
            if not any(cam_name == rc for rc in RIGHT_CAMS):
                continue
            cam_url = urljoin(col_url, cam_entry)
            try:
                sub = list_dir(cam_url)           # usually just "Local/"
            except Exception:
                continue
            local_url = urljoin(cam_url, "Local/") if "Local/" in sub else cam_url
            try:
                dates = list_dir(local_url)
            except Exception:
                continue

            for date_entry in dates:
                date_url = urljoin(local_url, date_entry)
                try:
                    files = list_dir(date_url)
                except Exception:
                    continue
                for f in files:
                    if not f.endswith(".mp4"):
                        continue
                    videos.append({
                        "collection": collection.rstrip("/"),
                        "date":       date_entry.rstrip("/"),
                        "time_str":   f.replace("--vv-1.mp4", "").replace(".mp4", ""),
                        "camera":     cam_name,
                        "url":        urljoin(date_url, f),
                    })

    print(f"\nFound {len(videos)} videos across right-side cameras")
    return videos


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device):
    try:
        import pkg_resources
        import packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib

    ckpt = torch.load(CKPT_PATH, map_location=device)
    arch = ckpt.get("arch", "linear")
    feat_dim = ckpt["feat_dim"]

    hidden = [int(x) for x in arch.replace("mlp_", "").split("_")] if arch != "linear" else []
    dims   = [feat_dim] + hidden + [2]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(0.3)]
    classifier = (nn.Linear(feat_dim, 2) if arch == "linear" else nn.Sequential(*layers)).to(device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()

    clip_model, preprocess = clip_lib.load(ckpt["clip_model"], device=device)
    clip_model.eval()
    print(f"Model: CLIP {ckpt['clip_model']} + {arch}  acc={ckpt.get('test_acc',0):.1%}")
    return clip_model, preprocess, classifier


# ── Motion detection ──────────────────────────────────────────────────────────

def get_motion_windows(url: str) -> list[dict]:
    """Run exp16-style motion scan and return top windows."""
    sys.path.insert(0, str(PROJECT / "phase2"))
    from motion_detector import scan_motion
    from exp16_motion_timeline import build_vote_grid, build_windows

    try:
        ts, scores = scan_motion(auth_url(url), fps=1.0)
    except Exception as e:
        print(f"    motion scan failed: {e}")
        return []

    if len(ts) == 0:
        return []

    vote_grid, cam_grids = build_vote_grid({url: (ts, scores)})
    return build_windows(vote_grid, cam_grids)


# ── CLIP classification ───────────────────────────────────────────────────────

def classify_window(url, start_sec, end_sec, clip_model, preprocess, classifier, device):
    duration = end_sec - start_sec
    if duration < CLIP_DURATION:
        return np.array([])
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = str(Path(tmpdir) / "f_%04d.jpg")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(start_sec), "-i", auth_url(url),
             "-t", str(duration), "-vf", "fps=1", "-q:v", "3", pattern],
            capture_output=True, timeout=300)
        if r.returncode != 0:
            return np.array([])
        frames = sorted(Path(tmpdir).glob("f_*.jpg"))
        if not frames:
            return np.array([])
        probs = []
        for i in range(0, len(frames), BATCH_SIZE):
            batch = []
            for f in frames[i:i+BATCH_SIZE]:
                try:
                    batch.append(preprocess(Image.open(f).convert("RGB")))
                except Exception:
                    batch.append(torch.zeros(3, 224, 224))
            bt = torch.stack(batch).to(device)
            with torch.no_grad():
                feat = clip_model.encode_image(bt).float()
                feat = feat / feat.norm(dim=-1, keepdim=True)
                p = torch.softmax(classifier(feat), dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
    return np.array(probs)


def best_window(probs):
    w = CLIP_DURATION
    if len(probs) < w:
        return 0, float((probs > VISIBLE_THRESHOLD).mean()) if len(probs) else 0.0
    visible = (probs > VISIBLE_THRESHOLD).astype(float)
    scores  = np.convolve(visible, np.ones(w), mode="valid") / w
    i = int(np.argmax(scores))
    return i, float(scores[i])


# ── Clip extraction ───────────────────────────────────────────────────────────

def extract_clip(url, start_sec, out_path):
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(start_sec), "-i", auth_url(url),
         "-t", str(CLIP_DURATION), "-c:v", "libx264",
         "-crf", "23", "-preset", "fast", "-an", str(out_path)],
        capture_output=True, timeout=120)
    return r.returncode == 0


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def safe(s):
    return s.replace(":", "").replace(" ", "_").replace("/", "-")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = json.load(open(PROGRESS_PATH)) if PROGRESS_PATH.exists() else {"done": [], "clips": []}
    done_set = set(progress["done"])

    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_model, preprocess, classifier = load_model(device)

    # Group videos by (date, time_str) so we process all cameras together
    all_videos = discover_videos()
    # Group: key = (collection, date, time_str)
    groups: dict[tuple, list] = {}
    for v in all_videos:
        key = (v["collection"], v["date"], v["time_str"])
        groups.setdefault(key, []).append(v)

    keys    = sorted(groups.keys())
    todo    = [k for k in keys if str(k) not in done_set]
    total   = len(todo)
    print(f"\n{len(keys)} video groups total, {total} pending\n")

    n_clips = 0
    for gi, key in enumerate(todo[:args.limit] if args.limit else todo):
        collection, date, time_str = key
        cam_videos = groups[key]
        print(f"\n[{gi+1}/{total}] {date} {time_str} ({collection})", flush=True)

        best_cam_url  = None
        best_frac     = 0.0
        best_offset   = 0
        best_cam_name = ""

        for v in cam_videos:
            print(f"  [{v['camera']}] scanning motion …", flush=True)
            windows = get_motion_windows(v["url"])
            if not windows:
                print(f"    no motion windows"); continue

            for w in windows[:TOP_WINDOWS]:
                from exp16_motion_timeline import parse_sec as ps16
                peak = ps16(w["peak"])
                s    = max(ps16(w["start"]), peak - MAX_SCAN_SEC // 2)
                e    = min(ps16(w["end"]),   peak + MAX_SCAN_SEC // 2)
                if e - s < CLIP_DURATION:
                    continue

                probs = classify_window(v["url"], s, e, clip_model, preprocess, classifier, device)
                if len(probs) < CLIP_DURATION:
                    continue
                offset, frac = best_window(probs)
                print(f"    window {w['start']}–{w['end']}: {frac:.0%} visible", flush=True)

                if frac > best_frac:
                    best_frac     = frac
                    best_cam_url  = v["url"]
                    best_cam_name = v["camera"]
                    best_offset   = s + offset

        if best_frac >= MIN_VISIBLE_FRAC:
            clip_start = best_offset
            fname = f"{safe(date)}_{time_str}_{safe(best_cam_name)}_{fmt(clip_start)}-{fmt(clip_start+CLIP_DURATION)}.mp4"
            out   = OUT_DIR / fname
            if not out.exists():
                print(f"  → extracting {fmt(clip_start)}–{fmt(clip_start+CLIP_DURATION)} from {best_cam_name} ({best_frac:.0%}) …", flush=True)
                ok = extract_clip(best_cam_url, clip_start, out)
                if not ok:
                    print("  → ffmpeg failed");
            else:
                print(f"  → already exists: {fname}")
            if out.exists():
                n_clips += 1
                progress["clips"].append({"file": fname, "date": date, "time": time_str,
                                           "camera": best_cam_name, "visible_frac": round(best_frac, 3)})
        else:
            print(f"  → skip (best {best_frac:.0%} < {MIN_VISIBLE_FRAC:.0%})")

        progress["done"].append(str(key))
        with open(PROGRESS_PATH, "w") as f:
            json.dump(progress, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"Done. {n_clips} new clips → {OUT_DIR}")


if __name__ == "__main__":
    main()
