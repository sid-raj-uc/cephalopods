"""
Exp 24 — Scan unused Right-camera footage, pseudo-label frames for review.

Goal: grow the presence-classifier training set via active learning.
  1. Build a candidate pool of LOCAL full recordings that are
       (a) "Right" cameras only, and
       (b) NOT already used to build the dataset (i.e. not in octopus_patches.json)
  2. Sample 1 frame / 10s (fps=0.1) from each candidate  [override with --fps]
  3. Classify each frame with the trained CLIP+MLP (weights/clip_mlp_best.pt)
  4. Save frames into present / absent / uncertain bands (prob in the filename)
     so they can be eyeballed and corrected, then folded back into training.

Resource-light: one video at a time, batched CLIP inference, resumable
(skips videos already in the manifest). Safe to Ctrl-C.

Usage:
  python3 phase2/exp24_scan_frames.py --limit-videos 1      # smoke test
  python3 phase2/exp24_scan_frames.py                       # all candidates
"""
import argparse, csv, json, re, subprocess, sys, tempfile
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT     = Path(__file__).resolve().parent.parent
FULL_DIR    = PROJECT / "data" / "aquarium" / "full"
PATCHES     = PROJECT / "data" / "octopus_patches.json"
CKPT_PATH   = PROJECT / "weights" / "clip_mlp_best.pt"
OUT_DIR     = PROJECT / "data" / "review_frames"
MANIFEST    = OUT_DIR / "manifest.csv"

FPS            = 0.1    # 1 frame / 10s (sparse — avoids near-duplicate frames)
BATCH_SIZE     = 32
ABSENT_MAX     = 0.35   # p < this  -> absent
PRESENT_MIN    = 0.65   # p > this  -> present ; in between -> uncertain


# ── Candidate pool ──────────────────────────────────────────────────────────────

def used_recordings() -> set[tuple[str, str, str]]:
    """(date, segment, camera) tuples already used to build the labeled dataset."""
    patches = json.load(open(PATCHES))["patches"]
    url_re  = re.compile(r"/public/[^/]+/([^/]+)/Local/(\d{4}-\d{2}-\d{2})/(\d+)--")
    used = set()
    for p in patches:
        m = url_re.search(p.get("video_url", ""))
        if m:
            used.add((m.group(2), m.group(3), unquote(m.group(1))))
    return used


def candidate_videos() -> list[tuple[Path, str, str, str]]:
    used = used_recordings()
    out = []
    for p in sorted(FULL_DIR.rglob("*.mp4")):
        date, seg = p.parts[-3], p.parts[-2]
        cam = p.stem.replace("_", " ")
        if "Right" not in cam:                 # Right cameras only
            continue
        if (date, seg, cam) in used:           # skip dataset videos
            continue
        out.append((p, date, seg, cam))
    return out


# ── Model (mirrors exp18) ─────────────────────────────────────────────────────────

def build_classifier(ckpt: dict) -> nn.Module:
    feat_dim = ckpt["feat_dim"]
    arch     = ckpt.get("arch", "linear")
    if arch == "linear":
        return nn.Linear(feat_dim, 2)
    hidden = [int(x) for x in arch.replace("mlp_", "").split("_")]
    dims   = [feat_dim] + hidden + [2]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(0.3)]
    return nn.Sequential(*layers)


def load_model(device):
    # clip's import touches pkg_resources.packaging, which is gone in new setuptools.
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
    vis_idx = ckpt.get("label_map", {}).get("visible", 1)
    print(f"Model: CLIP {ckpt['clip_model']} + {ckpt.get('arch')}  "
          f"(test_acc {ckpt.get('test_acc', 0):.1%}, visible=idx{vis_idx})")
    return clip_model, preprocess, classifier, vis_idx


# ── Frame scan ────────────────────────────────────────────────────────────────────

def extract_frames(video: Path, tmpdir: str) -> list[Path]:
    pattern = str(Path(tmpdir) / "f_%05d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={FPS}", "-q:v", "3", pattern],
        capture_output=True,
    )
    return sorted(Path(tmpdir).glob("f_*.jpg"))


def classify(frames, clip_model, preprocess, classifier, vis_idx, device) -> np.ndarray:
    probs = []
    for i in range(0, len(frames), BATCH_SIZE):
        imgs = []
        for f in frames[i:i + BATCH_SIZE]:
            try:
                imgs.append(preprocess(Image.open(f).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            p = torch.softmax(classifier(feats), dim=1)[:, vis_idx]
        probs.extend(p.cpu().tolist())
    return np.array(probs)


def band(p: float) -> str:
    if p < ABSENT_MAX:  return "absent"
    if p > PRESENT_MIN: return "present"
    return "uncertain"


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    global FPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--fps", type=float, default=FPS,
                    help=f"frames sampled per second (default {FPS} = 1 frame/{int(1/FPS)}s)")
    args = ap.parse_args()
    FPS = args.fps

    cands = candidate_videos()
    print(f"{len(cands)} candidate Right-camera recordings (unused in dataset)")
    if args.limit_videos:
        cands = cands[:args.limit_videos]
        print(f"  limited to first {len(cands)} for this run")

    for b in ("present", "absent", "uncertain"):
        (OUT_DIR / b).mkdir(parents=True, exist_ok=True)

    # Resume: which videos are already in the manifest
    done = set()
    rows = []
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            rows = list(csv.DictReader(f))
        done = {r["video"] for r in rows}

    todo = [c for c in cands if str(c[0].relative_to(PROJECT)) not in done]
    print(f"{len(todo)} to scan ({len(done)} videos already done)\n" + "-" * 64)
    if not todo:
        print("Nothing to do."); return

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess, classifier, vis_idx = load_model(device)

    fieldnames = ["video", "date", "segment", "camera", "t_sec",
                  "p_visible", "pred", "band", "frame_path"]
    counts = {"present": 0, "absent": 0, "uncertain": 0}

    for vi, (video, date, seg, cam) in enumerate(todo, 1):
        rel = str(video.relative_to(PROJECT))
        print(f"[{vi}/{len(todo)}] {date} {seg} {cam}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            frames = extract_frames(video, tmp)
            if not frames:
                print("   ! no frames extracted, skip"); continue
            probs = classify(frames, clip_model, preprocess, classifier, vis_idx, device)

            cam_safe = cam.replace(" ", "_")
            for idx, (f, p) in enumerate(zip(frames, probs)):
                t_sec = int(idx / FPS)
                bnd   = band(float(p))
                fname = f"p{p:.2f}_{date}_{seg}_{cam_safe}_t{t_sec:04d}.jpg"
                dst   = OUT_DIR / bnd / fname
                Image.open(f).convert("RGB").save(dst, quality=90)
                counts[bnd] += 1
                rows.append({
                    "video": rel, "date": date, "segment": seg, "camera": cam,
                    "t_sec": t_sec, "p_visible": f"{p:.4f}",
                    "pred": "visible" if p >= 0.5 else "hidden",
                    "band": bnd, "frame_path": str(dst.relative_to(PROJECT)),
                })
            v = (probs >= 0.5).sum()
            print(f"   {len(frames)} frames -> {v} visible / {len(frames)-v} hidden "
                  f"| uncertain: {int(((probs>ABSENT_MAX)&(probs<PRESENT_MIN)).sum())}", flush=True)

        # save manifest after each video (resumable)
        with open(MANIFEST, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

    print("-" * 64)
    print(f"Done. {sum(counts.values())} frames saved to {OUT_DIR}/")
    print(f"   present:   {counts['present']}")
    print(f"   absent:    {counts['absent']}")
    print(f"   uncertain: {counts['uncertain']}  <- review these first")
    print(f"Manifest: {MANIFEST.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
