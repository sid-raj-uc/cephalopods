"""
Exp 28 — Verify extracted clips with the new (letterbox) CLIP+MLP model.

For every clip in data/octopus_clips_auto/:
  - sample 1 frame/sec, classify octopus visible/hidden (letterbox preprocessing)
  - count, at p_visible thresholds 0.5 / 0.6 / 0.7, the fraction of visible frames
  - a clip "has octopus" if that fraction > MIN_FRAC (0.40)

Writes data/octopus_clips_verified.json (per-clip results) and prints, per camera,
how many clips pass the >40% bar at each threshold so the cutoff can be chosen.

Usage: venv/bin/python3 phase2/exp28_verify_clips.py
"""
import json, subprocess, tempfile, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT   = Path(__file__).resolve().parents[2]   # repo root (file is phase2/octo-clip-extraction/)
CLIPS_DIR = PROJECT / "data" / "octopus_clips_auto"
CKPT_PATH = PROJECT / "weights" / "clip_mlp_letterbox_v1.pt"   # letterbox model
OUT_JSON  = PROJECT / "data" / "octopus_clips_verified.json"

THRESHOLDS = [0.5, 0.6, 0.7]
MIN_FRAC   = 0.40
SAMPLE_FPS = 1.0
SIZE, BATCH = 224, 64


def letterbox(img, size=224, fill=(128, 128, 128)):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def build_clf(ck):
    feat = ck["feat_dim"]; arch = ck.get("arch", "linear")
    hid = [int(x) for x in arch.replace("mlp_", "").split("_")]; dims = [feat] + hid + [2]
    L = []
    for i in range(len(dims) - 1):
        L.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            L += [nn.ReLU(), nn.Dropout(0.3)]
    return nn.Sequential(*L)


def load_model(device):
    try:
        import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib
    ck = torch.load(CKPT_PATH, map_location=device)
    cm, pp = clip_lib.load(ck["clip_model"], device=device); cm.eval()
    clf = build_clf(ck).to(device); clf.load_state_dict(ck["state_dict"]); clf.eval()
    vis = ck.get("label_map", {}).get("visible", 1)
    print(f"model: {ck['clip_model']}+{ck['arch']}  acc={ck.get('test_acc',0):.1%}  (letterbox)")
    return cm, pp, clf, vis


def classify_clip(clip, cm, pp, clf, vis_idx, device):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip),
                        "-vf", f"fps={SAMPLE_FPS}", "-q:v", "2", f"{tmp}/f_%04d.jpg"],
                       capture_output=True)
        frames = sorted(Path(tmp).glob("f_*.jpg"))
        if not frames:
            return np.array([])
        probs = []
        for i in range(0, len(frames), BATCH):
            imgs = []
            for f in frames[i:i + BATCH]:
                try:
                    imgs.append(pp(letterbox(Image.open(f).convert("RGB"))))
                except Exception:
                    imgs.append(torch.zeros(3, SIZE, SIZE))
            x = torch.stack(imgs).to(device)
            with torch.no_grad():
                ft = cm.encode_image(x).float(); ft = ft / ft.norm(dim=-1, keepdim=True)
                p = torch.softmax(clf(ft), dim=1)[:, vis_idx]
            probs.extend(p.cpu().tolist())
    return np.array(probs, np.float32)


def main():
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    cm, pp, clf, vis_idx = load_model(device)
    print(f"device: {device}  | thresholds {THRESHOLDS}  | min_frac {MIN_FRAC}")

    clips = sorted(CLIPS_DIR.rglob("*.mp4"))
    print(f"{len(clips)} clips to verify\n" + "-" * 60)

    results = []
    # pass counters: camera -> threshold -> count
    from collections import defaultdict
    passes = defaultdict(lambda: {t: 0 for t in THRESHOLDS})
    totals = defaultdict(int)

    for i, clip in enumerate(clips, 1):
        parts = clip.parts
        date, seg = parts[-3], parts[-2]
        cam = clip.stem.rsplit("_", 1)[0]
        totals[cam] += 1
        probs = classify_clip(clip, cm, pp, clf, vis_idx, device)
        if len(probs) == 0:
            continue
        rec = {"clip_path": str(clip.relative_to(PROJECT)), "camera": cam,
               "date": date, "segment": seg, "n_frames": int(len(probs)),
               "mean_p": round(float(probs.mean()), 4), "max_p": round(float(probs.max()), 4)}
        for t in THRESHOLDS:
            frac = float((probs >= t).mean())
            rec[f"frac_{t}"] = round(frac, 3)
            rec[f"pass40_{t}"] = bool(frac > MIN_FRAC)
            if frac > MIN_FRAC:
                passes[cam][t] += 1
        results.append(rec)

        if i % 100 == 0 or i == len(clips):
            json.dump({"model": "clip_mlp_letterbox_v1.pt (letterbox)", "min_frac": MIN_FRAC,
                       "thresholds": THRESHOLDS,
                       "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                       "count": len(results), "clips": results},
                      open(OUT_JSON, "w"), indent=1)
            print(f"  [{i}/{len(clips)}] processed", flush=True)

    print("-" * 60)
    print(f"Verified {len(results)} clips. Results -> {OUT_JSON.relative_to(PROJECT)}")
    print(f"\nClips passing >{int(MIN_FRAC*100)}% visible, by camera and threshold:")
    print(f"  {'camera':12s} {'total':>6s}  " + "  ".join(f"p>={t}" for t in THRESHOLDS))
    grand = {t: 0 for t in THRESHOLDS}
    for cam in sorted(totals):
        row = "  ".join(f"{passes[cam][t]:5d}" for t in THRESHOLDS)
        print(f"  {cam:12s} {totals[cam]:6d}  {row}")
        for t in THRESHOLDS:
            grand[t] += passes[cam][t]
    print(f"  {'TOTAL':12s} {sum(totals.values()):6d}  " + "  ".join(f"{grand[t]:5d}" for t in THRESHOLDS))


if __name__ == "__main__":
    main()
