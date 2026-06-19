"""
Augment hidden frames 2x to balance the dataset.

Augmentation per copy:
  Copy 1 — horizontal flip + brightness/contrast jitter
  Copy 2 — gaussian noise + random crop+resize

Output: data/frames/hidden_aug/
Manifest: updated to include augmented frames (label=hidden)

Usage: venv/bin/python3 phase2/augment_hidden.py
"""
import csv, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

PROJECT    = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT / "data" / "frames" / "hidden"
AUG_DIR    = PROJECT / "data" / "frames" / "hidden_aug"
MANIFEST   = PROJECT / "data" / "frames" / "manifest.csv"

random.seed(42)
np.random.seed(42)


def aug_flip_brightness(img: Image.Image) -> Image.Image:
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img


def aug_noise_crop(img: Image.Image) -> Image.Image:
    # gaussian noise
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # random crop 90-100% then resize back
    w, h = img.size
    scale = random.uniform(0.90, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    img = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BILINEAR)
    return img


def main():
    AUG_DIR.mkdir(parents=True, exist_ok=True)

    hidden_frames = sorted(HIDDEN_DIR.glob("*.jpg"))
    print(f"Hidden frames: {len(hidden_frames)}")
    print(f"Generating 2 augmented copies each → {len(hidden_frames)*2} new frames\n")

    aug_paths = []
    for src in tqdm(hidden_frames, desc="Augmenting"):
        img = Image.open(src).convert("RGB")

        for i, fn in enumerate([aug_flip_brightness, aug_noise_crop], start=1):
            out_name = src.stem + f"_aug{i}.jpg"
            out_path = AUG_DIR / out_name
            if not out_path.exists():
                fn(img).save(out_path, quality=90)
            aug_paths.append(out_path)

    print(f"\nSaved {len(aug_paths)} augmented frames to data/frames/hidden_aug/")

    # Update manifest — append augmented entries
    existing = []
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            existing = list(csv.DictReader(f))

    aug_rel = [{"path": str(p.relative_to(PROJECT)), "label": "hidden"} for p in aug_paths]
    # Remove stale aug entries before re-adding
    existing = [r for r in existing if "hidden_aug" not in r["path"]]
    all_rows = existing + aug_rel

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows(all_rows)

    counts = {}
    for r in all_rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print("\nManifest updated:")
    for label, count in sorted(counts.items()):
        print(f"  {label:12s}: {count:5d} frames")
    print(f"  manifest   : {MANIFEST}")


if __name__ == "__main__":
    main()
