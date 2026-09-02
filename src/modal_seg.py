"""modal_seg.py — run the octopus-segmentation teacher+student pipeline on Modal (A100).

Ports the local A100-VM workflow to Modal so it's serverless + reproducible + parallel:
  * auto_label  — GroundingDINO+SAM2 teacher over harvest clips, fanned out with .map()
  * train       — the tiny segmenter (aug LR-ASPP) on v3 (positives+negatives) + harvest masks = v4
  * presence_eval — mask-area AUC on held-out reflection/absent negatives

Reuses the exact logic in src/{auto_segment,train_segmenter,segment_octopus}.py (added to the image).

Data lives in a Modal Volume `octo-seg-data` mounted at /data:
  /data/harvest_clips/...                     (raw clips, uploaded)
  /data/dataset_seg/v3/{images,masks,manifest.jsonl}   (existing positives + negatives, uploaded)
  /data/dataset_seg/harvest/{images,masks}    (auto-labeled here)
  /data/weights/octo_seg_v4_lraspp.pt         (output)

Usage (after `modal volume put` — see UPLOAD below):
  modal run src/modal_seg.py            # full pipeline: auto-label -> train -> eval
"""
import json
import sys
from pathlib import Path

import modal

GPU = "A100"
app = modal.App("octo-seg")
vol = modal.Volume.from_name("octo-seg-data", create_if_missing=True)
DATA = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers>=4.44", "opencv-python-headless", "pillow", "numpy<2")
    .run_commands("SAM2_BUILD_CUDA=0 pip install 'git+https://github.com/facebookresearch/sam2.git'")
    # our ported logic — importable at runtime
    .add_local_dir("src", "/root/segsrc")
)


def _src():
    if "/root/segsrc" not in sys.path:
        sys.path.insert(0, "/root/segsrc")


# ── auto-labeling (teacher) ─────────────────────────────────────────────────────────
@app.function(image=image, gpu=GPU, volumes={DATA: vol}, timeout=60 * 60)
def auto_label(shard):
    """shard = list of clip paths relative to /data. Writes (image,mask) pairs to
    /data/dataset_seg/harvest and returns manifest rows."""
    _src()
    from PIL import Image
    import numpy as np
    from auto_segment import load_models, segment_clip, camera_of

    out = Path(DATA) / "dataset_seg" / "harvest"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    M = load_models("cuda")
    rows = []
    for idx, rel in enumerate(shard):
        clip = str(Path(DATA) / rel)
        cam = camera_of(clip)
        try:
            pairs, info = segment_clip(clip, M)
        except Exception as e:
            print("ERR", rel, type(e).__name__, e, flush=True)
            continue
        for j, (img, mask) in enumerate(pairs):
            stem = f"{Path(clip).stem}_{cam}_{abs(hash(rel)) % 10**8}_{j}"
            img.save(out / "images" / f"{stem}.jpg", quality=90)
            Image.fromarray((mask * 255).astype(np.uint8)).save(out / "masks" / f"{stem}.png")
            rows.append({"clip": clip, "camera": cam,
                         "image": f"dataset_seg/harvest/images/{stem}.jpg",
                         "mask": f"dataset_seg/harvest/masks/{stem}.png",
                         "area": round(float(mask.mean()), 4), "best_conf": info.get("best_conf")})
    vol.commit()
    print(f"shard done: {len(shard)} clips -> {len(rows)} pairs", flush=True)
    return rows


# ── training (student) ──────────────────────────────────────────────────────────────
@app.function(image=image, gpu=GPU, volumes={DATA: vol}, timeout=3 * 60 * 60)
def train(harvest_rows, arch="lraspp", epochs=60, base_ch=32, in_size=256, val_frac=0.2, seed=42):
    """Merge v3 (positives+negatives) + harvest masks = v4, train, return best metrics + ckpt path."""
    _src()
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from train_segmenter import (build_model, SegDS, dice_bce_loss, evaluate,
                                  source_video, n_params_of)

    vol.reload()
    root = Path(DATA)
    # v3 rows -> paths relative to /data
    v3 = []
    for l in open(root / "dataset_seg/v3/manifest.jsonl"):
        r = json.loads(l)
        r["image"] = f"dataset_seg/v3/{r['image']}"
        r["mask"] = f"dataset_seg/v3/{r['mask']}"
        v3.append(r)
    rows = v3 + list(harvest_rows)
    n_pos = sum(1 for r in rows if not r.get("negative"))
    n_neg = len(rows) - n_pos
    print(f"v4: {len(rows)} pairs = {n_pos} pos + {n_neg} neg (v3 {len(v3)} + harvest {len(harvest_rows)})", flush=True)

    rng = np.random.RandomState(seed)
    vids = sorted({source_video(r["clip"]) for r in rows}); rng.shuffle(vids)
    n_val = max(1, int(len(vids) * val_frac)); val_vids = set(vids[:n_val])
    tr = [r for r in rows if source_video(r["clip"]) not in val_vids]
    va = [r for r in rows if source_video(r["clip"]) in val_vids]
    print(f"videos {len(vids)} (train {len(vids)-n_val}/val {n_val}) -> train {len(tr)}/val {len(va)} frames", flush=True)

    dev = "cuda"
    tl = DataLoader(SegDS(tr, root, in_size, train=True, aug="strong"), batch_size=32,
                    shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    vl = DataLoader(SegDS(va, root, in_size), batch_size=32, num_workers=8, pin_memory=True)
    model = build_model(arch, base_ch).to(dev)
    print(f"{arch}: {n_params_of(model)/1e6:.3f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    best_iou, best_state, best = -1, None, None
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = dice_bce_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        m = evaluate(model, vl, dev)
        print(f"ep {ep+1}/{epochs} val IoU {m['iou']:.4f} dice {m['dice']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m["iou"] > best_iou:
            best_iou, best = m["iou"], m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    outp = root / "weights" / f"octo_seg_v4_{arch}.pt"
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch": arch, "base_ch": base_ch, "in_size": in_size,
                "aug": "strong", "val": best, "n_params": n_params_of(model), "ds": "v4"}, outp)
    vol.commit()
    print(f"BEST val IoU {best_iou:.4f} -> {outp}", flush=True)
    return {"best_iou": best_iou, "metrics": best, "ckpt": str(outp),
            "n_pos": n_pos, "n_neg": n_neg, "n_videos": len(vids)}


# ── presence eval ────────────────────────────────────────────────────────────────────
@app.function(image=image, gpu=GPU, volumes={DATA: vol}, timeout=60 * 60)
def presence_eval(ckpt_rel="weights/octo_seg_v4_lraspp.pt", neg_dir="seg_neg"):
    """AUC of mask-area separating present (harvest val) vs negatives under /data/<neg_dir>."""
    _src()
    import glob, subprocess, tempfile, os
    import numpy as np
    from PIL import Image
    from segment_octopus import OctoSegmenter

    vol.reload()
    root = Path(DATA)
    seg = OctoSegmenter(str(root / ckpt_rel))
    # positives: harvest images
    pos = sorted(glob.glob(str(root / "dataset_seg/harvest/images/*.jpg")))[:300]
    pos_a = np.array([seg.segment(Image.open(p).convert("RGB"))[1] for p in pos])
    # negatives: clips under /data/<neg_dir>
    negc = sorted(glob.glob(str(root / neg_dir / "**/*.mp4"), recursive=True))
    neg_a = []
    for c in negc:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["ffmpeg", "-v", "error", "-i", c, "-vf", "fps=0.2",
                            "-frames:v", "3", f"{td}/f%02d.jpg"], check=False)
            for f in sorted(glob.glob(f"{td}/*.jpg")):
                neg_a.append(seg.segment(Image.open(f).convert("RGB"))[1])
    neg_a = np.array(neg_a) if neg_a else np.array([])

    def auc(p, n):
        if not len(p) or not len(n):
            return None
        lab = np.r_[np.ones(len(p)), np.zeros(len(n))]; sc = np.r_[p, n]
        o = np.argsort(sc); rk = np.empty(len(sc)); rk[o] = np.arange(1, len(sc)+1)
        return (rk[lab == 1].sum() - len(p)*(len(p)+1)/2) / (len(p)*len(n))
    res = {"n_pos": len(pos_a), "n_neg": len(neg_a), "val_iou": seg.val,
           "pos_area_median": float(np.median(pos_a)) if len(pos_a) else None,
           "neg_area_median": float(np.median(neg_a)) if len(neg_a) else None,
           "AUC": auc(pos_a, neg_a)}
    print(json.dumps(res, indent=1), flush=True)
    return res


@app.local_entrypoint()
def main(epochs: int = 60, shards: int = 8):
    """Full pipeline. Requires the volume pre-populated (see UPLOAD in the module docstring)."""
    # discover harvest clips inside the volume via a tiny helper
    clips = list_harvest.remote()
    print(f"harvest clips in volume: {len(clips)}")
    sh = [clips[i::shards] for i in range(shards)]
    sh = [s for s in sh if s]
    all_rows = []
    for rows in auto_label.map(sh):
        all_rows.extend(rows)
    print(f"auto-labeled -> {len(all_rows)} (image,mask) pairs")
    res = train.remote(all_rows, epochs=epochs)
    print("TRAIN:", json.dumps(res, indent=1))
    ev = presence_eval.remote()
    print("PRESENCE:", json.dumps(ev, indent=1))


@app.function(image=image, volumes={DATA: vol})
def list_harvest():
    import glob
    vol.reload()
    base = Path(DATA)
    return [str(Path(p).relative_to(base)) for p in
            glob.glob(str(base / "harvest_clips/**/*.mp4"), recursive=True)
            if Path(p).stat().st_size > 0]
