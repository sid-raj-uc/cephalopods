"""train_segmenter.py — Phase 2: train the tiny deployed octopus mask model.

Trains a compact single-class U-Net on the (image, mask) pairs produced by
`auto_segment.py` (the GroundingDINO+SAM2 teacher). This is the SMALL model that
actually ships in the extraction gate — the teacher never deploys.

Design (per SEGMENTATION_PLAN.md):
  * single class (octopus vs background), low input res (default 256).
  * smallest-first: a compact U-Net whose width is set by --base-ch. Sweep --base-ch
    (8/16/24/32) to trace the IoU-vs-size curve and pick the smallest model clearing
    the bar (val mask IoU >= 0.85 on colour cameras).
  * split BY SOURCE VIDEO (date/segment) so frames from one recording never straddle
    train/val — the honest generalization number.
  * loss = BCEWithLogits + soft Dice; metric = IoU@0.5 (+ Dice, mean area error).

Reads a dataset dir written by auto_segment.py:  <ds>/images/*.jpg, <ds>/masks/*.png,
<ds>/manifest.jsonl (one row per pair: image, mask, clip, camera, area, best_conf).

Saves weights/octo_seg_<ver>.pt = {state_dict, arch, base_ch, in_size, val: {...},
n_params, cameras}. Use src/segment_octopus.py to run inference from that checkpoint.

CLI:
  python3 train_segmenter.py --ds src/dataset_seg/v1 --base-ch 16 --epochs 40
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


# ── model: compact U-Net ──────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """4-level U-Net; width scales with base_ch (8->~0.13M, 16->~0.5M, 32->~2M params)."""
    def __init__(self, base_ch=16, in_ch=3):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.d1 = DoubleConv(in_ch, c[0])
        self.d2 = DoubleConv(c[0], c[1])
        self.d3 = DoubleConv(c[1], c[2])
        self.bott = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.u3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.u2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        xb = self.bott(self.pool(x3))
        y = self.u3(torch.cat([self.up3(xb), x3], 1))
        y = self.u2(torch.cat([self.up2(y), x2], 1))
        y = self.u1(torch.cat([self.up1(y), x1], 1))
        return self.head(y)  # logits [B,1,H,W]


# ── data ────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def source_video(clip_path):
    """Group key = the source recording: .../{date}/{segment}/Camera_s-e.mp4 -> 'date/segment'."""
    p = Path(clip_path)
    return f"{p.parent.parent.name}/{p.parent.name}"


class SegDS(Dataset):
    def __init__(self, rows, ds_root, size=256, train=False):
        self.rows, self.root, self.size, self.train = rows, Path(ds_root), size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["image"]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        m = Image.open(self.root / r["mask"]).convert("L").resize((self.size, self.size), Image.NEAREST)
        img = np.asarray(img, np.float32) / 255.0
        m = (np.asarray(m, np.float32) > 127).astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:                       # horizontal flip
                img = img[:, ::-1].copy(); m = m[:, ::-1].copy()
            if np.random.rand() < 0.5:                       # brightness/contrast jitter (image only)
                img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08), 0, 1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(img.transpose(2, 0, 1)),
                torch.from_numpy(m)[None])


# ── loss / metrics ────────────────────────────────────────────────────────────────
def dice_bce_loss(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious, dices, area_err = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (p * y).sum((1, 2, 3))
        union = ((p + y) > 0).float().sum((1, 2, 3))
        iou = torch.where(union > 0, inter / union, torch.ones_like(union))  # both empty => perfect
        dice = torch.where((p.sum((1, 2, 3)) + y.sum((1, 2, 3))) > 0,
                           2 * inter / (p.sum((1, 2, 3)) + y.sum((1, 2, 3))), torch.ones_like(inter))
        ae = (p.mean((1, 2, 3)) - y.mean((1, 2, 3))).abs()
        ious += iou.tolist(); dices += dice.tolist(); area_err += ae.tolist()
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices)),
            "area_err": float(np.mean(area_err)), "n": len(ious)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(HERE / "dataset_seg" / "v1"))
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--in-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of SOURCE VIDEOS held out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="checkpoint path (default weights/octo_seg_<ver>_ch<base>.pt)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)

    ds_root = Path(args.ds)
    rows = [json.loads(l) for l in open(ds_root / "manifest.jsonl") if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {ds_root}/manifest.jsonl — run auto_segment.py first")

    # split by source video (no frame/video leakage)
    vids = sorted({source_video(r["clip"]) for r in rows})
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac))
    val_vids = set(vids[:n_val])
    tr = [r for r in rows if source_video(r["clip"]) not in val_vids]
    va = [r for r in rows if source_video(r["clip"]) in val_vids]
    print(f"device={device}  pairs={len(rows)}  videos={len(vids)} "
          f"(train {len(vids)-n_val} / val {n_val})  ->  train {len(tr)} / val {len(va)} frames", flush=True)

    tl = DataLoader(SegDS(tr, ds_root, args.in_size, train=True), batch_size=args.batch,
                    shuffle=True, num_workers=4, pin_memory=(device == "cuda"), drop_last=True)
    vl = DataLoader(SegDS(va, ds_root, args.in_size), batch_size=args.batch,
                    shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    model = TinyUNet(base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyUNet base_ch={args.base_ch}: {n_params/1e6:.3f}M params "
          f"(~{n_params*4/1e6:.1f} MB fp32)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_iou, best_state, best_metrics = -1.0, None, None
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = dice_bce_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        m = evaluate(model, vl, device)
        print(f"ep {ep+1:3d}/{args.epochs}  loss {np.mean(losses):.4f}  "
              f"val IoU {m['iou']:.4f}  Dice {m['dice']:.4f}  areaErr {m['area_err']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if m["iou"] > best_iou:
            best_iou = m["iou"]; best_metrics = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out = Path(args.out) if args.out else (REPO / "weights" / f"octo_seg_{args.ver}_ch{args.base_ch}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch": "TinyUNet", "base_ch": args.base_ch,
                "in_size": args.in_size, "val": best_metrics, "n_params": n_params,
                "ds": str(ds_root)}, out)
    bar = "PASS" if best_iou >= 0.85 else "below 0.85 bar"
    print(f"\nBEST val IoU {best_iou:.4f} ({bar})  |  {n_params/1e6:.3f}M params  ->  {out}", flush=True)


if __name__ == "__main__":
    main()
