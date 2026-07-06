"""
Behavior-classifier STUDENT that copies the v2 Qwen labels (distillation).

Teacher = Qwen3-VL v2 (data/octopus_clips_verified-2.json `ethogram_label`).
Student = frozen CLIP ViT-B/32 features, mean+max pooled over each clip's frames,
          -> MLP -> N classes. Cheap, runs locally; learns to reproduce v2.

Targets: v2 label remapped to the compact 7-behavior sheet (ethogram_list_v2.json)
plus 'octopus not present'. 'uncertain' clips are dropped.

Split: GROUPED by (date, segment) so near-duplicate clips from the same segment
don't leak across train/val (clip-level splits fake high accuracy).

Outputs: weights/behavior_student_v1.pt  (state_dict, classes, arch, val_acc)
Feature cache: data/behavior_feats.npz  (keyed by clip_path)

Usage: venv/bin/python3 phase2/octo-clip-extraction/train_behavior_student.py
"""
import json, subprocess, tempfile, datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

try:
    import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
    pkg_resources.packaging = packaging
except Exception:
    pass
import clip as clip_lib

PROJECT   = Path(__file__).resolve().parents[2]
V2_JSON   = PROJECT / "data" / "octopus_clips_verified-2.json"
ETHOGRAM  = PROJECT / "data" / "ethogram_list_v2.json"
CACHE     = PROJECT / "data" / "behavior_feats.npz"
OUT       = PROJECT / "weights" / "behavior_student_v1.pt"

FRAMES_PER_CLIP = 6      # evenly sampled; mean+max pooled -> 1024-d
EPOCHS, BATCH, LR, SEED = 80, 32, 1e-3, 42
torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def letterbox(img, size=224, fill=(128, 128, 128)):
    w, h = img.size; s = size / max(w, h); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    cv = Image.new("RGB", (size, size), fill); cv.paste(img, ((size - nw) // 2, (size - nh) // 2)); return cv


def build_remap():
    m = {}
    for bd in json.load(open(ETHOGRAM))["behaviors"]:
        for o in bd.get("maps_from", []): m[o] = bd["label"]
        m[bd["label"]] = bd["label"]
    m["octopus not present"] = "octopus not present"
    return m


def main():
    remap = build_remap()
    classes = [b["label"] for b in json.load(open(ETHOGRAM))["behaviors"]] + ["octopus not present"]
    cls_idx = {c: i for i, c in enumerate(classes)}

    v2 = json.load(open(V2_JSON))["clips"]
    rows = []  # (clip_path, label_idx, group)
    for c in v2:
        cp = c["clip_path"]
        if not (PROJECT / cp).exists():
            continue
        lab = remap.get(c.get("ethogram_label"))
        if lab not in cls_idx:            # drops None / 'uncertain'
            continue
        rows.append((cp, cls_idx[lab], f"{c.get('date')}/{c.get('segment')}"))
    print(f"{len(rows)} clips with a trainable v2 label (classes: {len(classes)})", flush=True)
    print("label dist:", dict(Counter(classes[r[1]] for r in rows)), flush=True)

    # ---- CLIP features (cached) ----
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {p: f for p, f in zip(list(z["paths"]), z["feats"])}
        print(f"feature cache: {len(cache)} clips", flush=True)
    clip_model, preprocess = clip_lib.load("ViT-B/32", device=device); clip_model.eval()

    def clip_feat(cp):
        with tempfile.TemporaryDirectory() as t:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(PROJECT / cp),
                            "-vf", f"fps=1", "-q:v", "3", f"{t}/f_%03d.jpg"], capture_output=True)
            fs = sorted(Path(t).glob("f_*.jpg"))
            if not fs:
                return None
            idx = np.linspace(0, len(fs) - 1, min(FRAMES_PER_CLIP, len(fs))).round().astype(int)
            imgs = [preprocess(letterbox(Image.open(fs[i]).convert("RGB"))) for i in idx]
            with torch.no_grad():
                f = clip_model.encode_image(torch.stack(imgs).to(device)).float()
                f = f / f.norm(dim=-1, keepdim=True)
            return np.concatenate([f.mean(0).cpu().numpy(), f.max(0).values.cpu().numpy()])  # 1024

    todo = [cp for cp, _, _ in rows if cp not in cache]
    print(f"extracting features for {len(todo)} new clips ...", flush=True)
    for i, cp in enumerate(todo, 1):
        fv = clip_feat(cp)
        if fv is not None:
            cache[cp] = fv
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
            np.savez(CACHE, paths=list(cache), feats=np.array(list(cache.values()), np.float32))
    np.savez(CACHE, paths=list(cache), feats=np.array(list(cache.values()), np.float32))

    rows = [r for r in rows if r[0] in cache]
    X = np.array([cache[cp] for cp, _, _ in rows], np.float32)
    y = np.array([li for _, li, _ in rows], np.int64)
    groups = np.array([g for _, _, g in rows])
    print(f"\nfeatures ready: {X.shape} | groups: {len(set(groups))}", flush=True)

    # ---- grouped split ----
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
    Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
    print(f"train {len(tr)} | val {len(va)} (no segment overlap)", flush=True)

    present = sorted(set(ytr))
    w = np.ones(len(classes), np.float32)
    cnt = Counter(ytr)
    for k in present: w[k] = len(ytr) / (len(present) * cnt[k])
    cw = torch.tensor(w, device=device)

    clf = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(64, len(classes))).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=cw)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    best, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        clf.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]; opt.zero_grad()
            crit(clf(Xtr_t[b]), ytr_t[b]).backward(); opt.step()
        sch.step(); clf.eval()
        with torch.no_grad():
            va_acc = (clf(Xva_t).argmax(1).cpu().numpy() == yva).mean()
        if va_acc > best: best, best_state = va_acc, {k: v.clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state); clf.eval()
    with torch.no_grad():
        pred = clf(Xva_t).argmax(1).cpu().numpy()
    print(f"\n=== STUDENT vs v2 labels (val, grouped) — best val acc {best:.3f} ===")
    seen = sorted(set(yva) | set(pred))
    print(classification_report(yva, pred, labels=seen,
                                target_names=[classes[i] for i in seen], zero_division=0))
    print("confusion (rows=true):"); print(confusion_matrix(yva, pred, labels=seen))

    torch.save({"state_dict": clf.state_dict(), "feat_dim": X.shape[1], "arch": "mlp_256_64",
                "classes": classes, "clip_model": "ViT-B/32", "pooling": "mean+max",
                "frames_per_clip": FRAMES_PER_CLIP, "teacher": "v2", "val_acc": float(best),
                "trained_at": datetime.datetime.now().isoformat(timespec="seconds")}, OUT)
    print(f"\nsaved -> {OUT.relative_to(PROJECT)}  (copies v2; val acc vs v2 = {best:.1%})")


if __name__ == "__main__":
    main()
