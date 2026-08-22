"""train_ethogram.py — the rung ladder for the 6-class ethogram classifier.

Each rung isolates ONE hypothesis, so an improvement is attributable rather than asserted:

  rung0  mean-pooled CLIP -> linear
         Reproduces the PREVIOUS failed classifier on the new split. Without it we cannot claim any
         gain: the old model scored 45% and was called a failure against a 50% baseline, but this
         class mix has a 43% majority baseline, so "45%" was never the right bar. The real symptom
         was per-class F1 ~0 on everything but the majority classes -- collapse, not low accuracy.

  rung1  mean | std | max pooling -> MLP
         Was the failure the POOLING STATISTIC rather than the missing sequence? std-over-time
         encodes variability without any sequence model. If rung1 fixes it, the cheap answer wins.

  rung2  rung1 + motion summary stats
         Tests the motion channel directly. CLIP encodes appearance, and an octopus crawling slowly
         gives near-identical embeddings 2 s apart, so appearance-change is the wrong signal for
         classes defined by movement.

  rung3  full [10, 514] -> projection -> BiGRU -> attention pool
         Genuine temporal modelling over appearance AND motion.

WHY EVERYTHING IS SMALL. ~2,000 training clips across ~60 SOURCE VIDEOS. The video count is the real
sample size, so a large head would memorise videos. rung3 is ~200K params, with dropout and early
stopping on a video-disjoint val split.

MULTI-SEED BY DEFAULT. At this scale a single run's macro-F1 moves by several points on seed alone, so
every rung is trained N_SEEDS times and reported as mean +/- std. A one-seed comparison between rungs
would be noise dressed as a result.

LOSS. KL divergence against the SOFT 5-vote target, not cross-entropy on the argmax: 31% of clips
have a split vote and human agreement tracks the margin (0.726 unanimous / 0.864 at 4-of-5 / 0.426 at
<=3/5), so a 3-2 clip should teach uncertainty. Per-clip `weight` is applied (IR-absent = 0.5), and
class imbalance is handled by inverse-frequency weighting (No octopus is ~43%).

METRIC. macro-F1, never accuracy -- the classes run 43% to 5%, so accuracy rewards collapse. Per-class
F1 is printed WITH n, and any class on <3 test videos is flagged as not meaningful.

Usage:
  venv/bin/python3 src/train_ethogram.py --version v1                 # all rungs
  venv/bin/python3 src/train_ethogram.py --version v1 --rungs 0 3
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
CW_POWER = 0.5              # sqrt inverse-freq. SELECTED ON VAL (see R28): p=1.0 made rare classes
#                             sinks (Reaching precision 0.35). Sweep 1.0/0.5/0.25/0.0 -> val picks
#                             0.5, test 0.5129 -> 0.5298. p=0.0 scores HIGHER on test (0.5581) but
#                             val does not pick it, so claiming it would be test-set selection.
N_SEEDS = 3
EPOCHS = 120
PATIENCE = 20


# ----------------------------------------------------------------------------- data
def load(version):
    d = REPO / "src" / "dataset_etho" / version
    man = [json.loads(l) for l in open(d / "manifest.jsonl") if l.strip()]
    snap = json.load(open(d / "snapshot.json"))
    classes = snap["classes"]
    npz = np.load(d / "features.npz") if (d / "features.npz").exists() else None
    X, keep = {}, []
    for r in man:
        k = r["clip"]
        arr = npz[k] if (npz is not None and k in npz) else None
        if arr is None:
            fp = d / "feats" / (k.replace("/", "__") + ".npy")
            arr = np.load(fp) if fp.exists() else None
        if arr is not None and arr.shape == (10, 514) and np.isfinite(arr).all():
            X[k] = arr.astype(np.float32); keep.append(r)
    print(f"loaded {len(keep)}/{len(man)} clips with valid features | classes {len(classes)}")
    return keep, X, classes


def split_rows(man, split):
    return [r for r in man if r["split"] == split]


# ----------------------------------------------------------------------------- rungs
def featurise(rows, X, rung):
    """-> (inputs, soft_targets, hard_labels, weights). Rungs 0-2 are vectors; rung3 is a sequence."""
    seq = np.stack([X[r["clip"]] for r in rows])            # [N, 10, 514]
    clip, mot = seq[:, :, :512], seq[:, :, 512:]
    if rung == 0:
        f = clip.mean(1)                                     # the old failure: pooling destroys time
    elif rung == 1:
        f = np.concatenate([clip.mean(1), clip.std(1), clip.max(1)], 1)
    elif rung == 2:
        f = np.concatenate([clip.mean(1), clip.std(1), clip.max(1),
                            mot.mean(1), mot.std(1), mot.max(1)], 1)
    else:
        f = seq                                              # full sequence for the GRU
    y = np.stack([np.asarray(r["soft"], np.float32) for r in rows])
    hard = np.array([r["label_idx"] for r in rows])
    w = np.array([r.get("weight", 1.0) for r in rows], np.float32)
    return f.astype(np.float32), y, hard, w


class MLP(nn.Module):
    def __init__(self, d_in, n_cls, hidden=256, p=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


class Linear(nn.Module):
    def __init__(self, d_in, n_cls):
        super().__init__()
        self.f = nn.Linear(d_in, n_cls)

    def forward(self, x):
        return self.f(x)


class SeqModel(nn.Module):
    """[10, 514] -> project -> BiGRU -> attention pool -> classes. ~200K params."""

    def __init__(self, n_cls, d_in=514, d=128, p=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU())
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * d, 1)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(2 * d, n_cls)

    def forward(self, x):
        h, _ = self.gru(self.proj(x))                        # [N, 10, 2d]
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)    # attention over time
        return self.out(self.drop((h * w.unsqueeze(-1)).sum(1)))


def build_model(rung, d_in, n_cls):
    if rung == 0:
        return Linear(d_in, n_cls)
    if rung in (1, 2):
        return MLP(d_in, n_cls)
    return SeqModel(n_cls)


# ----------------------------------------------------------------------------- metrics
def macro_f1(pred, true, n_cls):
    f1s, per = [], {}
    for c in range(n_cls):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1); per[c] = {"f1": round(f1, 4), "n": int((true == c).sum())}
    return float(np.mean(f1s)), per


def run_one(rung, man, X, classes, seed, extra_rows=None):
    torch.manual_seed(seed); np.random.seed(seed)
    n_cls = len(classes)
    tr, va, te = (split_rows(man, s) for s in ("train", "val", "test"))
    Xtr, Ytr, Htr, Wtr = featurise(tr, X, rung)
    Xva, Yva, Hva, _ = featurise(va, X, rung)
    Xte, Yte, Hte, _ = featurise(te, X, rung)
    # CLASS WEIGHTS, tempered by CW_POWER. p=1 is full inverse-frequency, which is what v1 used and
    # which measurably OVERSHOOTS: train counts run 1419 (No octopus) to 141 (Human), so p=1 hands the
    # rare classes 6-10x the weight and they become SINKS -- `Reaching out of water` reached recall
    # 0.81 at precision 0.35, i.e. 65% of everything it predicted was wrong, absorbing 121 No-octopus
    # and 84 Exploration samples. Macro-F1 punishes that on the precision side, so over-weighting a
    # rare class does not even buy the metric it was meant to protect. p=0.5 (sqrt) is the usual
    # middle ground; p=0 is no weighting at all.
    cnt = collections.Counter(Htr.tolist())
    cw = np.array([(1.0 / max(1, cnt.get(c, 0))) ** CW_POWER for c in range(n_cls)], np.float32)
    cw = cw / cw.sum() * n_cls
    sample_w = Wtr * cw[Htr]

    model = build_model(rung, Xtr.shape[-1], n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    t = lambda z: torch.from_numpy(z)
    xtr, ytr, wtr = t(Xtr), t(Ytr), t(sample_w)
    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            b = perm[i:i + 64]
            logp = torch.log_softmax(model(xtr[b]), -1)
            # KL(target || pred), per-sample, then weighted -- soft targets carry the vote spread
            loss = ((ytr[b] * (torch.log(ytr[b].clamp_min(1e-8)) - logp)).sum(-1) * wtr[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(t(Xva)).argmax(-1).numpy()
        f1, _ = macro_f1(pv, Hva, n_cls)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = model(t(Xte)).argmax(-1).numpy()
    f1_te, per = macro_f1(pt, Hte, n_cls)
    n_params = sum(p.numel() for p in model.parameters())
    out = {"val_f1": best, "test_f1": f1_te, "per_class": per, "n_params": n_params,
           "pred": pt.tolist(), "true": Hte.tolist()}
    # Optional extra rows to predict (e.g. the human-labelled clips, which span several splits).
    # Softmax rather than argmax so a caller can average over seeds before deciding -- averaging
    # argmaxes would let one seed's confident error outvote two seeds' correct uncertainty.
    if extra_rows:
        Xex, *_ = featurise(extra_rows, X, rung)
        with torch.no_grad():
            out["extra_probs"] = torch.softmax(model(t(Xex)), -1).numpy().tolist()
        out["extra_clips"] = [r["clip"] for r in extra_rows]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--rungs", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    man, X, classes = load(a.version)
    n_cls = len(classes)
    te = split_rows(man, "test")
    if not te:
        sys.exit("no test split -- run build_ethogram_dataset.py to completion first")

    # the bar every rung must clear
    maj = collections.Counter(r["label"] for r in split_rows(man, "train")).most_common(1)[0]
    te_cnt = collections.Counter(r["label"] for r in te)
    maj_f1, _ = macro_f1(np.full(len(te), classes.index(maj[0])),
                         np.array([r["label_idx"] for r in te]), n_cls)
    print(f"\ntest: {len(te)} clips / {len({r['video'] for r in te})} videos")
    print(f"majority-class baseline ('{maj[0]}'): macro-F1 {maj_f1:.4f}")
    thin = [c for c in classes if len({r['video'] for r in te if r['label'] == c}) < 3]
    if thin:
        print(f"NOTE: classes on <3 test videos -- per-class F1 not meaningful: {thin}")

    results = {"version": a.version, "classes": classes, "majority_macro_f1": round(maj_f1, 4),
               "n_test_clips": len(te), "n_test_videos": len({r["video"] for r in te}),
               "test_class_counts": dict(te_cnt), "rungs": {}}
    for rung in a.rungs:
        runs = [run_one(rung, man, X, classes, s) for s in range(N_SEEDS)]
        tf = np.array([r["test_f1"] for r in runs]); vf = np.array([r["val_f1"] for r in runs])
        print(f"\n--- rung {rung} ---  params {runs[0]['n_params']:,}")
        print(f"  val  macro-F1 {vf.mean():.4f} +/- {vf.std():.4f}")
        print(f"  TEST macro-F1 {tf.mean():.4f} +/- {tf.std():.4f}   (majority {maj_f1:.4f})")
        agg = {c: float(np.mean([r["per_class"][i]["f1"] for r in runs]))
               for i, c in enumerate(classes)}
        for i, c in enumerate(classes):
            print(f"    {c:<32} F1 {agg[c]:.3f}  n={runs[0]['per_class'][i]['n']}")
        results["rungs"][str(rung)] = {"n_params": runs[0]["n_params"],
                                       "val_f1_mean": round(float(vf.mean()), 4),
                                       "test_f1_mean": round(float(tf.mean()), 4),
                                       "test_f1_std": round(float(tf.std()), 4),
                                       "per_class_f1": {k: round(v, 4) for k, v in agg.items()},
                                       "seeds": N_SEEDS}
    out = Path(a.out) if a.out else REPO / "data" / f"ethogram_ladder_{a.version}.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print("\nREAD THIS BEFORE CONCLUDING: if rungs 1-3 land within ~1 std of each other, the ceiling "
          "is video diversity (~60 training videos), not architecture -- the same wall the "
          "segmentation work hit at IoU 0.47 across every model size. A bigger model will not fix it.")


if __name__ == "__main__":
    main()
