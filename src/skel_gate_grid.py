"""skel_gate_grid.py — tune the anti-mess arm-selection gates for tip-F1 on SKEL-50.

The gates (unique-suffix + tip-thinness) were introduced to kill duplicate/tangle arms after visual
inspection, with no recall metric available. The frozen-benchmark tip-F1 then showed the cost:
precision 0.71 but recall 0.35 (F1 0.42) — the gates are too strict and drop real arms.

This grid re-scores gate settings on the same 50 frames. Masks are segmented ONCE and cached, so
each extra config only re-runs skeletonisation. Reports precision/recall/F1/arms per config and
names the F1-optimal setting.

Usage: venv/bin/python3 src/skel_gate_grid.py [--refine]
"""
import argparse, json, math, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
import skeleton as SK
from segment_octopus import OctoSegmenter, _largest_blob
from skel_phaseA_loss import finger_tips
from seg_skeleton_pipeline import DEFAULT_CKPT
from benchmarks import _match, MATCH_FRAC, MAX_GT_TIPS, GT_MIN_PROMINENCE, GT_MIN_LEN_FRAC

DS = REPO / "data" / "dataset_seg_human"
BENCH50 = REPO / "data" / "skel_bench50" / "frames.json"
OUT = REPO / "data" / "skel_diag"
SMOOTHS = [0.45, 0.65, 0.90]

# (min_unique_scale, min_unique_frac, tip_ratio); current shipped = (2.0, 0.30, 0.55)
GRID = [
    (0.0, 0.00, 9.99),    # gates OFF (pre-anti-mess behaviour)
    (2.0, 0.30, 0.55),    # current shipped
    (1.5, 0.20, 0.70),
    (1.0, 0.15, 0.85),
    (1.0, 0.10, 1.00),
    (0.5, 0.10, 1.20),
    (1.5, 0.20, 1.00),
]


def paths_for(mask255, cfg):
    """Best-scoring skeleton across the smoothing schedule -> (tips, small-res paths, points)."""
    us, uf, tr = cfg
    best = None
    for sm in SMOOTHS:
        try:
            small, sx, sy = SK.prepare_mask(mask255, 1024, sm)
            dt = cv2.distanceTransform(small, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            skel = SK.zhang_suen_thinning(small)
            skel = SK.remove_tiny_spurs(skel, dt, passes=2)
            pts, adj, _ = SK.pixel_graph(skel)
            if len(pts) < 20:
                continue
            root = SK.choose_anatomical_root(pts, adj, dt, small)
            geod, parent = SK.dijkstra_tree(pts, adj, root, dt)
            try:
                paths = SK.select_arm_paths(pts, adj, root, parent, geod, dt, 1, 8,
                                            min_unique_scale=us, min_unique_frac=uf, tip_ratio=tr)
            except Exception:
                paths = []
            tips = [(float(pts[p[-1], 1] * sx), float(pts[p[-1], 0] * sy)) for p in paths]
            if best is None or len(tips) > len(best[0]):
                best = (tips, paths, pts)
        except Exception:
            continue
    return best or ([], [], None)


def duplicate_rate(paths, pts):
    """Fraction of selected arms whose UNSHARED suffix is < 30% of their own length.

    This is the quantity the anti-mess gates target and the thing a human sees as 'tangle':
    two arms that run together from the root and fork only near the tip. tip-F1 does NOT
    penalise it (both tips can still land on distinct real protrusions), so it must be
    reported alongside F1 — cleanliness and correctness are different axes."""
    if len(paths) < 2 or pts is None:
        return 0.0
    def arc(idx):
        if len(idx) < 2:
            return 0.0
        q = pts[idx].astype(float)
        return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    dup = 0
    for i, a in enumerate(paths):
        share = 0
        for j, b in enumerate(paths):
            if i == j:
                continue
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            share = max(share, k)
        total = arc(a)
        uniq = arc(a[max(share - 1, 0):])
        if total > 0 and uniq / total < 0.30:
            dup += 1
    return dup / len(paths)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()
    frames = json.load(open(BENCH50))
    S = OctoSegmenter(str(DEFAULT_CKPT))
    cache = []
    print(f"segmenting {len(frames)} frames once (refine={args.refine}) …", flush=True)
    for f in frames:
        img = cv2.imread(str(DS / f["image"]))
        gtm = (cv2.imread(str(DS / f["mask"]), 0) > 127).astype(np.uint8) * 255
        mm, _ = S.segment(img)
        if args.refine and mm.any():
            from mask_refine import sam2_refine
            mm = sam2_refine(img, mm, largest_blob=_largest_blob)
        cache.append({"m255": (mm.astype(np.uint8)) * 255,
                      "gt": finger_tips(gtm, min_prominence=GT_MIN_PROMINENCE,
                                        min_len_frac=GT_MIN_LEN_FRAC)[:MAX_GT_TIPS],
                      "r": MATCH_FRAC * math.hypot(*gtm.shape)})

    results = []
    for cfg in GRID:
        P, R, F, C, D = [], [], [], [], []
        for c in cache:
            tips, paths, pts = paths_for(c["m255"], cfg)
            D.append(duplicate_rate(paths, pts))
            nm = _match(tips, c["gt"], c["r"])
            p = nm / len(tips) if tips else (1.0 if not c["gt"] else 0.0)
            rc = nm / len(c["gt"]) if c["gt"] else 1.0
            P.append(p); R.append(rc); F.append(0.0 if p + rc == 0 else 2 * p * rc / (p + rc))
            C.append(len(tips))
        row = {"cfg": list(cfg), "precision": round(float(np.mean(P)), 4),
               "recall": round(float(np.mean(R)), 4), "f1": round(float(np.mean(F)), 4),
               "arms": round(float(np.mean(C)), 2), "dup_rate": round(float(np.mean(D)), 4)}
        results.append(row)
        print(f"  uniq={cfg[0]:.1f}/{cfg[1]:.2f} tip_ratio={cfg[2]:.2f}  "
              f"P {row['precision']:.3f}  R {row['recall']:.3f}  F1 {row['f1']:.3f}  "
              f"dup {row['dup_rate']:.3f}  arms {row['arms']:.2f}",
              flush=True)

    win = max(results, key=lambda r: r["f1"])
    cur = next(r for r in results if r["cfg"] == [2.0, 0.30, 0.55])
    print(f"\ncurrent  F1 {cur['f1']:.3f} (P {cur['precision']:.3f} R {cur['recall']:.3f} arms {cur['arms']:.2f})")
    print(f"BEST     F1 {win['f1']:.3f} (P {win['precision']:.3f} R {win['recall']:.3f} arms {win['arms']:.2f})"
          f"  cfg={win['cfg']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"grid": results, "winner": win, "current": cur, "refine": args.refine},
              open(OUT / "gate_grid_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
