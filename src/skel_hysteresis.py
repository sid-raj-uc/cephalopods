"""skel_hysteresis.py — skeletonize the seg PROBABILITY field via hysteresis, not the 0.5 mask.

The seg model sees thin tentacles as p~0.2-0.45 ridges; binarizing at 0.5 erases them. Hysteresis
keeps weak (p >= weak_thr) pixels only when 8-connected to strong (p >= 0.5) — recovering tentacle
continuations while dropping background-only blobs. Grid over weak_thr on the frozen bench50,
scoring arms + tip-match against the HUMAN-GT mask's protrusions (truth). Overlays -> 8018 UI.
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter, _largest_blob
from seg_skeleton_pipeline import DEFAULT_CKPT
from skel_bench50 import skeleton_paths, AFTER
from skel_phaseA_loss import finger_tips

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"
WEAK_GRID = [0.5, 0.35, 0.30, 0.25, 0.20]     # 0.5 == current binary baseline


def hysteresis_mask(prob_full, weak):
    strong = prob_full >= 0.5
    if not strong.any():
        return np.zeros(prob_full.shape, np.uint8)
    if weak >= 0.5:
        m = strong
    else:
        wk = (prob_full >= weak).astype(np.uint8)
        n, lab = cv2.connectedComponents(wk, 8)
        keep = np.zeros(n, bool)
        keep[np.unique(lab[strong])] = True
        keep[0] = False
        m = keep[lab]
    m = _largest_blob(m)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    return (m.astype(np.uint8)) * 255


def tip_match(tips, fingers, r):
    if not tips:
        return 1.0
    return sum(1 for t in tips if any(math.hypot(t[0] - fx, t[1] - fy) <= r
                                      for fx, fy in fingers)) / len(tips)


def main():
    frames = json.load(open(HERE.parent / "data" / "skel_bench50" / "frames.json"))
    S = OctoSegmenter(str(DEFAULT_CKPT))
    per = {w: {"arms": [], "match": []} for w in WEAK_GRID}
    cache = []
    for j, f in enumerate(frames):
        img = cv2.imread(str(DS / f["image"]))
        H, W = img.shape[:2]
        prob = S.prob(img)
        prob_full = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        gt = (cv2.imread(str(DS / f["mask"]), 0) > 127).astype(np.uint8) * 255
        fingers = finger_tips(gt)
        r = 0.05 * math.hypot(H, W)
        row = {"img": img, "masks": {}, "tips": {}}
        for w in WEAK_GRID:
            m = hysteresis_mask(prob_full, w)
            paths = skeleton_paths(m, AFTER)
            tips = [tuple(p[-1]) for p in paths]
            per[w]["arms"].append(len(paths))
            per[w]["match"].append(tip_match(tips, fingers, r))
            row["masks"][w] = m; row["tips"][w] = paths
        cache.append(row)
        print(f"  [{j+1}/{len(frames)}] " +
              "  ".join(f"w{w}:{per[w]['arms'][-1]}" for w in WEAK_GRID), flush=True)

    print()
    base_match = float(np.mean(per[0.5]["match"]))
    results = []
    for w in WEAK_GRID:
        a, m = float(np.mean(per[w]["arms"])), float(np.mean(per[w]["match"]))
        results.append({"weak": w, "arms": a, "match": m})
        print(f"  weak={w}: arms {a:.2f}  tip_match {m:.3f}")
    ok = [r for r in results if r["match"] >= base_match - 0.03]
    win = max(ok, key=lambda r: r["arms"])
    print(f"\nbaseline (0.5): arms {results[0]['arms']:.2f} match {base_match:.3f}")
    print(f"winner  ({win['weak']}): arms {win['arms']:.2f} match {win['match']:.3f}")

    # UI overlays: baseline mask+skeleton vs winner
    rows_ui = []
    for j, row in enumerate(cache):
        img = row["img"]
        panels = []
        for w, ttl, col in [(0.5, "binary 0.5", (100, 160, 255)),
                            (win["weak"], f"hysteresis {win['weak']}", (120, 255, 120))]:
            vis = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
            mm = row["masks"][w] > 0
            vis[mm] = (0.7 * vis[mm] + 0.3 * np.array([60, 150, 60])).astype(np.uint8)
            for a, p in enumerate(row["tips"][w], 1):
                import skeleton as SK
                c = tuple(int(v) for v in SK.branch_color(a)[::-1])
                cv2.polylines(vis, [np.rint(p).astype(np.int32).reshape(-1, 1, 2)], False, c, 2, cv2.LINE_AA)
                cv2.circle(vis, (int(p[-1][0]), int(p[-1][1])), 6, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(vis, f"{ttl}: {len(row['tips'][w])} arms", (6, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
            panels.append(vis)
        gap = np.full((panels[0].shape[0], 6, 3), 45, np.uint8)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), np.hstack([panels[0], gap, panels[1]]),
                    [cv2.IMWRITE_JPEG_QUALITY, 87])
        rows_ui.append({"file": f"{j:03d}.jpg",
                        "left_arms": len(row["tips"][0.5]),
                        "right_arms": len(row["tips"][win["weak"]])})
    json.dump({"meta": {"title": f"Hysteresis prob-field masks (weak={win['weak']}) vs binary 0.5",
                        "left": "binary arms", "right": "hysteresis arms"},
               "rows": rows_ui}, open(OUT / "summary.json", "w"), indent=1)
    json.dump({"grid": results, "winner": win}, open(OUT / "hysteresis_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
