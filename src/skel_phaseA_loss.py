"""skel_phaseA_loss.py — Skeleton-accuracy Phase A: WHERE are arms lost?

On the human-GT masks (clean silhouettes), decompose the arm loss per frame:
  fingers      silhouette protrusions visible in the mask (skeleton-independent proxy:
               contour points locally maximal in distance-from-mask-medial, i.e. "finger tips",
               via convexity/anchor analysis on the smoothed contour)
  endpoints    raw thinned-skeleton endpoints after spur removal (what thinning offers)
  selected     arms the selection heuristics keep (what the pipeline outputs)

If endpoints >> selected  -> the selection heuristics are the loss point (fixable in code).
If fingers  >> endpoints  -> thinning/mask-prep loses arms (fixable in prep).
If fingers ~= selected    -> the 2D silhouette genuinely lacks the rest (not fixable per-frame).

Writes per-frame overlay images + summary.json + chart into data/skel_diag/ (port-8018 UI).
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
import skeleton as SK

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"
SMOOTHS = [0.45, 0.65, 0.90]
MAXDIM = 1024


def finger_count(mask255, min_prominence=1.8, min_len_frac=0.06):
    """Skeleton-independent count of arm-like protrusions in a silhouette.

    A finger tip = a contour point whose geodesic 'reach' sticks out beyond the body: we take the
    largest contour, smooth it, and count locally-maximal distance-to-body peaks. 'Body' = the
    max-inscribed-circle centre (distance-transform argmax); prominence = tip distance relative to
    the local body radius. Deliberately generous — an upper bound on what a skeleton could find."""
    m = (mask255 > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(c) < 40:
        return 0
    dt = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
    body_r = float(dt[cy, cx])
    # distance of each contour point from the body centre, smoothed along the contour
    d = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    k = max(5, int(len(c) * 0.02) | 1)
    ker = np.ones(k) / k
    ds = np.convolve(np.r_[d[-k:], d, d[:k]], ker, mode="same")[k:-k]
    # local maxima with prominence over the body radius
    n = len(ds)
    peaks = []
    min_sep = int(n * min_len_frac)
    order = np.argsort(ds)[::-1]
    for i in order:
        if ds[i] < min_prominence * body_r:
            break
        if all(min(abs(i - j), n - abs(i - j)) >= min_sep for j in peaks):
            peaks.append(int(i))
    return len(peaks)


def stage_counts(mask255):
    """(endpoints_raw, selected_arms, nodes, edges) for one mask via the pipeline's own stages."""
    best = None
    for sm in SMOOTHS:
        try:
            small, sx, sy = SK.prepare_mask(mask255, MAXDIM, sm)
            dt = cv2.distanceTransform(small, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            skel = SK.zhang_suen_thinning(small)
            skel = SK.remove_tiny_spurs(skel, dt, passes=2)
            pts, adj, _ = SK.pixel_graph(skel)
            if len(pts) < 20:
                continue
            endpoints = int(sum(len(a) == 1 for a in adj))
            root = SK.choose_anatomical_root(pts, adj, dt, small)
            geod, parent = SK.dijkstra_tree(pts, adj, root, dt)
            try:
                paths = SK.select_arm_paths(pts, adj, root, parent, geod, dt, 1, 8)
            except Exception:
                paths = []
            score = len(paths)
            if best is None or score > best[0]:
                best = (score, endpoints, len(paths))
        except Exception:
            continue
    if best is None:
        return 0, 0
    return best[1], best[2]


def main(n=40):
    OUT.mkdir(parents=True, exist_ok=True)
    rows_manifest = [json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()]
    rows_manifest = [r for r in rows_manifest if r.get("source") == "human" and r.get("image")]
    idx = np.linspace(0, len(rows_manifest) - 1, min(n, len(rows_manifest))).astype(int)
    rows = []
    F, E, S = [], [], []
    for j, i in enumerate(idx):
        r = rows_manifest[int(i)]
        gt = (cv2.imread(str(DS / r["mask"]), 0) > 127).astype(np.uint8) * 255
        fingers = finger_count(gt)
        endpoints, selected = stage_counts(gt)
        F.append(fingers); E.append(endpoints); S.append(selected)
        # overlay: mask + annotation of the three counts
        img = cv2.imread(str(DS / r["image"]))
        vis = cv2.addWeighted(img, 0.55, np.zeros_like(img), 0.45, 0)
        mm = gt > 0
        vis[mm] = (0.6 * vis[mm] + 0.4 * np.array([60, 160, 60])).astype(np.uint8)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, f"fingers {fingers}  endpoints {endpoints}  selected {selected}",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 86])
        rows.append({"file": f"{j:03d}.jpg", "left_arms": endpoints, "right_arms": selected,
                     "fingers": fingers})
        print(f"  [{j+1}/{len(idx)}] fingers {fingers}  endpoints {endpoints}  selected {selected}", flush=True)

    F, E, S = np.array(F), np.array(E), np.array(S)
    json.dump({"meta": {"title": "Skeleton accuracy Phase A — where are arms lost? (GT masks)",
                        "left": "raw endpoints", "right": "selected arms"},
               "rows": rows}, open(OUT / "summary.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#111"); ax.set_facecolor("#111")
    bins = np.arange(-0.5, 14.5, 1)
    ax.hist(F, bins=bins, alpha=.62, label=f"silhouette fingers (mean {F.mean():.1f})", color="#7ed47e")
    ax.hist(E, bins=bins, alpha=.62, label=f"raw skeleton endpoints (mean {E.mean():.1f})", color="#4ea3ff")
    ax.hist(S, bins=bins, alpha=.62, label=f"selected arms (mean {S.mean():.1f})", color="#ff7a5c")
    ax.axvline(8, color="#aaa", ls="--", lw=1)
    ax.set_xlabel("count per frame (GT masks, n=%d)" % len(F), color="#ccc")
    ax.set_ylabel("# frames", color="#ccc"); ax.tick_params(colors="#aaa")
    ax.set_title("Phase A: arm loss decomposition", color="#eee")
    ax.legend(facecolor="#222", labelcolor="#ddd")
    plt.tight_layout()
    plt.savefig(OUT / "chart.png", dpi=130, facecolor="#111")
    plt.savefig(HERE.parent / "results" / "segmentation" / "skel_phaseA_loss.png", dpi=130, facecolor="#111")
    print(f"\nfingers mean {F.mean():.2f} | endpoints mean {E.mean():.2f} | selected mean {S.mean():.2f}")
    print(f"loss thinning->selection: {E.mean()-S.mean():+.2f}   silhouette->thinning: {F.mean()-E.mean():+.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
