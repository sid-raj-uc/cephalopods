"""skel_phaseB_selection.py — Skeleton-accuracy Phase B: tune the arm-SELECTION heuristics.

Phase A showed selection discards 1.10 real skeleton endpoints/frame. Grid over the three selection
knobs on the 40 GT frames, scoring each config on BOTH:
  arms        mean selected arms (want up)
  tip_match   fraction of selected arm tips landing within r of a silhouette finger tip
              (want to STAY high — buying arms with junk shows up here)
Also evaluates the winning config on the MODEL masks (the deployment input).
Writes chart + rows (8018 UI) with baseline-vs-winner overlays.
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
import skeleton as SK
from skel_phaseA_loss import finger_tips
from segment_octopus import OctoSegmenter
from seg_skeleton_pipeline import DEFAULT_CKPT

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"
SMOOTHS = [0.45, 0.65, 0.90]
MAXDIM = 1024

GRID = [  # (floor_scale, floor_med, prefix_max)
    (4.0, 0.40, 0.58),   # baseline
    (2.5, 0.40, 0.58),
    (1.5, 0.30, 0.58),
    (2.5, 0.30, 0.70),
    (1.5, 0.30, 0.70),
    (1.0, 0.20, 0.70),
]


def arms_for(mask255, floor_scale, floor_med, prefix_max):
    """Best (by count) selected arm TIP positions across the smoothing schedule, full-res coords."""
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
            root = SK.choose_anatomical_root(pts, adj, dt, small)
            geod, parent = SK.dijkstra_tree(pts, adj, root, dt)
            try:
                paths = SK.select_arm_paths(pts, adj, root, parent, geod, dt, 1, 8,
                                            floor_scale=floor_scale, floor_med=floor_med,
                                            prefix_max=prefix_max)
            except Exception:
                paths = []
            tips = [(float(pts[p[-1], 1] * sx), float(pts[p[-1], 0] * sy)) for p in paths]
            if best is None or len(tips) > len(best):
                best = tips
        except Exception:
            continue
    return best or []


def tip_match(tips, fingers, r):
    if not tips:
        return 1.0
    ok = 0
    for tx, ty in tips:
        if any(math.hypot(tx - fx, ty - fy) <= r for fx, fy in fingers):
            ok += 1
    return ok / len(tips)


def main(n=40):
    OUT.mkdir(parents=True, exist_ok=True)
    rowsm = [json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()]
    rowsm = [r for r in rowsm if r.get("source") == "human" and r.get("image")]
    idx = np.linspace(0, len(rowsm) - 1, min(n, len(rowsm))).astype(int)
    frames = []
    for i in idx:
        r = rowsm[int(i)]
        gt = (cv2.imread(str(DS / r["mask"]), 0) > 127).astype(np.uint8) * 255
        fg = finger_tips(gt)
        diag = math.hypot(*gt.shape)
        frames.append({"r": r, "gt": gt, "fingers": fg, "match_r": 0.05 * diag})

    results = []
    for cfg in GRID:
        arms_all, match_all = [], []
        for f in frames:
            tips = arms_for(f["gt"], *cfg)
            arms_all.append(len(tips))
            match_all.append(tip_match(tips, f["fingers"], f["match_r"]))
        results.append({"cfg": cfg, "arms": float(np.mean(arms_all)),
                        "tip_match": float(np.mean(match_all))})
        print(f"  cfg floor={cfg[0]:.1f}/{cfg[1]:.2f} prefix={cfg[2]:.2f}  "
              f"arms {results[-1]['arms']:.2f}  tip_match {results[-1]['tip_match']:.3f}", flush=True)

    base = results[0]
    # winner: most arms among configs whose tip_match is within 0.03 of baseline
    ok = [r for r in results if r["tip_match"] >= base["tip_match"] - 0.03]
    win = max(ok, key=lambda r: r["arms"])
    print(f"\nbaseline arms {base['arms']:.2f} match {base['tip_match']:.3f}")
    print(f"winner   arms {win['arms']:.2f} match {win['tip_match']:.3f}  cfg {win['cfg']}")

    # winner on MODEL masks
    S = OctoSegmenter(str(DEFAULT_CKPT))
    mb, mw = [], []
    rows_ui = []
    for j, f in enumerate(frames):
        img = cv2.imread(str(DS / f["r"]["image"]))
        mm, _ = S.segment(img)
        m255 = (mm.astype(np.uint8)) * 255
        tb = arms_for(m255, *base["cfg"]); tw = arms_for(m255, *win["cfg"])
        mb.append(len(tb)); mw.append(len(tw))
        rows_ui.append({"file": f"{j:03d}.jpg", "left_arms": len(tb), "right_arms": len(tw)})
        vis = cv2.addWeighted(img, 0.55, np.zeros_like(img), 0.45, 0)
        for (x, y) in tb:
            cv2.circle(vis, (int(x), int(y)), 9, (80, 120, 255), 2)
        for (x, y) in tw:
            cv2.circle(vis, (int(x), int(y)), 5, (80, 255, 120), -1)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, f"model mask: baseline {len(tb)} (blue) vs winner {len(tw)} (green) tips",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(f"MODEL masks: baseline {np.mean(mb):.2f} -> winner {np.mean(mw):.2f} arms")

    json.dump({"meta": {"title": "Phase B — selection tuning (winner vs baseline, model masks)",
                        "left": "baseline arms", "right": "winner arms"},
               "rows": rows_ui}, open(OUT / "summary.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor="#111"); ax.set_facecolor("#111")
    xs = [r["arms"] for r in results]; ys = [r["tip_match"] for r in results]
    ax.plot(xs, ys, "o-", color="#4ea3ff")
    for r in results:
        ax.annotate(f"{r['cfg'][0]:.1f}/{r['cfg'][1]:.2f}/{r['cfg'][2]:.2f}",
                    (r["arms"], r["tip_match"]), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color="#ccc")
    ax.scatter([base["arms"]], [base["tip_match"]], c="#ff7a5c", s=90, zorder=5, label="baseline")
    ax.scatter([win["arms"]], [win["tip_match"]], c="#7ed47e", s=90, zorder=5, label="winner")
    ax.set_xlabel("mean selected arms (GT masks)", color="#ccc")
    ax.set_ylabel("tip-finger match rate", color="#ccc"); ax.tick_params(colors="#aaa")
    ax.set_title("Phase B: arms vs correctness across selection configs", color="#eee")
    ax.legend(facecolor="#222", labelcolor="#ddd")
    plt.tight_layout()
    plt.savefig(OUT / "chart.png", dpi=130, facecolor="#111")
    plt.savefig(HERE.parent / "results" / "segmentation" / "skel_phaseB_selection.png", dpi=130, facecolor="#111")
    json.dump({"grid": results, "winner": win, "model_baseline": float(np.mean(mb)),
               "model_winner": float(np.mean(mw))}, open(OUT / "phaseB_result.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
