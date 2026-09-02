"""skel_phaseC_prep.py — Skeleton-accuracy Phase C: recover arms lost in thinning/mask-prep.

Phase A: raw endpoints 5.95 vs ~7-8 visible arms — prep loses ~1.5-2. Suspects: the blur+threshold
in prepare_mask erases thin arms (a blurred 3px arm drops below 112), and remove_tiny_spurs eats
real short arms. Grid over (bin_thresh, spur width_factor, max_dimension) on the 40 GT masks,
scoring selected arms (with the Phase-B winner selection) + tip-match guard. Winner re-checked on
model masks. Outputs chart + rows for the 8018 UI.
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

GRID = [  # (bin_thresh, width_factor, max_dim)
    (112, 0.55, 1024),   # baseline (with Phase-B selection)
    (96, 0.55, 1024),
    (80, 0.55, 1024),
    (96, 0.35, 1024),
    (80, 0.35, 1024),
    (96, 0.35, 1280),
]


def arms_for(mask255, bin_thresh, width_factor, max_dim):
    best = None
    for sm in SMOOTHS:
        try:
            small, sx, sy = SK.prepare_mask(mask255, max_dim, sm, bin_thresh=bin_thresh)
            dt = cv2.distanceTransform(small, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            skel = SK.zhang_suen_thinning(small)
            skel = SK.remove_tiny_spurs(skel, dt, passes=2, width_factor=width_factor)
            pts, adj, _ = SK.pixel_graph(skel)
            if len(pts) < 20:
                continue
            root = SK.choose_anatomical_root(pts, adj, dt, small)
            geod, parent = SK.dijkstra_tree(pts, adj, root, dt)
            try:
                paths = SK.select_arm_paths(pts, adj, root, parent, geod, dt, 1, 8)
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
    return sum(1 for tx, ty in tips
               if any(math.hypot(tx - fx, ty - fy) <= r for fx, fy in fingers)) / len(tips)


def main(n=40):
    OUT.mkdir(parents=True, exist_ok=True)
    rowsm = [json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()]
    rowsm = [r for r in rowsm if r.get("source") == "human" and r.get("image")]
    idx = np.linspace(0, len(rowsm) - 1, min(n, len(rowsm))).astype(int)
    frames = []
    for i in idx:
        r = rowsm[int(i)]
        gt = (cv2.imread(str(DS / r["mask"]), 0) > 127).astype(np.uint8) * 255
        frames.append({"r": r, "gt": gt, "fingers": finger_tips(gt),
                       "match_r": 0.05 * math.hypot(*gt.shape)})

    results = []
    for cfg in GRID:
        arms, match = [], []
        for f in frames:
            tips = arms_for(f["gt"], *cfg)
            arms.append(len(tips)); match.append(tip_match(tips, f["fingers"], f["match_r"]))
        results.append({"cfg": cfg, "arms": float(np.mean(arms)), "tip_match": float(np.mean(match))})
        print(f"  thr={cfg[0]} wf={cfg[1]} dim={cfg[2]}  arms {results[-1]['arms']:.2f}  "
              f"tip_match {results[-1]['tip_match']:.3f}", flush=True)

    base = results[0]
    ok = [r for r in results if r["tip_match"] >= base["tip_match"] - 0.03]
    win = max(ok, key=lambda r: r["arms"])
    print(f"\nbaseline arms {base['arms']:.2f} match {base['tip_match']:.3f}")
    print(f"winner   arms {win['arms']:.2f} match {win['tip_match']:.3f}  cfg {win['cfg']}")

    S = OctoSegmenter(str(DEFAULT_CKPT))
    mb, mw, rows_ui = [], [], []
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
        cv2.putText(vis, f"model mask: phaseB {len(tb)} (blue) vs +phaseC {len(tw)} (green)",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(f"MODEL masks: phaseB {np.mean(mb):.2f} -> +phaseC {np.mean(mw):.2f} arms")

    json.dump({"meta": {"title": "Phase C — mask-prep/thinning tuning (winner vs Phase-B, model masks)",
                        "left": "phaseB arms", "right": "+phaseC arms"},
               "rows": rows_ui}, open(OUT / "summary.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor="#111"); ax.set_facecolor("#111")
    ax.plot([r["arms"] for r in results], [r["tip_match"] for r in results], "o-", color="#4ea3ff")
    for r in results:
        ax.annotate(f"{r['cfg'][0]}/{r['cfg'][1]}/{r['cfg'][2]}", (r["arms"], r["tip_match"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7.5, color="#ccc")
    ax.scatter([base["arms"]], [base["tip_match"]], c="#ff7a5c", s=90, zorder=5, label="baseline")
    ax.scatter([win["arms"]], [win["tip_match"]], c="#7ed47e", s=90, zorder=5, label="winner")
    ax.set_xlabel("mean selected arms (GT masks)", color="#ccc")
    ax.set_ylabel("tip-finger match rate", color="#ccc"); ax.tick_params(colors="#aaa")
    ax.set_title("Phase C: prep/thinning configs — arms vs correctness", color="#eee")
    ax.legend(facecolor="#222", labelcolor="#ddd")
    plt.tight_layout()
    plt.savefig(OUT / "chart.png", dpi=130, facecolor="#111")
    plt.savefig(HERE.parent / "results" / "segmentation" / "skel_phaseC_prep.png", dpi=130, facecolor="#111")
    json.dump({"grid": results, "winner": win, "model_baseline": float(np.mean(mb)),
               "model_winner": float(np.mean(mw))}, open(OUT / "phaseC_result.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
