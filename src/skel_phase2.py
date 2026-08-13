"""skel_phase2.py — Phase 2: recover arms from the SAME model mask by tuning the skeletonizer's
mask-prep (less Gaussian/close smoothing + higher working resolution keeps close/thin arms distinct).

For each sampled human-GT frame, take our seg-model mask and skeletonize it TWICE: with the old
default config and with the thin-preserving config. Emits left(default)-vs-right(tuned) overlays,
a summary.json (generic {meta,rows}) and an arm-count chart into data/skel_diag/ for the port-8018 UI.
"""
import sys, json
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter
import skeleton as SK
from seg_skeleton_pipeline import _draw_skeleton, DEFAULT_CKPT

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"
DEFAULT = ([0.75, 1.00, 1.25], 760)          # old config
TUNED = ([0.35, 0.50, 0.70, 0.90], 1024)     # thin-preserving


def skel_cfg(mask255, smooths, max_dim, spline=0.5):
    """Best (score-selected) skeleton across the smoothing schedule -> (arm_count, nodes, edges)."""
    best = None
    for sm in smooths:
        try:
            dense = SK.dense_iteration(mask255, 1, max_dim, sm, 1, 8)
            br = SK.build_branches(dense, mask255, spline)
            nodes, edges = SK.construct_graph(br, mask255)
            met = SK.graph_metrics(nodes, edges, mask255, br)
            sc = SK.quality_score(met, 8)
            if best is None or sc > best[0]:
                best = (sc, met["arm_count"], nodes, edges)
        except Exception:
            pass
    return (best[1], best[2], best[3]) if best else (0, None, None)


def _panel(base, nodes, edges, title, arms):
    c = base.copy()
    if nodes:
        _draw_skeleton(c, nodes, edges, 2)
    cv2.rectangle(c, (0, 0), (c.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(c, f"{title}: {arms} arms", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return c


def main(n=40):
    OUT.mkdir(parents=True, exist_ok=True)
    S = OctoSegmenter(str(DEFAULT_CKPT))
    rows = [json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("source") == "human" and r.get("image")]
    idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
    summ = []
    for j, i in enumerate(idx):
        r = rows[int(i)]
        img = cv2.imread(str(DS / r["image"]))
        mm, _ = S.segment(img); m255 = (mm.astype(np.uint8)) * 255
        a0, n0, e0 = skel_cfg(m255, *DEFAULT)
        a1, n1, e1 = skel_cfg(m255, *TUNED)
        dim = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
        left = _panel(dim, n0, e0, "default", a0)
        right = _panel(dim, n1, e1, "thin-preserving", a1)
        gap = np.full((left.shape[0], 6, 3), 40, np.uint8)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), np.hstack([left, gap, right]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        summ.append({"file": f"{j:03d}.jpg", "left_arms": a0, "right_arms": a1})
        print(f"  [{j+1}/{len(idx)}] default {a0} -> tuned {a1}", flush=True)

    json.dump({"meta": {"title": "Phase 2 — skeletonizer mask-prep tuning (model masks)",
                        "left": "default config", "right": "thin-preserving"},
               "rows": summ}, open(OUT / "summary.json", "w"), indent=1)

    la = np.array([s["left_arms"] for s in summ]); ra = np.array([s["right_arms"] for s in summ])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bins = np.arange(-0.5, 9.5, 1)
    plt.figure(figsize=(9, 4.5), facecolor="#111"); ax = plt.gca(); ax.set_facecolor("#111")
    ax.hist(la, bins=bins, alpha=.7, label=f"default (mean {la.mean():.1f})", color="#ff7a5c")
    ax.hist(ra, bins=bins, alpha=.7, label=f"thin-preserving (mean {ra.mean():.1f})", color="#4ea3ff")
    ax.set_xlabel("arms detected (model mask)", color="#ccc"); ax.set_ylabel("# frames", color="#ccc")
    ax.set_title("Phase 2: mask-prep tuning recovers arms from the same model mask", color="#eee")
    ax.tick_params(colors="#aaa"); ax.legend(facecolor="#222", labelcolor="#ddd")
    plt.tight_layout()
    plt.savefig(OUT / "chart.png", dpi=130, facecolor="#111")
    plt.savefig(HERE.parent / "results" / "segmentation" / "skel_phase2_armcount.png", dpi=130, facecolor="#111")
    print(f"\ndefault mean {la.mean():.2f} -> tuned mean {ra.mean():.2f}  (delta {ra.mean()-la.mean():+.2f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
