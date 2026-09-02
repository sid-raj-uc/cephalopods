"""skel_phase3.py — Phase 3: temporal arm-template aggregation.

A single 2D silhouette of a curled octopus can't show all 8 arms, but different arms are visible in
different frames. The current sequence seeds from the FIRST present frame and the temporal fit can
only keep/drop arms -> the whole clip is capped by the (often poor) opening pose. This phase seeds
from the BEST-resolved frame (max detected arms) and propagates BOTH directions, so the clip keeps
the richer template. Compares per-clip median arm count: first-seed vs best-seed. Writes a chart +
sequence overlay montages into data/skel_diag/ for the port-8018 UI.
"""
import sys, json, math, glob
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter
from segment_to_skeleton import segment_masks, union_bbox
from seg_skeleton_pipeline import _draw_skeleton, DEFAULT_CKPT
from multi_frame import process_frame, arm_signatures, match_arms, relabel, temporal_fit

OUT = HERE.parent / "data" / "skel_diag"


def _detect_all(crops):
    """process_frame each crop once -> cached (arm_count, nodes, edges). Reused by both methods."""
    det = []
    for cm in crops:
        try:
            n, e, m, _ = process_frame(cm, 2, 1024, 1, 8, None)
            det.append((int(m["arm_count"]), n, e))
        except Exception:
            det.append((0, None, None))
    return det


def _propagate(order, crops, det):
    """Walk `order` (a frame-index sequence starting at the seed), temporal-fitting each step.
    Returns {frame_idx: (nodes, edges)}."""
    out = {}
    prev_nodes = prev_mask = prev_sig = None
    for k in order:
        cm = crops[k]; d_arm, dn, de = det[k]
        if prev_nodes is None:
            if dn is None:
                continue
            nodes, edges = dn, de
        else:
            if dn is not None:
                relabel(dn, de, match_arms(prev_sig or {}, arm_signatures(dn, de), math.hypot(*cm.shape)))
            try:
                nodes, edges, _, _ = temporal_fit(prev_nodes, prev_mask, dn, cm)
            except Exception:
                continue
        prev_nodes, prev_mask, prev_sig = nodes, edges and nodes, cm.copy()
        prev_sig = arm_signatures(nodes, edges)
        out[k] = (nodes, edges)
    return out


def sequence_arms(crops, det, mode):
    present = [k for k in range(len(crops)) if det[k][1] is not None]
    if len(present) < 3:
        return None, {}
    if mode == "first":
        order = present                                   # seed = first present, forward only
    else:
        seed = max(present, key=lambda k: det[k][0])      # seed = most-arms frame
        fwd = [k for k in present if k >= seed]
        bwd = [k for k in present if k < seed][::-1]
        order = fwd + bwd                                  # both directions from the seed
    graphs = _propagate(order, crops, det)
    arms = [len({n["branch_id"] for n in g[0] if n["branch_id"] > 0}) for g in graphs.values()]
    return (float(np.median(arms)) if arms else 0.0), graphs


def main(n_clips=6, fps=3.0):
    OUT.mkdir(parents=True, exist_ok=True)
    S = OctoSegmenter(str(DEFAULT_CKPT))
    br = json.load(open(HERE.parent / "data" / "behaviour_records.json"))
    clips = []
    for rel in br:
        p = HERE.parent / "src" / rel
        if p.exists() and any(c in rel for c in ("Right_Front", "Right_Back", "Right_Right")):
            clips.append(p)
        if len(clips) >= n_clips:
            break

    rows, first_meds, best_meds = [], [], []
    for ci, clip in enumerate(clips):
        masks, sfps, step = segment_masks(str(clip), S, fps, 0.004)
        pm = [(k, m) for k, m in enumerate(masks) if m is not None]
        if len(pm) < 4:
            continue
        y0, y1, x0, x1 = union_bbox([m for _, m in pm])
        crops = [(m[y0:y1, x0:x1].astype(np.uint8)) * 255 for _, m in pm]
        det = _detect_all(crops)
        med_first, _ = sequence_arms(crops, det, "first")
        med_best, gbest = sequence_arms(crops, det, "best")
        if med_first is None or med_best is None:
            continue
        first_meds.append(med_first); best_meds.append(med_best)
        rows.append({"file": f"seq_{ci:02d}.jpg", "left_arms": round(med_first, 1),
                     "right_arms": round(med_best, 1)})
        # montage: 4 frames of the best-seed tracked sequence
        keys = sorted(gbest)
        pick = [keys[int(t)] for t in np.linspace(0, len(keys) - 1, min(4, len(keys)))]
        tiles = []
        for k in pick:
            nodes, edges = gbest[k]
            c = _draw_skeleton(np.full((crops[k].shape[0], crops[k].shape[1], 3), 25, np.uint8),
                               nodes, edges, 2)
            th = 300; s = th / c.shape[0]; tiles.append(cv2.resize(c, (int(c.shape[1] * s), th)))
        if tiles:
            Wm = max(t.shape[1] for t in tiles)
            row = np.hstack([np.pad(t, ((0, 0), (0, Wm - t.shape[1]), (0, 0))) for t in tiles])
            cv2.imwrite(str(OUT / f"seq_{ci:02d}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  clip {ci}: first-seed median {med_first} -> best-seed median {med_best}", flush=True)

    json.dump({"meta": {"title": "Phase 3 — temporal arm-template (first-seed vs best-seed)",
                        "left": "first-seed median", "right": "best-seed median"}, "rows": rows},
              open(OUT / "summary.json", "w"), indent=1)
    fm, bm = np.array(first_meds), np.array(best_meds)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.5), facecolor="#111"); ax = plt.gca(); ax.set_facecolor("#111")
    x = np.arange(len(fm)); ax.bar(x - 0.2, fm, 0.4, label=f"first-seed (mean {fm.mean():.2f})", color="#ff7a5c")
    ax.bar(x + 0.2, bm, 0.4, label=f"best-seed (mean {bm.mean():.2f})", color="#4ea3ff")
    ax.set_xlabel("clip", color="#ccc"); ax.set_ylabel("median arms over sequence", color="#ccc")
    ax.set_title("Phase 3: best-frame seeding maintains more arms across a clip", color="#eee")
    ax.tick_params(colors="#aaa"); ax.legend(facecolor="#222", labelcolor="#ddd")
    plt.tight_layout()
    plt.savefig(OUT / "chart.png", dpi=130, facecolor="#111")
    plt.savefig(HERE.parent / "results" / "segmentation" / "skel_phase3_seed.png", dpi=130, facecolor="#111")
    print(f"\nfirst-seed mean median {fm.mean():.2f} -> best-seed {bm.mean():.2f}  (delta {bm.mean()-fm.mean():+.2f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
