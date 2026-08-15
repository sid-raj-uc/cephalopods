"""skel_bench_latest.py — render the LATEST full skeleton (arms A-C + anatomical head) on the
frozen 50-frame benchmark for the 8018 UI. Left number = arms, right = head-plausible (1/0)."""
import sys, json
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
from skel_head_fix import full_graph, plaus
from seg_skeleton_pipeline import _draw_skeleton, DEFAULT_CKPT
from segment_octopus import OctoSegmenter

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"


def main(refine=False):
    frames = json.load(open(HERE.parent / "data" / "skel_bench50" / "frames.json"))
    S = OctoSegmenter(str(DEFAULT_CKPT))
    rows, arms_all, head_all = [], [], []
    for j, f in enumerate(frames):
        img = cv2.imread(str(DS / f["image"]))
        mm, _ = S.segment(img)
        if refine:
            from mask_refine import sam2_refine
            from segment_octopus import _largest_blob
            mm = sam2_refine(img, mm, largest_blob=_largest_blob)
        m255 = (mm.astype(np.uint8)) * 255
        nodes, edges = full_graph(m255)
        vis = cv2.addWeighted(img, 0.62, np.zeros_like(img), 0.38, 0)
        mmask = m255 > 0
        vis[mmask] = (0.78 * vis[mmask] + 0.22 * np.array([60, 150, 60])).astype(np.uint8)
        if nodes is None:
            arms, hok = 0, 0
        else:
            _draw_skeleton(vis, nodes, edges, 2)
            arms = len({n["branch_id"] for n in nodes if n["branch_id"] > 0})
            c = next((n for n in nodes if n["is_center"]), None)
            hd = next((n for n in nodes if n.get("is_head")), None)
            bases = [(n["x"], n["y"]) for n in nodes if "Base" in n.get("body_part", "")]
            hok = int(bool(c and hd and len(bases) >= 2 and
                           plaus((hd["x"], hd["y"]), (c["x"], c["y"]),
                                 tuple(np.mean(np.asarray(bases, float), axis=0)))))
        arms_all.append(arms); head_all.append(hok)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, f"LATEST skeleton: {arms} arms | head {'ok' if hok else 'off'} "
                         f"(red=mantle green=head yellow=tips)",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append({"file": f"{j:03d}.jpg", "left_arms": arms, "right_arms": hok})
        print(f"  [{j+1}/{len(frames)}] arms {arms} head {'ok' if hok else 'off'}", flush=True)
    json.dump({"meta": {"title": "LATEST skeleton on frozen bench50 (arms A-C + anatomical head)",
                        "left": "arms", "right": "head ok"},
               "rows": rows}, open(OUT / "summary.json", "w"), indent=1)
    print(f"\nmean arms {np.mean(arms_all):.2f} | head ok {int(np.mean(head_all)*100)}%")


if __name__ == "__main__":
    import sys
    main(refine="--refine" in sys.argv)
