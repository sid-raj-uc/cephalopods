"""skel_zoom_sam2.py — two untried OFFLINE mask upgrades for skeleton extraction, on bench50:

  A) ZOOM two-pass: student segments full frame -> crop octopus bbox -> re-segment the crop
     (octopus gets ~5x the pixels; tentacles become resolvable). No retraining.
  B) SAM2-REFINE: student locates -> SAM2 image predictor refines with positive points sampled
     inside the student mask (+ its bbox). Teacher-grade boundaries at ~1s/frame, offline-only.

Scores arms/frame + tip-match (vs human-GT protrusions) against the thin768 baseline.
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter, _largest_blob
from skel_bench50 import skeleton_paths, AFTER
from skel_phaseA_loss import finger_tips

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"
CKPT = HERE.parent / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def clean(m):
    if m.any():
        m = _largest_blob(m)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL).astype(bool)
    return (m.astype(np.uint8)) * 255


def zoom_mask(S, img, pad=0.25):
    """Pass 1 full frame -> bbox -> pass 2 on the crop -> paste back."""
    m1, _ = S.segment(img)
    if not m1.any():
        return clean(m1)
    ys, xs = np.where(m1)
    H, W = img.shape[:2]
    ph = int((ys.max() - ys.min()) * pad) + 8; pw = int((xs.max() - xs.min()) * pad) + 8
    y0, y1 = max(0, ys.min() - ph), min(H, ys.max() + ph)
    x0, x1 = max(0, xs.min() - pw), min(W, xs.max() + pw)
    crop = img[y0:y1, x0:x1]
    m2, _ = S.segment(crop)
    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = m2
    return clean(full)


_SAM = None
def sam2_refine(img, student_mask255):
    """SAM2 image predictor prompted by the student mask (box + interior positive points)."""
    global _SAM
    import torch
    if _SAM is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _SAM = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small", device=dev)
    m = student_mask255 > 0
    if not m.any():
        return student_mask255
    ys, xs = np.where(m)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    # positive points: interior maxima of the student mask's distance transform (body + arm anchors)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    pts = []
    dd = dt.copy()
    for _ in range(5):
        y, x = np.unravel_index(np.argmax(dd), dd.shape)
        if dd[y, x] <= 1:
            break
        pts.append([x, y])
        cv2.circle(dd, (int(x), int(y)), max(8, int(dt.max() * 0.8)), 0, -1)
    _SAM.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    masks, scores, _ = _SAM.predict(point_coords=np.array(pts, np.float32),
                                    point_labels=np.ones(len(pts), np.int32),
                                    box=box, multimask_output=False)
    return clean(masks[0].astype(bool))


def tip_match(tips, fingers, r):
    if not tips:
        return 1.0
    return sum(1 for t in tips if any(math.hypot(t[0] - fx, t[1] - fy) <= r
                                      for fx, fy in fingers)) / len(tips)


def main(n=50):
    frames = json.load(open(HERE.parent / "data" / "skel_bench50" / "frames.json"))[:n]
    S = OctoSegmenter(str(CKPT))
    res = {k: {"arms": [], "match": []} for k in ("base", "zoom", "sam2")}
    rows_ui = []
    for j, f in enumerate(frames):
        img = cv2.imread(str(DS / f["image"]))
        gt = (cv2.imread(str(DS / f["mask"]), 0) > 127).astype(np.uint8) * 255
        fingers = finger_tips(gt); r = 0.05 * math.hypot(*gt.shape)
        masks = {}
        masks["base"] = clean(S.segment(img)[0])
        masks["zoom"] = zoom_mask(S, img)
        try:
            masks["sam2"] = sam2_refine(img, masks["base"])
        except Exception as e:
            print(f"  sam2 fail: {e}", flush=True); masks["sam2"] = masks["base"]
        paths = {}
        for k, m in masks.items():
            p = skeleton_paths(m, AFTER)
            paths[k] = p
            res[k]["arms"].append(len(p))
            res[k]["match"].append(tip_match([tuple(q[-1]) for q in p], fingers, r))
        # overlay: base | zoom | sam2
        import skeleton as SK
        panels = []
        for k, ttl, col in (("base", "thin768 base", (100, 160, 255)),
                            ("zoom", "A) zoom 2-pass", (120, 255, 120)),
                            ("sam2", "B) SAM2-refined", (0, 215, 255))):
            vis = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
            mm = masks[k] > 0
            vis[mm] = (0.7 * vis[mm] + 0.3 * np.array([60, 150, 60])).astype(np.uint8)
            for a, p in enumerate(paths[k], 1):
                c = tuple(int(v) for v in SK.branch_color(a)[::-1])
                cv2.polylines(vis, [np.rint(p).astype(np.int32).reshape(-1, 1, 2)], False, c, 2, cv2.LINE_AA)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(vis, f"{ttl}: {len(paths[k])} arms", (6, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            panels.append(cv2.resize(vis, (vis.shape[1] * 2 // 3, vis.shape[0] * 2 // 3)))
        gap = np.full((panels[0].shape[0], 5, 3), 45, np.uint8)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), np.hstack([panels[0], gap, panels[1], gap, panels[2]]),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        rows_ui.append({"file": f"{j:03d}.jpg", "left_arms": len(paths["base"]),
                        "right_arms": max(len(paths["zoom"]), len(paths["sam2"]))})
        print(f"  [{j+1}/{len(frames)}] base {len(paths['base'])}  zoom {len(paths['zoom'])}  "
              f"sam2 {len(paths['sam2'])}", flush=True)

    print()
    for k in ("base", "zoom", "sam2"):
        print(f"  {k:5s}: arms {np.mean(res[k]['arms']):.2f}  tip_match {np.mean(res[k]['match']):.3f}")
    json.dump({"meta": {"title": "Offline mask upgrades — base vs best(zoom/SAM2)",
                        "left": "base arms", "right": "best-upgrade arms"},
               "rows": rows_ui}, open(OUT / "summary.json", "w"), indent=1)
    json.dump({k: {"arms": float(np.mean(res[k]["arms"])), "match": float(np.mean(res[k]["match"]))}
               for k in res}, open(OUT / "zoom_sam2_result.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
