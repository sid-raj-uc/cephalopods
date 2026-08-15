"""skel_head_eval.py — score head-placement candidates against HUMAN head clicks (head_gt.json).

Error metric: distance(prediction, human click) in units of the body radius (distance-transform
max), so it is pose/scale independent. Candidates:
  current     anatomical_head (neck constriction, nudged crown-side)  — the incumbent
  bump        the width LOCAL MAX after the neck (the head/eye bulge, not the constriction)
  dark        darkest blob in the mantle->crown corridor (eyes are dark; uses appearance)
  frac        fixed t=0.8 along mantle->crown (dumb control)
Writes per-frame overlays (GT magenta ring, candidates coloured) + summary for the 8018 UI.
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
import skeleton as SK
from skel_head_fix import full_graph
from segment_octopus import OctoSegmenter
from seg_skeleton_pipeline import DEFAULT_CKPT

OUT = HERE.parent / "data" / "skel_diag"
DS = HERE.parent / "data" / "dataset_seg_human"
GT = HERE.parent / "data" / "skel_bench50" / "head_gt.json"


def _line(mask, dt, m, crown, n=60):
    ts = np.linspace(0.0, 1.0, n)
    pts = np.asarray(m, float)[None, :] + ts[:, None] * (np.asarray(crown, float) - np.asarray(m, float))[None, :]
    pts = SK.mask_constrain_polyline(pts, mask)
    h, w = mask.shape
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return pts, dt[yi, xi]


def cand_current(mask, dt, grey, m, bases):
    r = SK.anatomical_head(mask, dt, m, bases)
    return None if r is None else (r[0], r[1])


def cand_bump(mask, dt, grey, m, bases):
    crown = np.mean(np.asarray(bases, float), axis=0)
    pts, widths = _line(mask, dt, m, crown)
    lo, hi = int(0.25 * len(pts)), int(0.92 * len(pts))
    neck = lo + int(np.argmin(widths[lo:hi]))
    if neck + 2 >= hi:
        return float(pts[neck, 0]), float(pts[neck, 1])
    bump = neck + int(np.argmax(widths[neck:hi]))
    return float(pts[bump, 0]), float(pts[bump, 1])


def cand_dark(mask, dt, grey, m, bases):
    crown = np.mean(np.asarray(bases, float), axis=0)
    mm = mask > 0
    h, w = mask.shape
    root_r = float(dt.max())
    # corridor: within 1.2*root_r of the mantle->crown segment, inside the mask
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.asarray(m, float); b = np.asarray(crown, float); ab = b - a
    L2 = max(float(ab @ ab), 1e-6)
    t = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2, 0, 1)
    dx = xx - (a[0] + t * ab[0]); dy = yy - (a[1] + t * ab[1])
    corridor = mm & (dx * dx + dy * dy <= (1.2 * root_r) ** 2) & (t > 0.2)
    if not corridor.any():
        return None
    k = max(3, int(root_r * 0.5) | 1)
    blur = cv2.blur(grey.astype(np.float32), (k, k))
    blur[~corridor] = 1e9
    y, x = np.unravel_index(np.argmin(blur), blur.shape)
    return float(x), float(y)


def cand_frac(mask, dt, grey, m, bases):
    crown = np.mean(np.asarray(bases, float), axis=0)
    pts, _ = _line(mask, dt, m, crown)
    k = int(0.8 * (len(pts) - 1))
    return float(pts[k, 0]), float(pts[k, 1])


CANDS = [("current", cand_current, (0, 165, 255)),   # orange
         ("bump", cand_bump, (0, 255, 255)),         # yellow
         ("dark", cand_dark, (80, 255, 120)),        # green
         ("frac", cand_frac, (255, 120, 80))]        # blue-ish


def main():
    gt = json.load(open(GT))
    frames = json.load(open(HERE.parent / "data" / "skel_bench50" / "frames.json"))
    S = OctoSegmenter(str(DEFAULT_CKPT))
    errs = {name: [] for name, _, _ in CANDS}
    rows = []
    j = 0
    for f in frames:
        p = str(DS / f["image"])
        if p not in gt:
            continue
        img = cv2.imread(p)
        mm, _ = S.segment(img)
        m255 = (mm.astype(np.uint8)) * 255
        nodes, edges = full_graph(m255)
        if nodes is None:
            continue
        c = next((n for n in nodes if n["is_center"]), None)
        bases = [(n["x"], n["y"]) for n in nodes if "Base" in n.get("body_part", "")]
        if c is None or len(bases) < 2:
            continue
        dt = cv2.distanceTransform(m255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        root_r = max(float(dt.max()), 1.0)
        gx, gy = gt[p]
        vis = cv2.addWeighted(img, 0.65, np.zeros_like(img), 0.35, 0)
        sel = m255 > 0
        vis[sel] = (0.8 * vis[sel] + 0.2 * np.array([60, 150, 60])).astype(np.uint8)
        cv2.circle(vis, (int(gx), int(gy)), 12, (255, 0, 255), 2, cv2.LINE_AA)
        errs_frame = {}
        for name, fn, col in CANDS:
            try:
                r = fn(m255, dt, grey, (c["x"], c["y"]), bases)
            except Exception:
                r = None
            if r is None:
                continue
            e = math.hypot(r[0] - gx, r[1] - gy) / root_r
            errs[name].append(e); errs_frame[name] = e
            cv2.drawMarker(vis, (int(r[0]), int(r[1])), col, cv2.MARKER_TRIANGLE_UP, 16, 2)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, "GT=magenta | current=orange bump=yellow dark=green frac=blue | " +
                    " ".join(f"{k}:{v:.2f}R" for k, v in errs_frame.items()),
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 87])
        rows.append({"file": f"{j:03d}.jpg",
                     "left_arms": round(errs_frame.get("current", 9), 2),
                     "right_arms": round(min([v for k, v in errs_frame.items() if k != "current"],
                                             default=9), 2)})
        j += 1
        print(f"  [{j}] " + "  ".join(f"{k} {v:.2f}R" for k, v in errs_frame.items()), flush=True)

    print(f"\nn={j} frames with GT")
    for name in errs:
        e = np.array(errs[name])
        if len(e):
            print(f"  {name:8s} mean {e.mean():.2f}R  median {np.median(e):.2f}R  "
                  f"hit@0.75R {(e <= 0.75).mean()*100:.0f}%")
    json.dump({"meta": {"title": "Head placement vs human GT (error in body radii)",
                        "left": "current err (R)", "right": "best-other err (R)"},
               "rows": rows}, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
