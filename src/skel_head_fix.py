"""skel_head_fix.py — diagnose + fix Head-node placement on the frozen 50-frame benchmark.

CURRENT head = 2nd-highest distance-transform peak (any wide blob — often a curled arm mass), so it
lands anywhere. ANATOMICAL prior: the head lies BETWEEN the mantle and the arm crown, at the neck
constriction. Fix: head = the width-minimum ("neck") along the mantle->crown medial line, nudged
crown-side, mask-constrained.

Plausibility metric (per frame): head's projection parameter t onto the mantle->crown segment must
be in [0.15, 1.05] and its perpendicular offset < 0.8 x |mantle-crown| — i.e. "roughly between
mantle and arms". Reports the rate BEFORE vs AFTER + side-by-side overlays for the 8018 UI.
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
import skeleton as SK
from segment_octopus import OctoSegmenter
from seg_skeleton_pipeline import DEFAULT_CKPT

OUT = HERE.parent / "data" / "skel_diag"
BENCH = HERE.parent / "data" / "skel_bench50" / "frames.json"
DS = HERE.parent / "data" / "dataset_seg_human"
SMOOTHS = [0.45, 0.65, 0.90]


def full_graph(mask255):
    """(nodes, edges) from the single-frame pipeline at the current tuned defaults, or None."""
    best = None
    for sm in SMOOTHS:
        try:
            dense = SK.dense_iteration(mask255, 1, 1024, sm, 1, 8)
            br = SK.build_branches(dense, mask255, 0.55)
            nodes, edges = SK.construct_graph(br, mask255)
            met = SK.graph_metrics(nodes, edges, mask255, br)
            sc = SK.quality_score(met, 8)
            if best is None or sc > best[0]:
                best = (sc, nodes, edges)
        except Exception:
            continue
    return (best[1], best[2]) if best else (None, None)


def anatomical_head(mask255, mantle_xy, base_xys, dt=None):
    """Head = neck constriction along the mantle->arm-crown medial line, nudged crown-side."""
    if dt is None:
        dt = cv2.distanceTransform(mask255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    h, w = mask255.shape
    crown = np.mean(np.asarray(base_xys, float), axis=0)
    m = np.asarray(mantle_xy, float)
    L = np.linalg.norm(crown - m)
    if L < 4:
        return None
    ts = np.linspace(0.0, 1.0, 60)
    pts = m[None, :] + ts[:, None] * (crown - m)[None, :]
    pts = SK.mask_constrain_polyline(pts, mask255)
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    widths = dt[yi, xi]
    lo, hi = int(0.25 * len(ts)), int(0.92 * len(ts))
    neck = lo + int(np.argmin(widths[lo:hi]))
    k = min(len(ts) - 1, neck + int(0.08 * len(ts)))      # nudge crown-side of the constriction
    hx, hy = float(pts[k, 0]), float(pts[k, 1])
    return hx, hy, float(dt[int(round(hy)), int(round(hx))])


def plaus(head_xy, mantle_xy, crown_xy):
    m, c, hh = map(lambda p: np.asarray(p, float), (mantle_xy, crown_xy, head_xy))
    v = c - m; L = np.linalg.norm(v)
    if L < 4:
        return False
    t = float(np.dot(hh - m, v) / (L * L))
    perp = float(np.linalg.norm((hh - m) - t * v))
    return (0.15 <= t <= 1.05) and (perp < 0.8 * L)


def main():
    frames = json.load(open(BENCH))
    S = OctoSegmenter(str(DEFAULT_CKPT))
    ok_b, ok_a, rows_ui = [], [], []
    for j, f in enumerate(frames):
        img = cv2.imread(str(DS / f["image"]))
        mm, _ = S.segment(img)
        m255 = (mm.astype(np.uint8)) * 255
        nodes, edges = full_graph(m255)
        if nodes is None:
            continue
        center = next((n for n in nodes if n["is_center"]), None)
        headN = next((n for n in nodes if n.get("is_head")), None)
        bases = [(n["x"], n["y"]) for n in nodes if "Base" in n.get("body_part", "")]
        if center is None or headN is None or len(bases) < 2:
            continue
        mantle = (center["x"], center["y"])
        crown = tuple(np.mean(np.asarray(bases, float), axis=0))
        old_ok = plaus((headN["x"], headN["y"]), mantle, crown)
        new = anatomical_head(m255, mantle, bases)
        new_ok = plaus(new[:2], mantle, crown) if new else False
        ok_b.append(old_ok); ok_a.append(new_ok)
        # overlay: mask dim + mantle (red), crown (white), old head (orange), new head (green)
        vis = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
        mmask = m255 > 0
        vis[mmask] = (0.75 * vis[mmask] + 0.25 * np.array([60, 150, 60])).astype(np.uint8)
        cv2.circle(vis, (int(mantle[0]), int(mantle[1])), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(crown[0]), int(crown[1])), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(vis, (int(mantle[0]), int(mantle[1])), (int(crown[0]), int(crown[1])),
                 (160, 160, 160), 1, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(headN["x"]), int(headN["y"])), (0, 165, 255),
                       cv2.MARKER_TRIANGLE_UP, 18, 3)
        if new:
            cv2.drawMarker(vis, (int(new[0]), int(new[1])), (80, 255, 120),
                           cv2.MARKER_TRIANGLE_UP, 18, 3)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, f"head: OLD orange ({'ok' if old_ok else 'BAD'}) vs NEW green "
                         f"({'ok' if new_ok else 'BAD'}) | mantle red, crown white",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{j:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 87])
        rows_ui.append({"file": f"{j:03d}.jpg", "left_arms": int(old_ok), "right_arms": int(new_ok)})
        print(f"  [{j+1}/{len(frames)}] old {'ok' if old_ok else 'BAD'} -> new "
              f"{'ok' if new_ok else 'BAD'}", flush=True)

    rb, ra = float(np.mean(ok_b)), float(np.mean(ok_a))
    json.dump({"meta": {"title": "Head placement — OLD (2nd DT peak) vs NEW (anatomical neck)",
                        "left": "old plausible", "right": "new plausible"},
               "rows": rows_ui}, open(OUT / "summary.json", "w"), indent=1)
    print(f"\nhead plausible: OLD {rb*100:.0f}%  ->  NEW {ra*100:.0f}%  (n={len(ok_b)})")
    json.dump({"old_rate": rb, "new_rate": ra, "n": len(ok_b)},
              open(OUT / "head_fix_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
