"""seg_skeleton_pipeline.py — one video -> three synchronized overlay videos:
  1) RAW segmentation mask (per-frame threshold)
  2) SMOOTHED segmentation mask (temporal EMA + largest-blob + morph)
  3) SKELETON (anatomical graph tracked over time on the smoothed masks)

All three are cropped to the SAME fixed union bbox so they align and the octopus is large.
Used by ui/seg_skeleton_viewer.py.
"""
import subprocess, sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter, _largest_blob
from skeleton import branch_color
from multi_frame import tracked_sequence
import math

DEFAULT_CKPT = HERE.parent / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"
EMA_ALPHA = 0.45
MASK_BGR = np.array([120, 235, 0], np.float32)
ALPHA = 0.5
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def _overlay(frame, mask, outline=True):
    out = frame.astype(np.float32)
    if mask is not None and mask.any():
        out[mask] = (1 - ALPHA) * out[mask] + ALPHA * MASK_BGR
        if outline:
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 0, 200), 2)
    return out.astype(np.uint8)


def _draw_skeleton(canvas, nodes, edges, thick=2):
    for e in edges:
        p = np.rint(np.asarray(e["polyline"])).astype(np.int32).reshape(-1, 1, 2)
        if len(p) < 2:
            continue
        c = (0, 230, 120) if e.get("body_part") == "Head" else tuple(int(v) for v in branch_color(e["branch_id"])[::-1])
        cv2.polylines(canvas, [p], False, c, thick, cv2.LINE_AA)
    for n in nodes:
        pt = (int(round(n["x"])), int(round(n["y"])))
        if n["is_center"]:
            col, r = (0, 0, 255), 7
        elif n.get("is_head"):
            col, r = (120, 230, 0), 7
        elif n["is_tip"]:
            col, r = (0, 215, 255), 5
        else:
            col, r = (255, 255, 255), 4
        cv2.circle(canvas, pt, r, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, r, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def _draw_trails(canvas, trails, maxlen=12):
    """Recent tip positions per arm id, fading, ID-consistent colours — identity jumps become
    visible as a colour's trail teleporting."""
    for a, pts in trails.items():
        col = tuple(int(v) for v in branch_color(a)[::-1])
        seg = pts[-maxlen:]
        for i in range(1, len(seg)):
            f = 0.3 + 0.7 * i / len(seg)
            c = tuple(int(v * f) for v in col)
            cv2.line(canvas, (int(seg[i - 1][0]), int(seg[i - 1][1])),
                     (int(seg[i][0]), int(seg[i][1])), c, 2, cv2.LINE_AA)
    return canvas


def _label(img, text, color=(0, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return img


def _write_mp4(frames, path, fps):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    w += w % 2; h += h % 2
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for fr in frames:
        canvas = np.zeros((h, w, 3), np.uint8); canvas[:fr.shape[0], :fr.shape[1]] = fr
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close(); ff.wait()
    return path.exists() and path.stat().st_size > 1000


def process_video_3way(video, out_dir, S=None, fps=5.0, work_w=960, present=0.004,
                       min_arms=3, max_arms=8, on_stage=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if S is None:
        S = OctoSegmenter(str(DEFAULT_CKPT))

    def stage(s):
        if on_stage:
            on_stage(s)
        print(s, flush=True)

    # ---- pass 1: decode + segment (raw + EMA-smoothed), at working resolution ----
    stage("segmenting frames")
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps)))
    eff_fps = src_fps / step
    frames, raws, smooths = [], [], []
    i, ema = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            H0, W0 = frame.shape[:2]
            sw = work_w; sh = int(round(H0 * work_w / W0))
            fr = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            prob = S.prob(frame)
            ema = prob if ema is None else EMA_ALPHA * prob + (1 - EMA_ALPHA) * ema
            pr = cv2.resize(prob, (sw, sh), interpolation=cv2.INTER_LINEAR)
            pe = cv2.resize(ema, (sw, sh), interpolation=cv2.INTER_LINEAR)
            rm = pr > 0.5
            if rm.any():
                rm = _largest_blob(rm)
            sm = pe > 0.5
            if sm.any():
                sm = _largest_blob(sm)
                sm = cv2.morphologyEx(sm.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL).astype(bool)
            frames.append(fr); raws.append(rm); smooths.append(sm)
        i += 1
    cap.release()
    if not frames:
        return {"error": "no frames decoded"}

    # present set (octopus visible per smoothed mask); union bbox over those
    present_idx = [k for k, m in enumerate(smooths) if m.mean() >= present]
    if not present_idx:
        return {"error": "no octopus-present frames"}
    ys, xs = [], []
    for k in present_idx:
        yy, xx = np.where(smooths[k])
        ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
    H, W = frames[0].shape[:2]
    ph = int((max(ys) - min(ys)) * 0.12) + 6; pw = int((max(xs) - min(xs)) * 0.12) + 6
    y0, y1 = max(0, min(ys) - ph), min(H, max(ys) + ph)
    x0, x1 = max(0, min(xs) - pw), min(W, max(xs) + pw)

    def crop(a):
        return a[y0:y1, x0:x1]

    # ---- pass 2a: raw + smoothed overlay panels (present frames only, aligned) ----
    stage("rendering raw + smoothed overlays")
    raw_frames, smooth_frames = [], []
    for k in present_idx:
        fc = crop(frames[k])
        raw_frames.append(_label(_overlay(fc, crop(raws[k])), "1) RAW segmentation"))
        smooth_frames.append(_label(_overlay(fc, crop(smooths[k])), "2) SMOOTHED segmentation", (120, 255, 120)))

    # ---- pass 2b: skeleton tracked over the smoothed-mask crop sequence (best-frame seeded) ----
    stage("extracting skeleton")
    crops = [(crop(smooths[k]).astype(np.uint8)) * 255 for k in present_idx]
    greys = [cv2.cvtColor(crop(frames[k]), cv2.COLOR_BGR2GRAY) for k in present_idx]
    graphs = tracked_sequence(crops, min_arms, max_arms, 2, 1024, seed="best", greys=greys)
    n_tracked = len(graphs)
    skel_frames = []
    trails = {}
    for pos, k in enumerate(present_idx):
        fc = crop(frames[k]).copy()
        if pos in graphs:
            nodes, edges = graphs[pos]
            for n in nodes:
                if n.get("is_tip"):
                    trails.setdefault(n["branch_id"], []).append((n["x"], n["y"]))
            arms = len({n["branch_id"] for n in nodes if n["branch_id"] > 0})
            _draw_trails(fc, trails)
            skel_frames.append(_label(_draw_skeleton(fc, nodes, edges), f"3) SKELETON - {arms} arms", (0, 215, 255)))
        else:
            base = _overlay(fc, crop(smooths[k]), outline=True)
            skel_frames.append(_label(base, "3) SKELETON - tracking", (0, 165, 255)))

    stage("encoding videos")
    raw_p, sm_p, sk_p = out_dir / "raw.mp4", out_dir / "smooth.mp4", out_dir / "skeleton.mp4"
    _write_mp4(raw_frames, raw_p, eff_fps)
    _write_mp4(smooth_frames, sm_p, eff_fps)
    _write_mp4(skel_frames, sk_p, eff_fps)
    return {"raw": str(raw_p), "smooth": str(sm_p), "skeleton": str(sk_p),
            "n_present": len(present_idx), "n_tracked": n_tracked,
            "fps": round(eff_fps, 2), "crop": [int(x0), int(y0), int(x1), int(y1)]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--fps", type=float, default=5.0)
    a = ap.parse_args()
    print(process_video_3way(a.video, a.out, fps=a.fps))
