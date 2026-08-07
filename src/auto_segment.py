"""auto_segment.py — Phase 0/1 auto-labeler: octopus clips -> (frame, mask) training pairs.

Recipe (validated 2026-07-21, before/after on 4 cameras):
  1. GroundingDINO ("an octopus.") per sampled frame -> pick the HIGHEST-confidence frame as seed.
  2. GATE: if the best confidence < min_seed_conf, reject the whole clip (this is what filters out
     the reflection cameras — a human reflected in the glass grounds at ~0.50, real octopus 0.7-0.9).
  3. SAM2 video propagation from the seed frame (both directions) -> temporally-consistent masks.
  4. Cleanup: keep the largest connected component (drops detached tool/pipe/reflection fragments);
     area-continuity check drops frames whose area jumps >3x the clip median (transient errors).
  5. Emit N_PER_CLIP evenly-spaced clean frames as (image.jpg, mask.png) pairs + a manifest row.

Phase-0 fix (2026-07-26, `build_prompts`): seed SAM2 with box + POSITIVE points inside the box AND
NEGATIVE points on the brightest regions outside it (metal tools/pipes on IR, specular reflections)
plus the frame corners (background). The box alone gave SAM2 no "what is NOT octopus" cue, so it bled
onto bright tools (the #1 IR failure) and loose background (the colour failure). Toggle with
`--no-points` for an A/B; `--min-seed-conf` lowers the gate to accept resting/camouflaged frames
(route those through human-verify); `--debug-dir/--debug-n` dump seed overlays (box+prompts+mask).

Device auto-selects cuda -> (mps) -> cpu, so the SAME script runs locally (slow, CPU) or on a
Colab GPU (fast). Resumable: skips clips already in the manifest. Right_Left is excluded by default.

CLI:
  python3 auto_segment.py --clips-root <dir> --out src/dataset_seg/v1 [--limit N] [--cameras ...]
"""
import argparse, glob, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ── params ───────────────────────────────────────────────────────────────────────
PROMPT        = "an octopus."
BOX_THR, TXT_THR = 0.35, 0.25
MIN_SEED_CONF = 0.60      # reject a clip whose best detection is below this (kills reflections)
FPS           = 3         # propagation sampling rate
MAXSIDE       = 1024      # frame downscale for GroundingDINO/SAM2
N_PER_CLIP    = 4         # clean frames emitted per accepted clip
AREA_MIN, AREA_MAX = 0.003, 0.60   # sane octopus-mask area as a fraction of frame
DEFAULT_CAMERAS = ["Right_Front", "Right_Back", "Right_Right", "Right_Top"]  # NOT Right_Left


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_models(device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny").to(device).eval()
    # GroundingDINO's deformable-attention is unstable on MPS -> keep it on CPU there.
    gd_dev = "cpu" if device == "mps" else device
    if gd_dev != device: gd = gd.to(gd_dev)
    sam2 = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny", device=device)
    return {"gd_proc": gd_proc, "gd": gd, "gd_dev": gd_dev, "sam2": sam2}


def gd_best_box(img, M):
    inp = M["gd_proc"](images=img, text=PROMPT, return_tensors="pt").to(M["gd_dev"])
    with torch.no_grad():
        out = M["gd"](**inp)
    r = M["gd_proc"].post_process_grounded_object_detection(
        out, inp.input_ids, threshold=BOX_THR, text_threshold=TXT_THR,
        target_sizes=[img.size[::-1]])[0]
    if len(r["scores"]) == 0:
        return None, 0.0
    i = int(torch.argmax(r["scores"]))
    return r["boxes"][i].tolist(), float(r["scores"][i])


def largest_blob(mask):
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def build_prompts(img, box, n_neg_bright=4, bright_pct=99.0):
    """Phase-0 fix: return (points [K,2] float32, labels [K] int) to steer SAM2.

    POSITIVE points anchor the octopus body inside the GroundingDINO box; NEGATIVE points sit on the
    brightest regions OUTSIDE the box (metal tools / pipes on IR, specular reflections) and the frame
    corners (background). This stops SAM2 bleeding the mask onto bright tools (the #1 IR failure) and
    into the loose background (the colour failure) — the box alone gives it no "what is NOT octopus" cue.
    """
    W, H = img.size
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    pts, labs = [[(x0 + x1) / 2, (y0 + y1) / 2]], [1]           # box centre = strong positive
    for fx in (0.35, 0.65):                                      # interior grid keeps the whole body
        for fy in (0.35, 0.65):
            pts.append([x0 + fx * bw, y0 + fy * bh]); labs.append(1)
    g = np.asarray(img.convert("L"), np.float32)
    out = np.ones(g.shape, bool)
    out[int(max(0, y0)):int(y1), int(max(0, x0)):int(x1)] = False   # exclude the box region
    if out.any():
        thr = np.percentile(g[out], bright_pct)
        ys, xs = np.where((g >= thr) & out)
        if len(xs):
            idx = np.linspace(0, len(xs) - 1, min(n_neg_bright, len(xs))).astype(int)
            for k in idx:
                pts.append([float(xs[k]), float(ys[k])]); labs.append(0)   # bright tool/reflection
    for px, py in [(4, 4), (W - 4, 4), (4, H - 4), (W - 4, H - 4)]:         # background corners
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            pts.append([float(px), float(py)]); labs.append(0)
    return np.array(pts, np.float32), np.array(labs, np.int32)


def save_debug_overlay(img, box, pts, labs, mask, path):
    """Dump the seed frame with box (yellow), +points (green), -points (red), mask (green) for eyeballing."""
    import cv2
    im = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR).copy()
    if mask is not None and mask.any():
        im[mask] = (0.45 * im[mask] + 0.55 * np.array([0, 200, 0])).astype(np.uint8)
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 220, 220), 2)
    for (px, py), l in zip(pts, labs):
        cv2.circle(im, (int(px), int(py)), 5, (0, 220, 0) if l == 1 else (0, 0, 230), -1)
        cv2.circle(im, (int(px), int(py)), 5, (255, 255, 255), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), im)


def segment_clip(clip, M, cfg=None, debug_path=None):
    """Return list of (PIL frame, bool mask) for a clip, or [] if rejected/failed.

    cfg overrides: min_seed_conf, fps, n_per_clip, use_points (Phase-0 point/negative prompts).
    debug_path: if set, write a seed-frame overlay (box + prompts + mask) there for A/B eyeballing.
    """
    cfg = cfg or {}
    min_conf = cfg.get("min_seed_conf", MIN_SEED_CONF)
    fps = cfg.get("fps", FPS)
    n_per = cfg.get("n_per_clip", N_PER_CLIP)
    use_points = cfg.get("use_points", True)
    with tempfile.TemporaryDirectory() as td:
        fdir = f"{td}/frames"; os.makedirs(fdir)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
                        f"fps={fps},scale='min({MAXSIDE},iw)':-2", f"{fdir}/%05d.jpg"], check=False)
        files = sorted(glob.glob(f"{fdir}/*.jpg"))
        if not files:
            return [], {"reason": "no_frames"}
        imgs = [Image.open(f).convert("RGB") for f in files]
        # seed = most confident frame
        boxes = [gd_best_box(im, M) for im in imgs]
        scores = [s for _, s in boxes]
        seed = int(np.argmax(scores))
        if boxes[seed][0] is None or scores[seed] < min_conf:
            return [], {"reason": "low_conf", "best_conf": round(max(scores), 3)}
        sam2 = M["sam2"]
        st = sam2.init_state(video_path=fdir)
        # Phase-0 fix: box + positive/negative points (falls back to box-only on any failure)
        pts = labs = None
        if use_points:
            try:
                pts, labs = build_prompts(imgs[seed], boxes[seed][0])
            except Exception:
                pts = labs = None
        sam2.add_new_points_or_box(st, frame_idx=seed, obj_id=1,
                                   box=np.array(boxes[seed][0], np.float32),
                                   points=pts, labels=labs)
        masks = [None] * len(imgs)
        for oi, _, logits in sam2.propagate_in_video(st, start_frame_idx=seed):
            masks[oi] = (logits[0] > 0).cpu().numpy()[0]
        for oi, _, logits in sam2.propagate_in_video(st, start_frame_idx=seed, reverse=True):
            masks[oi] = (logits[0] > 0).cpu().numpy()[0]
        masks = [largest_blob(m) if (m is not None and m.any()) else None for m in masks]
        areas = np.array([m.mean() if m is not None else 0.0 for m in masks])
        med = np.median(areas[areas > 0]) if (areas > 0).any() else 0.0
        if debug_path is not None:
            dp, dl = (pts, labs) if pts is not None else (np.array([]).reshape(0, 2), np.array([]))
            save_debug_overlay(imgs[seed], boxes[seed][0], dp, dl, masks[seed], debug_path)
        # keep clean frames: area in range, not a >3x jump
        good = [k for k in range(len(imgs))
                if masks[k] is not None and AREA_MIN <= areas[k] <= AREA_MAX
                and (med == 0 or areas[k] <= 3 * med)]
        if not good:
            return [], {"reason": "no_clean_frames", "best_conf": round(scores[seed], 3)}
        pick = [good[i] for i in np.linspace(0, len(good) - 1, min(n_per, len(good))).astype(int)]
        return [(imgs[k], masks[k]) for k in sorted(set(pick))], \
               {"reason": "ok", "best_conf": round(scores[seed], 3), "seed": seed,
                "area_med": round(float(med), 4), "n_frames": len(imgs), "n_good": len(good)}


def camera_of(path):
    for c in ("Right_Front", "Right_Back", "Right_Right", "Right_Left", "Right_Top"):
        if c in Path(path).name:
            return c
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", default=str(REPO / "src" / "octopus_clips_verified"))
    ap.add_argument("--out", default=str(REPO / "src" / "dataset_seg" / "v1"))
    ap.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-seed-conf", type=float, default=MIN_SEED_CONF,
                    help="reject clip if best GroundingDINO conf below this; lower to accept "
                         "resting/camouflaged frames (route those through human-verify)")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--n-per-clip", type=int, default=N_PER_CLIP)
    ap.add_argument("--no-points", action="store_true",
                    help="disable the Phase-0 point/negative prompts (box-only, the old recipe) — for A/B")
    ap.add_argument("--debug-dir", default=None, help="write seed-frame prompt/mask overlays here")
    ap.add_argument("--debug-n", type=int, default=0, help="how many clips to emit debug overlays for")
    args = ap.parse_args()
    cfg = {"min_seed_conf": args.min_seed_conf, "fps": args.fps,
           "n_per_clip": args.n_per_clip, "use_points": not args.no_points}

    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in open(manifest):
            try: done.add(json.loads(line)["clip"])
            except Exception: pass

    clips = sorted(p for p in glob.glob(f"{args.clips_root}/**/*.mp4", recursive=True)
                   if camera_of(p) in args.cameras)
    clips = [c for c in clips if c not in done]
    if args.limit: clips = clips[:args.limit]
    print(f"device={pick_device()}  clips to do={len(clips)}  (already done={len(done)})", flush=True)

    M = load_models(pick_device())
    print("models loaded.", flush=True)
    stats = {"ok": 0, "low_conf": 0, "no_clean_frames": 0, "no_frames": 0, "pairs": 0}
    with open(manifest, "a") as mf:
        for i, clip in enumerate(clips):
            cam = camera_of(clip)
            dbg = None
            if args.debug_dir and i < args.debug_n:
                tag = "pts" if not args.no_points else "box"
                dbg = str(Path(args.debug_dir) / f"seed_{i:03d}_{cam}_{tag}.png")
            try:
                pairs, info = segment_clip(clip, M, cfg=cfg, debug_path=dbg)
            except Exception as e:
                info = {"reason": f"error:{type(e).__name__}"}; pairs = []
            stats[info["reason"]] = stats.get(info["reason"], 0) + 1
            for j, (img, mask) in enumerate(pairs):
                # key the filename on date/segment (NOT the run-local index i) so it is stable across
                # resumes — the harvest has many same-named clips (e.g. Right_Back_0-20.mp4) across
                # different dates, and i resets on resume (line 229 filters done clips first), so an
                # i-based stem would collide/overwrite across runs.
                vid = f"{Path(clip).parent.parent.name}_{Path(clip).parent.name}"
                stem = f"{vid}_{Path(clip).stem}_{cam}_{j}"
                img.save(out / "images" / f"{stem}.jpg", quality=90)
                Image.fromarray((mask * 255).astype(np.uint8)).save(out / "masks" / f"{stem}.png")
                mf.write(json.dumps({"clip": clip, "camera": cam, "image": f"images/{stem}.jpg",
                                     "mask": f"masks/{stem}.png", "area": round(float(mask.mean()), 4),
                                     "best_conf": info.get("best_conf")}) + "\n")
                stats["pairs"] += 1
            mf.flush()
            if (i + 1) % 20 == 0 or i == len(clips) - 1:
                print(f"[{i+1}/{len(clips)}] {cam} {info['reason']}  "
                      f"pairs={stats['pairs']} ok={stats['ok']} low_conf={stats['low_conf']}", flush=True)
    print(f"\nDONE. {stats}\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
