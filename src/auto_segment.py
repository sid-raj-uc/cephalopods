"""auto_segment.py — Phase 0/1 auto-labeler: octopus clips -> (frame, mask) training pairs.

Recipe (validated 2026-07-21, before/after on 4 cameras):
  1. GroundingDINO ("an octopus.") per sampled frame -> pick the HIGHEST-confidence frame as seed.
  2. GATE: if the best confidence < MIN_SEED_CONF, reject the whole clip (this is what filters out
     the reflection cameras — a human reflected in the glass grounds at ~0.50, real octopus 0.7-0.9).
  3. SAM2 video propagation from the seed frame (both directions) -> temporally-consistent masks.
  4. Cleanup: keep the largest connected component (drops detached tool/pipe/reflection fragments);
     area-continuity check drops frames whose area jumps >3x the clip median (transient errors).
  5. Emit N_PER_CLIP evenly-spaced clean frames as (image.jpg, mask.png) pairs + a manifest row.

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


def segment_clip(clip, M):
    """Return list of (PIL frame, bool mask) for a clip, or [] if rejected/failed."""
    with tempfile.TemporaryDirectory() as td:
        fdir = f"{td}/frames"; os.makedirs(fdir)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
                        f"fps={FPS},scale='min({MAXSIDE},iw)':-2", f"{fdir}/%05d.jpg"], check=False)
        files = sorted(glob.glob(f"{fdir}/*.jpg"))
        if not files:
            return [], {"reason": "no_frames"}
        imgs = [Image.open(f).convert("RGB") for f in files]
        # seed = most confident frame
        boxes = [gd_best_box(im, M) for im in imgs]
        scores = [s for _, s in boxes]
        seed = int(np.argmax(scores))
        if boxes[seed][0] is None or scores[seed] < MIN_SEED_CONF:
            return [], {"reason": "low_conf", "best_conf": round(max(scores), 3)}
        sam2 = M["sam2"]
        st = sam2.init_state(video_path=fdir)
        sam2.add_new_points_or_box(st, frame_idx=seed, obj_id=1,
                                   box=np.array(boxes[seed][0], np.float32))
        masks = [None] * len(imgs)
        for oi, _, logits in sam2.propagate_in_video(st, start_frame_idx=seed):
            masks[oi] = (logits[0] > 0).cpu().numpy()[0]
        for oi, _, logits in sam2.propagate_in_video(st, start_frame_idx=seed, reverse=True):
            masks[oi] = (logits[0] > 0).cpu().numpy()[0]
        masks = [largest_blob(m) if (m is not None and m.any()) else None for m in masks]
        areas = np.array([m.mean() if m is not None else 0.0 for m in masks])
        med = np.median(areas[areas > 0]) if (areas > 0).any() else 0.0
        # keep clean frames: area in range, not a >3x jump
        good = [k for k in range(len(imgs))
                if masks[k] is not None and AREA_MIN <= areas[k] <= AREA_MAX
                and (med == 0 or areas[k] <= 3 * med)]
        if not good:
            return [], {"reason": "no_clean_frames", "best_conf": round(scores[seed], 3)}
        pick = [good[i] for i in np.linspace(0, len(good) - 1, min(N_PER_CLIP, len(good))).astype(int)]
        return [(imgs[k], masks[k]) for k in sorted(set(pick))], \
               {"reason": "ok", "best_conf": round(scores[seed], 3), "seed": seed,
                "n_frames": len(imgs), "n_good": len(good)}


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
    args = ap.parse_args()

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
            try:
                pairs, info = segment_clip(clip, M)
            except Exception as e:
                info = {"reason": f"error:{type(e).__name__}"}; pairs = []
            stats[info["reason"]] = stats.get(info["reason"], 0) + 1
            for j, (img, mask) in enumerate(pairs):
                stem = f"{Path(clip).stem}_{cam}_{i:05d}_{j}"
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
