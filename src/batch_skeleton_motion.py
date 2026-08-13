"""batch_skeleton_motion.py — run segmentation -> skeleton -> (smoothed) motion over many clips and
summarize per-clip octopus KINEMATICS, then fold them into the behavioural records.

For each clip: octo_seg mask per frame (EMA-smoothed) -> fixed-bbox crop -> multi-frame skeleton with
temporal arm tracking -> smoothed per-node speed. Per clip we emit a compact kinematic summary
(arm-tip speed, mantle speed, per-frame activity, arm spread, arm count). Writes
data/skeleton_motion.json (keyed by the same relative clip path behaviour_records.json uses) and,
with --merge, adds a `kinematics` block onto each matching behaviour_records.json entry.

Usage:
  venv/bin/python3 src/batch_skeleton_motion.py --limit 20 [--date 2026-02-20] [--merge]
"""
import argparse, collections, json, math, sys, time
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter
from segment_to_skeleton import segment_masks, union_bbox, DEFAULT_CKPT
from multi_frame import tracked_sequence, compute_motion

CLIPS_ROOT = REPO / "src" / "octopus_clips_verified"
REL_PREFIX = "octopus_clips_verified"       # how behaviour_records.json keys clips


def _stat(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return None
    return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2), "max": round(float(a.max()), 2)}


def clip_to_motion(clip, S, fps=3.0, present=0.004, min_arms=3, max_arms=8,
                   iterations=2, max_dim=1024):
    """segment -> masks -> temporal skeleton -> smoothed motion summary (no figures). None if unusable."""
    masks, src_fps, step = segment_masks(clip, S, fps, present)
    pm = [(k, m) for k, m in enumerate(masks) if m is not None]
    if len(pm) < 4:
        return None
    y0, y1, x0, x1 = union_bbox([m for _, m in pm])
    eff_fps = src_fps / step
    crops = [(m[y0:y1, x0:x1].astype(np.uint8)) * 255 for _, m in pm]
    graphs = tracked_sequence(crops, min_arms, max_arms, iterations, max_dim, seed="best")
    processed = []
    for ci in sorted(graphs):                                 # emit in temporal order for motion
        nodes, edges = graphs[ci]
        arms = len({n["branch_id"] for n in nodes if n["branch_id"] > 0})
        processed.append({"name": f"{pm[ci][0]:05d}", "index": pm[ci][0], "nodes": nodes,
                          "edges": edges, "metrics": {"arm_count": arms}})
    if len(processed) < 3:
        return None

    rows = compute_motion(processed, 1, eff_fps, smooth=True)
    speeds = collections.defaultdict(list)
    for r in rows:
        try:
            speeds[r["node"]].append(float(r["speed_px_per_s"]))
        except Exception:
            pass
    tip_speeds = [s for k, v in speeds.items() if k.endswith("_Tip") for s in v]
    byframe = collections.defaultdict(list)
    for r in rows:
        if r["node"].endswith("_Tip"):
            try:
                byframe[r["frame_index"]].append(float(r["speed_px_per_s"]))
            except Exception:
                pass
    frame_activity = [float(np.mean(v)) for v in byframe.values()]
    spreads = []
    for fr in processed:
        c = next((n for n in fr["nodes"] if n["is_center"]), None)
        tips = [n for n in fr["nodes"] if n.get("is_tip")]
        if c and tips:
            spreads.append(float(np.mean([math.hypot(n["x"] - c["x"], n["y"] - c["y"]) for n in tips])))
    arm_counts = [fr["metrics"]["arm_count"] for fr in processed]
    return {
        "n_frames_tracked": len(processed),
        "fps": round(eff_fps, 3),
        "median_arm_count": int(np.median(arm_counts)) if arm_counts else 0,
        "tip_speed_px_s": _stat(tip_speeds),
        "mantle_speed_px_s": _stat(speeds.get("MantleCenter", [])),
        "activity_px_s": _stat(frame_activity),           # per-frame mean arm-tip speed = movement level
        "arm_spread_px": _stat(spreads),                  # mean tip distance from mantle = posture spread
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--date", default="", help="only clips under this date dir (e.g. 2026-02-20)")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(REPO / "data" / "skeleton_motion.json"))
    ap.add_argument("--merge", action="store_true", help="also add `kinematics` into behaviour_records.json")
    args = ap.parse_args()

    # clip list: prefer clips already in behaviour_records (so the merge lands), that exist locally
    br_path = REPO / "data" / "behaviour_records.json"
    keys = list(json.load(open(br_path)).keys()) if br_path.exists() else []
    clips = []
    for rel in keys:
        if args.date and f"/{args.date}/" not in "/" + rel:
            continue
        p = REPO / "src" / rel
        if p.exists():
            clips.append((rel, p))
    if not clips:      # fallback: just glob local clips
        for p in sorted(CLIPS_ROOT.glob("**/*.mp4")):
            rel = str(p.relative_to(REPO / "src"))
            if not args.date or f"/{args.date}/" in "/" + rel:
                clips.append((rel, p))
    clips = clips[:args.limit]
    print(f"processing {len(clips)} clips with {Path(args.ckpt).name}", flush=True)

    S = OctoSegmenter(args.ckpt)
    out = json.load(open(args.out)) if Path(args.out).exists() else {}
    t0 = time.perf_counter(); ok = 0
    for i, (rel, p) in enumerate(clips, 1):
        try:
            summ = clip_to_motion(str(p), S, fps=args.fps)
        except Exception as exc:
            summ = None; print(f"  [{i}/{len(clips)}] {rel}  ERROR {exc}", flush=True)
        if summ is None:
            print(f"  [{i}/{len(clips)}] {rel}  skipped (too few tracked frames)", flush=True)
            continue
        out[rel] = summ; ok += 1
        act = summ["activity_px_s"]["mean"] if summ["activity_px_s"] else 0
        print(f"  [{i}/{len(clips)}] {rel}  {summ['n_frames_tracked']}f "
              f"{summ['median_arm_count']}arms  activity={act}", flush=True)
        json.dump(out, open(args.out, "w"), indent=1)      # checkpoint each clip (resumable)

    print(f"\n{ok}/{len(clips)} clips summarized in {time.perf_counter()-t0:.0f}s -> {args.out}", flush=True)

    if args.merge and br_path.exists():
        br = json.load(open(br_path)); n = 0
        for rel, summ in out.items():
            if rel in br:
                br[rel]["kinematics"] = summ; n += 1
        json.dump(br, open(br_path, "w"), indent=1)
        print(f"merged kinematics into {n} behaviour_records.json entries", flush=True)


if __name__ == "__main__":
    main()
