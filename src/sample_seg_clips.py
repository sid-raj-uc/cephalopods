"""sample_seg_clips.py — Phase 1a: pick a balanced clip subset for segmentation auto-labeling.

The segmentation auto-labeler (`auto_segment.py`) is slow on CPU, so we run it on a
Colab GPU over a *sampled* subset rather than all ~4k clips. This script chooses that
subset: it joins the on-disk clips to their behaviour labels in the clip index, drops
`octopus not present`, and water-fills a target count as EVENLY as possible across
behaviours (over-sampling rare classes like Swimming/Colour-change up to what exists) and,
within each behaviour, round-robins across cameras.

Colour cameras first (Right_Front/Right_Back/Right_Right) per the plan's colour-first v1;
add Right_Top with `--cameras ... Right_Top`. Right_Left (reflections) is never included.

Outputs `--out/sample_manifest.json` (the selection + per-behaviour/-camera counts + a size
estimate). With `--stage` it also COPIES the selected clips into `--out/clips/<camera>/` so
the folder can be zipped and uploaded to Colab; auto_segment.py then runs with
`--clips-root <out>/clips`.

Stdlib only — runs on a bare box (no torch/GPU needed).

CLI:
  python3 sample_seg_clips.py --target 800                 # write manifest only
  python3 sample_seg_clips.py --target 800 --stage         # + copy clips ready to zip
"""
import argparse, glob, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CAMERAS = ("Right_Front", "Right_Back", "Right_Right", "Right_Left", "Right_Top")
COLOUR_CAMERAS = ["Right_Front", "Right_Back", "Right_Right"]      # colour-first v1
NOT_PRESENT = "octopus not present"
UNLABELED = "(unlabeled)"


def camera_of(path):
    name = Path(path).name
    for c in CAMERAS:
        if c in name:
            return c
    return "unknown"


def label_of(entry):
    """Prefer the 235B label (better presence filter), fall back to the 30B label."""
    lab = entry.get("ethogram_label_235b") or entry.get("ethogram_label")
    if lab and lab.strip():
        return lab.strip()
    return UNLABELED


def water_fill(capacities, target):
    """Allocate `target` across buckets as evenly as possible, capped by each capacity.

    Returns {bucket: n}. Rare buckets contribute everything they have; the surplus
    redistributes evenly to the buckets that still have room.
    """
    alloc = {b: 0 for b in capacities}
    remaining = min(target, sum(capacities.values()))
    open_buckets = [b for b, c in capacities.items() if c > 0]
    while remaining > 0 and open_buckets:
        share = max(1, remaining // len(open_buckets))
        progressed = False
        for b in list(open_buckets):
            if remaining <= 0:
                break
            room = capacities[b] - alloc[b]
            take = min(share, room, remaining)
            if take > 0:
                alloc[b] += take
                remaining -= take
                progressed = True
            if alloc[b] >= capacities[b]:
                open_buckets.remove(b)
        if not progressed:
            break
    return alloc


def round_robin_by_camera(clips, n, rng):
    """Pick n clips from `clips` spreading picks across cameras, deterministically."""
    by_cam = defaultdict(list)
    for c in clips:
        by_cam[c["camera"]].append(c)
    for cam in by_cam:
        rng.shuffle(by_cam[cam])
    order = sorted(by_cam)            # stable camera order
    picked, i = [], 0
    while len(picked) < n and any(by_cam[cam] for cam in order):
        cam = order[i % len(order)]
        if by_cam[cam]:
            picked.append(by_cam[cam].pop())
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", default=str(REPO / "octopus_clips_verified"),
                    help="dir holding the extracted clips (searched recursively for *.mp4)")
    ap.add_argument("--index", default=str(HERE / "octopus_clips_verified.json"),
                    help="clip index (basename -> behaviour label)")
    ap.add_argument("--out", default=str(HERE / "dataset_seg" / "sample_v1"))
    ap.add_argument("--cameras", nargs="+", default=COLOUR_CAMERAS,
                    help="cameras to sample from (default: colour cameras)")
    ap.add_argument("--target", type=int, default=800, help="target subset size")
    ap.add_argument("--no-unlabeled", action="store_true",
                    help="drop clips with no behaviour label (default: keep as one bucket)")
    ap.add_argument("--stage", action="store_true",
                    help="copy selected clips into <out>/clips/<camera>/ for zipping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "Right_Left" in args.cameras:
        print("WARNING: Right_Left is the reflection camera — excluding it.")
        args.cameras = [c for c in args.cameras if c != "Right_Left"]
    rng = random.Random(args.seed)

    # index: basename -> entry
    idx = json.load(open(args.index))
    by_base = {Path(c["clip_path"]).name: c for c in idx["clips"]}

    # on-disk clips, joined to labels, filtered to requested cameras
    disk = glob.glob(f"{args.clips_root}/**/*.mp4", recursive=True)
    pool, unmatched, not_present = [], 0, 0
    for f in disk:
        cam = camera_of(f)
        if cam not in args.cameras:
            continue
        entry = by_base.get(Path(f).name)
        if entry is None:
            unmatched += 1
            continue
        lab = label_of(entry)
        if lab == NOT_PRESENT:
            not_present += 1
            continue
        if lab == UNLABELED and args.no_unlabeled:
            continue
        pool.append({"path": f, "camera": cam, "label": lab})

    print(f"on-disk mp4s: {len(disk)}  |  in requested cameras & present: {len(pool)}")
    print(f"  (skipped {unmatched} not-in-index, {not_present} octopus-not-present)")

    # bucket by behaviour, water-fill the target across behaviours, round-robin cameras within
    by_label = defaultdict(list)
    for c in pool:
        by_label[c["label"]].append(c)
    capacities = {lab: len(cs) for lab, cs in by_label.items()}
    alloc = water_fill(capacities, args.target)

    selected = []
    for lab in sorted(by_label):
        picks = round_robin_by_camera(by_label[lab], alloc[lab], rng)
        selected.extend(picks)
    rng.shuffle(selected)

    # size estimate
    total_bytes = sum(Path(c["path"]).stat().st_size for c in selected)

    lab_counts = Counter(c["label"] for c in selected)
    cam_counts = Counter(c["camera"] for c in selected)
    print(f"\nselected {len(selected)} clips  (~{total_bytes/1e9:.2f} GB)")
    print("  by behaviour:")
    for lab, n in lab_counts.most_common():
        print(f"    {n:4d}  {lab}   (of {capacities[lab]} available)")
    print("  by camera:")
    for cam, n in cam_counts.most_common():
        print(f"    {n:4d}  {cam}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Balanced clip subset for segmentation auto-labeling (Phase 1a).",
        "config": {"cameras": args.cameras, "target": args.target, "seed": args.seed,
                   "include_unlabeled": not args.no_unlabeled,
                   "clips_root": args.clips_root, "index": args.index},
        "count": len(selected),
        "size_gb": round(total_bytes / 1e9, 3),
        "by_behaviour": dict(lab_counts),
        "by_camera": dict(cam_counts),
        "clips": [{"path": c["path"], "camera": c["camera"], "label": c["label"]}
                  for c in sorted(selected, key=lambda c: c["path"])],
    }
    mpath = out / "sample_manifest.json"
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"\nmanifest -> {mpath}")

    if args.stage:
        clips_dir = out / "clips"
        copied = 0
        for c in selected:
            dst = clips_dir / c["camera"] / Path(c["path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(c["path"], dst)
                copied += 1
        print(f"staged {len(selected)} clips ({copied} newly copied) -> {clips_dir}")
        print(f"\nnext: zip and upload to Colab, then run auto_segment.py --clips-root <unzipped>/clips")
        print(f"  (cd {out} && zip -r sample_v1_clips.zip clips)")
    else:
        print("\n(no --stage: manifest only. Re-run with --stage to copy the clips for upload.)")


if __name__ == "__main__":
    main()
