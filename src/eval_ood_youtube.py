"""eval_ood_youtube.py — out-of-domain probe: how do our models behave on wild YouTube footage?

WHY: the paper's biggest external-validity weakness is scope — one individual (Nity), one tank,
IR + colour aquarium cameras. This runs the two deployed presence signals over Creative-Commons
wild-ocean footage (a domain the models have never seen: daylight colour, open water, other
species, camera motion, no tank glass) to see whether they transfer at all.

WHAT THIS IS *NOT*: a benchmark. There are no ground-truth masks, so no IoU is computable, and
frame-level presence is unlabelled — an "octopus video" does not contain the animal in every
frame. So:
  * on POSITIVE videos, the fire-rate is a LOWER bound on recall (the animal is often off-screen
    or occluded), NOT recall.
  * on NEGATIVE videos, reef footage is only *presumed* octopus-free. Per this project's
    hard-negative lesson (232 mined "negatives", 166 of which actually held the animal), that
    assumption is NOT trusted: the highest-scoring negative frames are written out as images so a
    human can check whether each apparent false positive is really a false positive.
Read the output as a diagnostic and a source of figures, not as a number for the paper.

Signals measured per frame at 1 fps:
  detector  — CLIP ViT-B/32 + MLP probe (clip_mlp_hardneg_v2), letterboxed: p_visible
  segmenter — the deployed tiny segmenter: mask area fraction (the pipeline's presence gate)

Usage
  venv/bin/python3 src/eval_ood_youtube.py
  venv/bin/python3 src/eval_ood_youtube.py --fps 1 --topk 6
"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from harvest_stream import letterbox, load_detector
from segment_octopus import OctoSegmenter

ROOT = REPO / "data" / "youtube_ood"
SEG_CKPT = REPO / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"
DET_CKPT = REPO / "weights" / "clip_mlp_hardneg_v2.pt"
OUT_JSON = REPO / "data" / "ood_youtube_results.json"
FRAME_DIR = REPO / "data" / "youtube_ood" / "_inspect"

# deployed operating points, taken from the pipeline (not tuned here)
VIS_THRESH = 0.60      # extract_octopus_clips.py --vis-thresh
AREA_GATE = 0.01       # segmenter presence gate (SEGMENTATION_LOG / R14)


def sample_frames(video, fps, tmpdir):
    """1 fps JPEGs, long side capped so CLIP/segmenter see a sane resolution."""
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                    f"fps={fps},scale='min(1024,iw)':-2", f"{tmpdir}/%05d.jpg"], check=False)
    return sorted(Path(tmpdir).glob("*.jpg"))


def score_and_dump(video, M, S, fps, topk, dump_dir):
    """Same as score_video but ALSO writes the top-k highest-area frames + overlay for review."""
    rows = []
    with tempfile.TemporaryDirectory() as td:
        files = sample_frames(video, fps, td)
        if not files:
            return rows
        cm, pre, clf, vis, dev = M["cm"], M["pre"], M["clf"], M["vis"], M["dev"]
        cache = {}
        B = 32
        for i in range(0, len(files), B):
            chunk = files[i:i + B]
            ims = [Image.open(f).convert("RGB") for f in chunk]
            batch = torch.stack([pre(letterbox(im)) for im in ims]).to(dev)
            with torch.no_grad():
                ft = cm.encode_image(batch).float(); ft = ft / ft.norm(dim=-1, keepdim=True)
                p = torch.softmax(clf(ft), dim=1)[:, vis].cpu().numpy()
            for j, fp in enumerate(chunk):
                bgr = cv2.imread(str(fp))
                mask, area = S.segment(bgr)
                idx = i + j
                rows.append({"t": idx, "p_visible": round(float(p[j]), 4),
                             "mask_area": round(float(area), 5)})
                cache[idx] = (bgr, mask)
        # dump the frames most likely to be informative: highest mask area and highest p_visible
        dump_dir.mkdir(parents=True, exist_ok=True)
        by_area = sorted(rows, key=lambda r: -r["mask_area"])[:topk]
        by_p = sorted(rows, key=lambda r: -r["p_visible"])[:topk]
        for tag, sel in (("area", by_area), ("pvis", by_p)):
            for r in sel:
                bgr, mask = cache[r["t"]]
                ov = bgr.copy()
                if mask is not None and mask.any():
                    m = mask
                    if m.shape != bgr.shape[:2]:
                        m = cv2.resize(m.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                                       interpolation=cv2.INTER_NEAREST) > 0
                    ov[m] = (0.45 * ov[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
                cv2.putText(ov, f"t={r['t']}s p={r['p_visible']:.2f} area={r['mask_area']:.4f}",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imwrite(str(dump_dir / f"{tag}_{r['mask_area']:.4f}_t{r['t']:04d}.jpg"), ov)
    return rows


def auc(pos, neg):
    """Rank-AUC. Frame-level and therefore only a video-level proxy here — see the docstring."""
    if not pos or not neg:
        return None
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    M = load_detector(str(DET_CKPT))
    S = OctoSegmenter(str(SEG_CKPT))
    print(f"detector dev={M['dev']}  segmenter={SEG_CKPT.name}\n")

    res = {"_meta": {"fps": args.fps, "detector": DET_CKPT.name, "segmenter": SEG_CKPT.name,
                     "vis_thresh": VIS_THRESH, "area_gate": AREA_GATE,
                     "note": "NOT a benchmark: no GT masks (no IoU), frame presence unlabelled. "
                             "Positive fire-rate is a LOWER bound on recall; negatives are only "
                             "PRESUMED octopus-free and must be verified from _inspect/."},
           "videos": {}}
    t0 = time.time()
    for split in ("pos", "neg"):
        for v in sorted((ROOT / split).glob("*.mp4")):
            rows = score_and_dump(v, M, S, args.fps, args.topk, FRAME_DIR / split / v.stem)
            if not rows:
                print(f"  {v.name}: NO FRAMES"); continue
            p = np.array([r["p_visible"] for r in rows])
            a = np.array([r["mask_area"] for r in rows])
            d = {"split": split, "n_frames": len(rows),
                 "p_visible": {"median": round(float(np.median(p)), 4),
                               "p90": round(float(np.percentile(p, 90)), 4),
                               "frac_over_thresh": round(float((p >= VIS_THRESH).mean()), 4)},
                 "mask_area": {"median": round(float(np.median(a)), 5),
                               "p90": round(float(np.percentile(a, 90)), 5),
                               "frac_over_gate": round(float((a >= AREA_GATE).mean()), 4)},
                 "per_frame": rows}
            res["videos"][v.stem] = d
            print(f"  [{split}] {v.stem:<14} n={len(rows):<4} "
                  f"p_vis med {d['p_visible']['median']:.3f} fire {d['p_visible']['frac_over_thresh']:.2f} | "
                  f"area med {d['mask_area']['median']:.4f} fire {d['mask_area']['frac_over_gate']:.2f}")

    pos_p = [r["p_visible"] for k, v in res["videos"].items() if v["split"] == "pos" for r in v["per_frame"]]
    neg_p = [r["p_visible"] for k, v in res["videos"].items() if v["split"] == "neg" for r in v["per_frame"]]
    pos_a = [r["mask_area"] for k, v in res["videos"].items() if v["split"] == "pos" for r in v["per_frame"]]
    neg_a = [r["mask_area"] for k, v in res["videos"].items() if v["split"] == "neg" for r in v["per_frame"]]
    res["aggregate"] = {
        "n_pos_frames": len(pos_p), "n_neg_frames": len(neg_p),
        "n_pos_videos": sum(v["split"] == "pos" for v in res["videos"].values()),
        "n_neg_videos": sum(v["split"] == "neg" for v in res["videos"].values()),
        "auc_p_visible": None if auc(pos_p, neg_p) is None else round(auc(pos_p, neg_p), 4),
        "auc_mask_area": None if auc(pos_a, neg_a) is None else round(auc(pos_a, neg_a), 4),
        "neg_fp_rate_detector_at_0.60": round(float(np.mean(np.array(neg_p) >= VIS_THRESH)), 4),
        "neg_fp_rate_segmenter_at_0.01": round(float(np.mean(np.array(neg_a) >= AREA_GATE)), 4),
        "pos_fire_rate_detector": round(float(np.mean(np.array(pos_p) >= VIS_THRESH)), 4),
        "pos_fire_rate_segmenter": round(float(np.mean(np.array(pos_a) >= AREA_GATE)), 4),
    }
    print("\n=== aggregate (frame-level, video-level proxy — few clusters, no frame labels) ===")
    for k, v in res["aggregate"].items():
        print(f"  {k:<34} {v}")
    print(f"\nelapsed {time.time()-t0:.0f}s")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out}\ninspect frames: {FRAME_DIR}")


if __name__ == "__main__":
    main()
