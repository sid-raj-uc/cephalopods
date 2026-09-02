#!/usr/bin/env python3
"""eval_presence_headtohead.py — CLIP detector vs mask area, on ONE identical verified set.

The paper asserts that mask area is a better presence gate than the CLIP+MLP detector
("the VLM is a far stronger presence filter than the detector", Sec. III-C) on the strength of a
single anecdote (534/847 clips came back not-present in the 235B captioning run). The two gates have
never been scored against each other on one verified set. This does that.

DESIGN CONSTRAINTS (all imposed by review, each one load-bearing):

1. REFL-28, not REFL-34. The detector was TRAINED on Right_Left frames, so any reflection video in
   its training manifest must be dropped — from BOTH arms, not just the detector's. Five sessions
   overlap (2026-02-20 at 0954/1724/1754/1824/1854), leaving 28 frames / 22 videos. Scoring the
   segmenter on 34 and the detector on 28 would commit exactly the sin this experiment exists to fix.
   The manifest is normalised to (date, HHMM) because it mixes `date_HHMM` and `date_HHMMSS` filename
   conventions; that normalisation OVER-excludes if two recordings start in the same minute, which is
   the safe direction.

2. The empty-tank negatives are reported DESCRIPTIVELY ONLY — no AUC, no CI, no "complementary"
   claim. There are 19 of them but they come from just 2 source videos, 18 from one recording. n is
   the number of videos, not frames.

3. The comparison is IN-DOMAIN vs ZERO-SHOT and must always be described that way. The detector saw
   1,519 Right_Left training frames across 11 sessions; the segmenter saw zero. If the detector wins
   on reflections that is expected, not impressive; if it loses, that is the striking result.

4. The detector is scored PER FRAME at p_visible >= 0.6. Deployment applies that threshold to >50% of
   frames in a 20 s window, so this is a per-frame PROXY for the deployed gate and is labelled as such.

5. One combined gate only, pre-registered before running: the rank-product of the two scores. (A
   second `min` variant was proposed and dropped — two variants on 28 frames is fishing.)

Everything is read-only: no gate, threshold or default anywhere in the pipeline is changed.
"""
import argparse, csv, json, re, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import caption_openrouter as C
from segment_octopus import OctoSegmenter, _largest_blob
from eval_reflection_presence import auc, fp_at_recall, boot_auc_by_video

DS = REPO / "data" / "dataset_seg_human"
CACHE = REPO / "data" / "fusion_probcache" / "none"
REFL = REPO / "data" / "reflection_negatives"
TRAIN_MANIFEST = REPO / "data" / "frames" / "manifest.csv"
OUT_CSV = REPO / "data" / "presence_headtohead_frames.csv"
OUT = REPO / "data" / "presence_headtohead.json"
DET_THRESH = 0.6


def detector_train_videos():
    """(date, HHMM) of every video in the detector's training manifest, both filename conventions.
    Fails loudly on any unparsed row — a leakage check that matches nothing looks like one that passes."""
    rows = list(csv.DictReader(open(TRAIN_MANIFEST)))
    vids, rl, bad = set(), set(), 0
    for r in rows:
        f = r["path"].split("/")[-1]
        m = re.search(r"(20\d\d-\d\d-\d\d)_(\d{6})_", f) or re.search(r"(20\d\d-\d\d-\d\d)_(\d{4})_", f)
        if not m:
            bad += 1
            continue
        k = (m.group(1), m.group(2)[:4])
        vids.add(k)
        if "Right_Left" in r["path"]:
            rl.add(k)
    if bad:
        sys.exit(f"LEAKAGE CHECK ABORTED: {bad}/{len(rows)} manifest rows unparsed")
    n_rl_frames = sum("Right_Left" in r["path"] for r in rows)
    return vids, rl, len(rows), n_rl_frames


def load_frames():
    """Positives + empty-tank negatives (from the prob cache) and REFL-28 reflection negatives."""
    train_all, train_rl, n_rows, n_rl_frames = detector_train_videos()
    idx = json.load(open(CACHE / "index.json"))
    pos, tank = [], []
    for r in idx["rows"]:
        (pos if r["label"] == "pos" else tank).append(
            {"key": r["key"], "path": str(DS / r["image"]), "cache": CACHE / f"{r['key']}.npz",
             "shape": r["shape"], "video": "seg-test", "kind": r["label"]})

    ridx = json.load(open(REFL / "index.json"))
    refl, dropped = [], []
    for r in ridx["rows"]:
        if r.get("verified") is not True:
            continue
        d, s = r["video"].split("/")
        # Leakage unit is the recording SESSION, not the camera: this repo splits by `date/segment`,
        # and different cameras in one session are the same scene, lighting and animal state at the
        # same moment. Excluding only the Right_Left training sessions would leave 4 further sessions
        # the detector had seen through another camera. Drop for BOTH arms so the sets stay identical.
        if (d, s[:4]) in train_all:
            dropped.append(r["video"])
            continue
        refl.append({"key": r["key"], "path": str(REFL / r["image"]), "cache": None,
                     "shape": None, "video": r["video"], "kind": "refl"})
    print(f"detector manifest: {n_rows} rows, {n_rl_frames} Right_Left frames, "
          f"{len(train_rl)} Right_Left sessions")
    print(f"REFL: {len(refl)} frames / {len({r['video'] for r in refl})} videos "
          f"(dropped {len(dropped)} frames from {len(set(dropped))} sessions overlapping detector training)")
    return pos, tank, refl, sorted(set(dropped)), n_rl_frames, len(train_rl)


def mask_area(S, rec, thresh=0.5):
    if rec["cache"] is not None:                          # positives / empty-tank: reuse cached probs
        pr = np.load(rec["cache"])["prob"].astype(np.float32)
        H, W = rec["shape"]
    else:
        img = cv2.imread(rec["path"])
        pr = S.prob(img)
        H, W = img.shape[:2]
    m = cv2.resize(pr, (W, H), interpolation=cv2.INTER_LINEAR) > thresh
    if m.any():
        m = _largest_blob(m)
    return float(m.mean())


def rank_norm(v):
    v = np.asarray(v, float)
    order = v.argsort()
    r = np.empty(len(v)); r[order] = np.arange(1, len(v) + 1)
    return r / len(v)


def paired_boot_dauc(pos, neg, a_key, b_key, iters=4000, seed=7):
    """Paired cluster bootstrap by video of AUC(a) - AUC(b): the same resampled videos feed both arms."""
    rng = np.random.default_rng(seed)
    pv, nv = {}, {}
    for p in pos:
        pv.setdefault(p["video"], []).append(p)
    for n in neg:
        nv.setdefault(n["video"], []).append(n)
    pk, nk = list(pv), list(nv)
    d = []
    for _ in range(iters):
        P = [x for k in rng.choice(pk, len(pk)) for x in pv[k]]
        N = [x for k in rng.choice(nk, len(nk)) for x in nv[k]]
        d.append(auc([{"area": x[a_key]} for x in P], [{"area": x[a_key]} for x in N]) -
                 auc([{"area": x[b_key]} for x in P], [{"area": x[b_key]} for x in N]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "weights/seg/octo_seg_thin768_lraspp.pt"))
    a = ap.parse_args()

    pos, tank, refl, dropped, n_rl_frames, n_rl_sess = load_frames()
    S = OctoSegmenter(a.ckpt)
    cm, pre, clf, vis, dev = C.load_detector()
    print(f"detector on {dev}; scoring {len(pos)+len(tank)+len(refl)} frames\n")

    allrecs = pos + tank + refl
    pvis = C.score([r["path"] for r in allrecs], cm, pre, clf, vis, dev)
    for r, p in zip(allrecs, pvis):
        r["p_visible"] = float(p)
        r["area"] = mask_area(S, r)

    # pre-registered combined gate: rank-product of the two scores, computed within this frame set
    ra, rp = rank_norm([r["area"] for r in allrecs]), rank_norm([r["p_visible"] for r in allrecs])
    for r, x, y in zip(allrecs, ra, rp):
        r["combined"] = float(x * y)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "kind", "video", "p_visible", "area", "combined"])
        w.writeheader()
        for r in allrecs:
            w.writerow({k: r[k] for k in w.fieldnames})

    res = {"n_pos": len(pos), "n_tank": len(tank), "n_refl": len(refl),
           "n_refl_videos": len({r["video"] for r in refl}), "dropped_sessions": dropped,
           "detector_thresh": DET_THRESH,
           "asymmetry": (f"detector trained on {n_rl_frames} Right_Left frames / {n_rl_sess} sessions; "
                         "segmenter trained on zero -> IN-DOMAIN vs ZERO-SHOT"),
           "caveat_detector": "per-frame proxy; deployment applies p>=0.6 to >50% of frames in a 20s window",
           "caveat_labels": "reflection negatives verified by an AI vision model, not a human — provisional"}

    print("=== REFL-28 (identical frame set for every arm) ===")
    for arm in ("area", "p_visible", "combined"):
        A = auc([{"area": r[arm]} for r in pos], [{"area": r[arm]} for r in refl])
        f90, _ = fp_at_recall([{"area": r[arm]} for r in pos], [{"area": r[arm]} for r in refl], 0.90)
        P = [dict(r, area=r[arm], video=r["video"]) for r in pos]
        N = [dict(r, area=r[arm], video=r["video"]) for r in refl]
        lo, hi = boot_auc_by_video(P, N)
        res[f"refl_{arm}"] = {"auc": round(A, 4), "ci95": [round(lo, 4), round(hi, 4)],
                              "fp_at_recall90": round(f90, 4)}
        print(f"  {arm:11s} AUC {A:.4f}  CI95[{lo:.3f},{hi:.3f}]  FP@R90 {f90:.3f}")

    d, lo, hi = paired_boot_dauc(pos, refl, "area", "p_visible")
    res["delta_auc_area_minus_detector"] = {"mean": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                            "includes_zero": bool(lo <= 0 <= hi)}
    print(f"\n  paired dAUC (area - detector) = {d:+.4f}  CI95[{lo:+.3f},{hi:+.3f}]"
          f"   {'-> INCLUDES 0: no A-beats-B claim' if lo <= 0 <= hi else '-> excludes 0'}")

    dc, lc, hc = paired_boot_dauc(pos, refl, "combined", "area")
    res["delta_auc_combined_minus_area"] = {"mean": round(dc, 4), "ci95": [round(lc, 4), round(hc, 4)],
                                            "includes_zero": bool(lc <= 0 <= hc)}
    print(f"  paired dAUC (combined - area) = {dc:+.4f}  CI95[{lc:+.3f},{hc:+.3f}]"
          f"   {'-> INCLUDES 0: mask area alone is the gate' if lc <= 0 <= hc else '-> excludes 0'}")

    print("\n=== empty tank: DESCRIPTIVE ONLY (19 frames but 2 source videos, 18 from one) ===")
    for arm in ("area", "p_visible"):
        v = [r[arm] for r in tank]
        pv_ = [r[arm] for r in pos]
        res[f"tank_{arm}_descriptive"] = {"median_neg": round(float(np.median(v)), 5),
                                          "median_pos": round(float(np.median(pv_)), 5),
                                          "n_frames": len(v), "n_videos": 2}
        print(f"  {arm:11s} median neg {np.median(v):.4f} vs median pos {np.median(pv_):.4f}  "
              "(no AUC/CI — n=2 videos)")
    fp_det = float(np.mean([r["p_visible"] >= DET_THRESH for r in tank]))
    fp_det_r = float(np.mean([r["p_visible"] >= DET_THRESH for r in refl]))
    res["detector_fp_rate_at_0.6"] = {"empty_tank": round(fp_det, 4), "reflection": round(fp_det_r, 4)}
    print(f"\n  detector FP rate at p>=0.6:  empty tank {fp_det:.3f}   reflection {fp_det_r:.3f}")

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}\n-> {OUT_CSV}")
