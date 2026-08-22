"""build_ethogram_dataset.py — freeze the training set for the 6-class ethogram classifier.

Turns the 5-pass 235B ensemble into a trainable dataset: filtered by MEASURED label reliability,
merged, split by SOURCE VIDEO, with SOFT targets from the vote distribution and cached per-frame
CLIP features.

--------------------------------------------------------------------------------------------------
CLASSES (6). Behaviour AND absence in one head, because that is how it is deployed: a clip arrives,
is there an animal, and what is it doing. Precedent: the segmentation work found that training WITH
negatives turned a coin-flip presence gate (AUC 0.50) into 0.86.
    No octopus · Resting/stationary · Exploration/manipulation ·
    Locomotion (crawl/swim) · Reaching out of water · Human/enrichment interaction

MERGE Crawling + Swimming/jetting -> one locomotion class. The model cannot separate them (5 of v2's
40 behaviour errors were Swimming->Crawling, all one direction) and Swimming had 42 clips in 14
videos, too few for a per-class F1 under a video-level split. TRAINING-TIME MAPPING ONLY: the
extraction prompt, ethogram_list_v2.json and every stored record keep the 7-class vocabulary, so
R15's kappa and the human rounds stay comparable and the merge is reversible.
Applied to the VOTE DISTRIBUTION too -- 3 Crawling / 2 Swimming becomes unanimous locomotion, which
is right: the ensemble was certain about locomotion, just not which kind.

DROP `Colour change / defensive`: 1 clip corpus-wide. Unlearnable.

--------------------------------------------------------------------------------------------------
CAMERA-DIRECTIONAL FILTERING -- the important part, and it is measured, not assumed.
From 298 human labels, label reliability depends on the camera AND the direction:

    camera        model says ABSENT -> human agrees      model says PRESENT
    Right_Left    36/36 = 100%                           WRONG 45% of the time
    Right_Right    8/8  = 100%                           0% FP
    Right_Front   18/20 =  90%                           3% FP
    Right_Back     7/8  =  88%                          15% FP
    Right_Top     11/20 =  55%   <- worst                 8% FP

Two opposite failure modes, both physically sensible: Right_Left HALLUCINATES presence (tank-glass
reflections) but is perfect when it says absent; Right_Top (IR) MISSES animals in dim footage but is
reliable when it does see one. So the filter is directional rather than a blanket camera exclusion --
excluding Right_Left outright would have discarded the most reliable hard negatives in the corpus.

    * Right_Left PRESENT  -> EXCLUDED (45% wrong)
    * Right_Left ABSENT   -> kept, full weight (100% agreement; these are the reflection negatives)
    * Right_Top  ABSENT   -> kept at REDUCED WEIGHT (55%) rather than dropped: IR is the largest
                             deployment camera and dropping these leaves the model no IR negatives.
    * everything else     -> kept, full weight
Cell sizes are 8-36 clips, so trust the ordering more than the exact percentages.

--------------------------------------------------------------------------------------------------
SOFT TARGETS. 31% of clips have a split vote, and human agreement tracks the margin (0.726 unanimous
/ 0.864 at 4-of-5 / 0.426 at <=3/5). So the target is the normalised 5-vote distribution, not the
argmax: a 3-2 clip teaches uncertainty instead of false confidence and no rows are discarded. Train
with KL divergence.

SPLIT BY SOURCE VIDEO, never by clip -- this project shipped a clip-level leak once (an apparent
0.49 -> 0.70 gain evaporated under a video-level holdout). The TEST videos double as the reserved
pool for a future BLIND human round, so teacher-reproduction and human-accuracy land on the same
holdout and are directly comparable.

The 154 existing human-labelled behaviour clips are held out of train/val as `human_secondary`. They
are a SUPPORTING figure only, with two caveats that must travel with any number computed on them:
(1) all were labelled `assisted` (the model's answer was on screen) so they measure AGREEMENT, not
accuracy; (2) they span 65 of 82 videos, so their video-mates are in training -- clip-level overlap.

WHY A SEQUENCE, NOT A POOLED VECTOR. The previous behaviour classifier pooled CLIP features and
collapsed onto the majority classes (per-class F1 ~0). Pooling destroys time, and every class here
except Resting is defined by motion. Features are cached as [10, 512] and the model must consume the
sequence.

Output: src/dataset_etho/<version>/{manifest.jsonl, features.npz, human_secondary.jsonl, snapshot.json}
Usage:  venv/bin/python3 src/build_ethogram_dataset.py --version v1
"""
import argparse, collections, json, random, sys, tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import caption_openrouter as C
from ensemble_235b import extract_frames_at, interleaved_draw, DENSE_FPS, N_DRAW

VOTED = REPO / "data" / "ensemble_235b_voted.json"
INDEX = REPO / "src" / "octopus_clips_verified.json"
ROOTS = [REPO / "src" / "octopus_clips_verified", REPO / "data" / "octopus_clips_verified"]
OUTROOT = REPO / "src" / "dataset_etho"
HUMAN = [("data/human_behaviour_labels.json", "data/human_eval_sample_v1.json"),
         ("data/human_behaviour_labels_v2.json", "data/human_eval_sample_v2.json")]

ABSENT = "No octopus"
MERGE = {"Crawling": "Locomotion (crawl/swim)", "Swimming / jetting": "Locomotion (crawl/swim)"}
DROP_LABELS = {"Colour change / defensive"}
IR_ABSENT_WEIGHT = 0.5      # Right_Top absent labels agree with the human only 55% of the time
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 20260822


def vid_of(clip):
    return "/".join(clip.split("/")[:2])


def merged(label):
    return MERGE.get(label, label)


def resolve(clip):
    for r in ROOTS:
        p = r / clip
        if p.exists():
            return p
    return None


def load_human():
    """clip -> human record, for the caveated secondary eval and to hold those clips out of train."""
    out = {}
    for lf, sf in HUMAN:
        lp, sp = REPO / lf, REPO / sf
        if not (lp.exists() and sp.exists()):
            continue
        lab = json.load(open(lp))
        samp = {c["clip"]: c for c in json.load(open(sp))["clips"]}
        for k, v in lab.items():
            if k in samp and not v.get("skipped"):
                out[k] = {**v, "camera": samp[k].get("camera")}
    return out


def video_split(by_video, rng):
    """Whole videos to splits, greedy so every split carries every class. Video-level: no clip leaks."""
    dom = {v: collections.Counter(ls).most_common(1)[0][0] for v, ls in by_video.items()}
    rarity = collections.Counter(dom.values())
    order = sorted(by_video, key=lambda v: (rarity[dom[v]], rng.random()))
    counts = {s: collections.Counter() for s in SPLIT_FRACS}
    assign = {}
    for v in order:
        c = dom[v]
        best, best_def = None, None
        for s, frac in SPLIT_FRACS.items():
            tot = sum(counts[x][c] for x in SPLIT_FRACS) or 1
            deficit = frac - (counts[s][c] / tot)
            if best_def is None or deficit > best_def:
                best, best_def = s, deficit
        assign[v] = best
        for l in by_video[v]:
            counts[best][l] += 1
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = OUTROOT / a.version
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    voted = json.load(open(VOTED))
    idx = json.load(open(INDEX))
    idx = idx if isinstance(idx, list) else idx.get("clips", [])
    motion = {"/".join(str(e.get("clip_path", "")).split("/")[-3:]): e.get("mean_motion")
              for e in idx if isinstance(e, dict)}
    human = load_human()
    print(f"human labels available: {len(human)}")

    rows, drop = [], collections.Counter()
    for k, v in voted.items():
        if v.get("n_passes") != 5 or not v.get("voted"):
            drop["not_5_passes"] += 1; continue
        cam = v.get("camera")
        absent = (not v["present"]) or "not present" in str(v.get("ethogram")).lower()
        if absent:
            # directional trust: Right_Left absent is the most reliable label in the corpus (36/36)
            w = IR_ABSENT_WEIGHT if cam == "Right_Top" else 1.0
            label = ABSENT
        else:
            if cam == "Right_Left":
                drop["right_left_present_45pct_FP"] += 1; continue
            if v.get("ethogram") in DROP_LABELS:
                drop["dropped_label"] += 1; continue
            label, w = merged(v["ethogram"]), 1.0
        if resolve(k) is None:
            drop["file_missing"] += 1; continue
        rows.append((k, v, label, w))
    print(f"selected {len(rows)} clips   dropped: {dict(drop)}")
    if a.limit:
        rows = rows[:a.limit]

    classes = [ABSENT] + sorted({l for _, _, l, _ in rows if l != ABSENT})
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"classes ({len(classes)}): {classes}")

    # split by video, using only non-human rows to decide (human clips are held out either way)
    by_video = collections.defaultdict(list)
    for k, _, l, _ in rows:
        by_video[vid_of(k)].append(l)
    assign = video_split(by_video, rng)

    det = C.load_detector()
    cm, pre, clf, vis, dev = det
    feats, manifest, secondary = {}, [], []
    for n, (k, v, label, w) in enumerate(rows, 1):
        soft = np.zeros(len(classes), np.float32)
        if label == ABSENT:
            # presence votes give the soft target for the absent class
            top, tot = (v.get("present_votes") or "0/0").split("/")
            tot = max(1, int(tot)); nab = int(top) if not v["present"] else tot - int(top)
            soft[cidx[ABSENT]] = nab / tot
            rest = 1.0 - soft[cidx[ABSENT]]
            if rest > 0:                       # spread the residual over the behaviour votes
                dist = {merged(x): c for x, c in (v.get("all_ethograms") or {}).items()
                        if merged(x) in cidx and "not present" not in x.lower()}
                s = sum(dist.values())
                for c2, c2n in dist.items():
                    soft[cidx[c2]] += rest * c2n / s if s else 0.0
        else:
            for lab, cnt in (v.get("all_ethograms") or {}).items():
                m = merged(lab)
                if m in cidx and "not present" not in lab.lower():
                    soft[cidx[m]] += cnt
        if soft.sum() <= 0:
            drop["no_votes_after_merge"] += 1; continue
        soft /= soft.sum()

        with tempfile.TemporaryDirectory() as td:
            fr = extract_frames_at(resolve(k), td, DENSE_FPS)
            if not fr:
                drop["no_frames"] += 1; continue
            pick = interleaved_draw(len(fr), N_DRAW, 1, 5)     # pass-1 grid: deterministic
            batch = torch.stack([pre(C.letterbox(Image.open(fr[i]).convert("RGB"))) for i in pick])
            with torch.no_grad():
                f = cm.encode_image(batch.to(dev)).float()
                f = f / f.norm(dim=-1, keepdim=True)
            feats[k] = f.cpu().numpy().astype(np.float32)       # [10, 512] -- a SEQUENCE

        split = "human_secondary" if k in human else assign[vid_of(k)]
        rec = {"clip": k, "video": vid_of(k), "split": split,
               "label": label, "label_idx": cidx[label],
               "soft": [round(float(x), 4) for x in soft], "weight": w,
               "margin": v.get("ethogram_margin"), "present_votes": v.get("present_votes"),
               "unanimous_after_merge": bool(float(soft.max()) == 1.0),
               "camera": v.get("camera"), "date": v.get("date"),
               "mean_motion": motion.get(k), "n_frames_available": len(fr), "frames_used": pick}
        manifest.append(rec)
        if k in human:
            h = human[k]
            secondary.append({**rec, "human_present": h.get("present"),
                              "human_ethogram": merged(h.get("ethogram")) if h.get("ethogram") else None,
                              "human_label": (ABSENT if h.get("present") is False
                                              else (merged(h["ethogram"]) if h.get("ethogram") else None)),
                              "human_assisted": h.get("assisted"), "human_seconds": h.get("seconds")})
        if n % 250 == 0:
            print(f"  featurised {n}/{len(rows)}", flush=True)

    with open(out / "manifest.jsonl", "w") as fh:
        for r in manifest:
            fh.write(json.dumps(r) + "\n")
    with open(out / "human_secondary.jsonl", "w") as fh:
        for r in secondary:
            fh.write(json.dumps(r) + "\n")
    np.savez_compressed(out / "features.npz", **feats)

    trainable = [r for r in manifest if r["split"] in SPLIT_FRACS]
    maj = (max(collections.Counter(r["label"] for r in trainable).values()) / len(trainable)) if trainable else 0
    print(f"\n{'split':<17}{'clips':>7}{'videos':>8}   " + "".join(f"{c[:13]:>15}" for c in classes))
    snap = {"version": a.version, "seed": SEED, "classes": classes, "merge": MERGE,
            "dropped_labels": sorted(DROP_LABELS),
            "camera_directional_filter": {
                "right_left_present": "EXCLUDED (45% presence FP vs human)",
                "right_left_absent": "kept, weight 1.0 (36/36 human agreement)",
                "right_top_absent": f"kept, weight {IR_ABSENT_WEIGHT} (11/20 = 55% agreement)",
                "note": "cells are 8-36 clips; trust the ordering, not the exact rates"},
            "frame_grid": {"dense_fps": DENSE_FPS, "n_frames": N_DRAW, "pass": 1},
            "soft_targets": "normalised 5-vote distribution after merge; train with KL divergence",
            "n_clips": len(manifest), "n_videos": len({r["video"] for r in manifest}),
            "majority_baseline_trainable": round(maj, 4), "drops": dict(drop), "splits": {},
            "human_secondary_caveats": [
                "all human labels were ASSISTED -> they measure AGREEMENT, not accuracy",
                "their video-mates are in train (65 of 82 videos) -> clip-level overlap",
                "SUPPORTING figure only; the primary test is held-out videos"],
            "test_videos_reserved_for_blind_human_round": []}
    for s in list(SPLIT_FRACS) + ["human_secondary"]:
        g = [r for r in manifest if r["split"] == s]
        cc = collections.Counter(r["label"] for r in g)
        vc = {c: len({r["video"] for r in g if r["label"] == c}) for c in classes}
        print(f"{s:<17}{len(g):>7}{len({r['video'] for r in g}):>8}   " +
              "".join(f"{str(cc[c])+'/'+str(vc[c])+'v':>15}" for c in classes))
        snap["splits"][s] = {"clips": len(g), "videos": sorted({r["video"] for r in g}),
                             "per_class_clips": dict(cc), "per_class_videos": vc}
    snap["test_videos_reserved_for_blind_human_round"] = snap["splits"]["test"]["videos"]
    json.dump(snap, open(out / "snapshot.json", "w"), indent=1)
    print(f"\nmajority baseline (trainable splits): {maj:.1%}   cells are clips/videos")
    print(f"test videos reserved for a future BLIND human round: {len(snap['splits']['test']['videos'])}")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
