# From Footage to Ethogram

Code, frozen results and human labels for a pipeline that turns continuous aquarium video into
a behavioural time series for a captive octopus.

This branch is **code and small artifacts only**. Video clips, cached CLIP/DINOv2/VideoMAE
features and model weights are gigabyte-scale and are not in git; see
[Data not in this branch](#data-not-in-this-branch).

---

## Every claim → the script that produced it

| claim | script | output |
|---|---|---|
| funnel: 1,769 videos, 892.5 h, 60% discarded on probe frames, ≈60 h decoded | `src/harvest_stream.py`, `src/merge_harvest.py` | `data/harvest_ledger_all.json` |
| 5-pass teacher labelling; vote differs from one pass on 13.5% of clips | `src/ensemble_235b.py`, `src/ensemble_235b_vote.py` | `data/ensemble_235b_voted.json`† |
| soft targets: human agreement 0.73 / 0.86 / 0.43 by vote margin | `src/eval_human_vs_ensemble.py` | `data/human_vs_ensemble_results.json` |
| frozen 6-class ethogram dataset, 4,665 clips / 204 videos | `src/build_ethogram_dataset.py` | `src/dataset_etho/v1/snapshot.json` |
| leakage / adequacy / motion-signal checks pass | `src/validate_ethogram_dataset.py` | stdout, exits non-zero on failure |
| rung ladder: 0.100 → 0.400 → 0.530 | `src/train_ethogram.py` | `data/ethogram_ladder_v1.json` |
| backbone swap: DINOv2 +0.047, VideoMAE +0.059 | `src/extract_backbone_feats.py` + `train_ethogram.py --backbone` | `data/ethogram_ladder_{dinov2,videomae}.json` |
| animal pixel resolution: 0.578 crop / 0.574 letterbox / 0.618 mask-crop | `extract_backbone_feats.py --crop` / `--letterbox` | `data/ethogram_lb.json`, `data/ethogram_crop_*.json` |
| 5-member ensemble: **0.665 macro-F1 / 75.4% accuracy** | `src/train_ethogram_fusion.py` | `data/ethogram_fusion_crop.json` |
| head capacity ≈ 0 (and wider is worse) | `src/sweep_ethogram_head.py` | `data/ethogram_head_sweep_*.json` |
| teacher 0.657 vs student 0.576 vs human | `src/eval_ethogram_human.py` | `data/ethogram_human_eval.json` |
| mask geometry: +0.043 was leakage | `src/extract_mask_feats.py`, `src/eval_mask_features.py` | `data/ethogram_mask_features.json` |
| segmentation IoU 0.642 / area err ≈1%; SKEL-50 tip-F1 0.539 | `src/benchmarks.py` | `data/benchmarks.json` |
| segmenter vs zero-shot teacher (0.642 vs 0.374, paired) | `src/eval_teacher_masks.py` | `PAPER_NOTES.md` R19 |
| trained probe vs zero-shot CLIP (0.745 vs 0.450 AUC) | `src/eval_zeroshot_vs_probe.py` | `data/zeroshot_vs_probe.json` |
| presence gate retrain on VLM labels | `src/build_detector_v3.py` | `data/detector_v3/results.json` |
| activity budget, circadian, stimulus response | `src/extract_behaviour_records.py`, `src/analyze_behaviour.py` | `data/behaviour_stats.json` |
| skeleton → arm kinematics | `src/segment_to_skeleton.py`, `src/batch_skeleton_motion.py`, `src/skeleton/` | `data/skeleton_motion.json` |

† `ensemble_235b_voted.json` is 2.5 MB of teacher labels and is part of the **data release**,
not this code branch.

## Human labels (in this branch)

| file | n | what |
|---|---|---|
| `data/human_behaviour_labels{,_v2,_v3}.json` | 456 | behaviour labels, three rounds |
| `data/human_eval_sample_v{1,2,3}.json` | — | the frozen samples those rounds drew from |
| `data/benchmarks.json` | — | frozen SEG-TEST / SKEL-50 / REFL-24 results |

**Read before using the behaviour labels:** every one was collected with the model's
suggestion visible, so they measure *agreement with the model*, not independent accuracy.
The `assisted` flag is recorded per label. A blind round is specified but not yet run.

## Protocol rules

The pipeline broke each of these at least once during development, so they are enforced in
code rather than documented as intentions:

1. **Splits are by source video, never by clip.** A held-out clip from a training video is
   not held out. `validate_ethogram_dataset.py` asserts it.
2. **Frozen sets are never regenerated to suit a result.** Figure sources are pinned rather
   than read live; pointing a figure at a scratch directory silently changed a published
   figure once.
3. **Negatives of different kinds are never pooled** (reflections vs empty tank vs IR).
4. **Holdout videos are excluded from every training source**, not merely the final stage
   (`--holdout-videos`).
5. **Derived artifacts of resumable jobs go stale.** A vote file left un-rederived cost 36%
   of the training set; `check_vote_fresh()` now refuses to build from one.

## Reproducing

```bash
python3 -m venv venv && ./venv/bin/pip install -r src/requirements.txt
# teacher labelling needs OPENROUTER_API_KEY; footage access needs OCTOPUS_USER/OCTOPUS_PASS
cp src/.env.example .env      # never commit .env

./venv/bin/python3 src/validate_ethogram_dataset.py --version v1   # checks before training
./venv/bin/python3 src/train_ethogram.py --version v1              # the rung ladder
./venv/bin/python3 src/train_ethogram_fusion.py                    # the 5-member ensemble
./venv/bin/python3 src/benchmarks.py --tag mytag                   # frozen suites
```

Apple Silicon notes: CLIP and the students run on MPS; **GroundingDINO must run on CPU**
(deformable attention is unstable on MPS). OpenAI CLIP needs `setuptools<81` for
`pkg_resources`.

## Data not in this branch

| artifact | size | where |
|---|---|---|
| 20 s video clips | ~46 GB | footage server / on request |
| cached backbone features (`feats_*`) | ~1 GB | regenerable by `extract_backbone_feats.py` |
| model weights (`weights/`) | ~220 MB | released separately |
| teacher labels + ~25,000 captions | ~10 MB | the data release |

## Full experimental record

`PAPER_NOTES.md` (R1–R35) is the chronological ledger — every measurement with its
provenance, **including the ones that failed and the conclusions that were retracted**
(teacher-label quality was *not* the segmentation ceiling; the mask-feature gain was
leakage; feature-space augmentation is negative). `RESULTS_ETHOGRAM.md` is the tidied
current state of the ethogram work. `src/SEGMENTATION_LOG.md` is the segmentation trail.

## Ethics

All footage is observational, from cameras already installed for husbandry, with no
intervention or manipulation of the animal.
