# PAPER_NOTES.md — running results ledger for the research paper

The project ends in a **research paper**. This file is the single running record of paper-worthy
results: metrics **with the test set they were measured on**, ablations, failure cases, figures, and
the methodology decisions behind them. Update after every meaningful experiment. Keep failures — they
are the ablations / limitations sections.

**Provenance rule:** every metric records *date · model/config · dataset · test set*. Any "A beats B"
claim must be a **head-to-head on ONE human-verified held-out set** (two numbers from different test
sets are not comparable — see Open Rigor Items).

---

## Working title / framing
Automated **behaviour & affective-state analysis** of a single octopus ("Nity") from continuous
multi-camera aquarium footage: detection → clip extraction → structured behavioural extraction →
distilled local models (caption, segmentation) → ethological time-series (activity budget, circadian,
stimulus-response). Framed as **arousal / behavioural-state**, not emotion.

## Contributions (draft)
1. A full pipeline turning raw 24/7 footage into a quantified behavioural time-series.
2. Teacher→student distillation for **local, laptop-runnable** caption + (WIP) segmentation models.
3. Ethological findings on Nity (activity budget, circadian rhythm, human-presence stimulus response).
4. Methodology lessons (below) that generalize to aquarium/animal video analysis.

---

## Results so far

### R1 — Behaviour analysis (the headline scientific results)
- Corpus: **3,205 clips** structured-extracted via Qwen3-VL-235B (OpenRouter), $2.22, ~$0.0006/clip.
  3,083 present. Full run 2026-07-20.
- **Activity budget:** 41% exploration/manipulation · 33% resting · 14% human-interaction · 9% reaching-out · 2% crawling · 1% swimming.
- **Circadian:** visible-activity ~1–5% overnight → **45% peak at 17:00** (13:00–19:00 plateau) + dawn bump 05–06h. Exposure-normalized (present ÷ all extracted windows/hour).
- **Stimulus response:** human presence **nearly doubles motion (0.045→0.095)** and lifts arousal **0.46→0.68**.
- Colour (colour cameras only): dark_red_brown most common at baseline (~16%) vs during human interaction (~6%).
- Artifacts: `data/behaviour_stats.json`, `data/behaviour_dashboard.html`, published artifact "Nity — behavioural profile".
- CAVEATS (limitations section): `context="enrichment_object"` fires ~66% (means "object in tank", not active enrichment); presence gate dirty upstream. Rate/response *contrasts* robust; absolute levels shift after detector retrain.

### R2 — Caption student (distillation)
- Qwen3-VL-2B + LoRA r16/α32, distilling 235B teacher captions. 3066 train / 392 val, 576 steps (DONE 2026-07-15).
- **Eval (50 held-out val): base emb-sim 0.702 / rougeL 0.269 → LoRA 0.834 / 0.455.**
- Local 4-bit MLX deploy: `models/qwen3vl2b_caption_v1_mlx_4bit` (1.7 GB, ~3 s/caption on 16GB Mac, no GPU).
- Cross-platform HF backend added (base Qwen3-VL-2B + LoRA) for non-Apple hardware.

### R3 — Segmentation (auto-labeler + tiny student) — A100 run 2026-07-23
- Auto-labeler: **GroundingDINO-tiny (box) → SAM2 (mask)**, seed = most-confident frame, **video propagation**
  (temporal consistency) + largest-blob + area-continuity. Phase-0 validated on 4 cameras (before/after):
  IR tool-bleed 11.8%→6.5%, colour bg-bleed 15.5%→5.8%, reflection camera rejected by low seed conf (~0.50).
- Labeled 1,824 colour clips → **4,412 (image,mask) pairs / 77 videos**.
- **Mask pixel-IoU: plateaus ~0.47** (bar 0.85) across TinyUNet(ch8/16/32), LR-ASPP, aug, +IR.
  Diagnosed **video-diversity generalization gap** (only 62 train videos; train IoU 0.68 / val 0.47;
  fails by *mislocating* a right-sized blob). → limitations + motivates DATA_PLAN.
- **Presence gate WIN:** v1 (positives-only) = random (AUC 0.50). **v3 = 4,412 pos + 1,388 empty-mask negatives
  → AUC 0.86 overall, 0.99 vs reflections, 0% reflection-FP at area≥0.01 (88% present-recall).** Model
  `weights/seg/octo_seg_v3_lraspp.pt` (LR-ASPP, 3.2M params). Full trail: `src/SEGMENTATION_LOG.md`, `results/segmentation/`.
- IR (Right_Top) unusable as-is (GroundingDINO low-conf on greyscale, SAM2 grabs bright tools) — needs Phase-0 IR fix.

### R3b — Segmentation diversity retrain (Modal, 2026-08-07) — tests the R3 diversity diagnosis
- The diverse-footage harvest (R5) closed the loop: **530 clips / 276 distinct videos / 149 dates** auto-labeled
  on Modal (A10G, `src/modal_seg_train.py`, GD+SAM2 teacher, min_seed_conf 0.60) → **178 accepted / 732 pairs**
  (345 low-conf recoverable via a lower-conf pass). Merged with old v1 and retrained (LR-ASPP, 60 ep, split BY VIDEO).
- **Head-to-head (best val IoU):** old **62 vid / 4,412 pairs = 0.468** (soft same-week val) → harvest **new-only
  100 vid / 732 pairs = 0.245** (overfits: 588 frames too few) → **merged 176 vid / 5,144 pairs = 0.494** (best,
  on a HARDER diverse-date 35-video val). Model `weights/seg/octo_seg_merged_lraspp.pt`.
- **Interpretation (paper):** diversity helps *robustly but modestly* (+0.026, and on a genuinely harder val);
  the merged val plateaus flat at ~0.49 with NO overfitting. Diversity helped up to ~176 videos, then flat.

### R3c — HQ-teacher upgrade: an HONEST NEGATIVE RESULT (2026-08-08)
Hypothesis: the ~0.49 plateau is teacher-label quality (a distilled student can't beat noisy GD+SAM2-tiny masks).
Motivating evidence: the merged model scored 0.49 vs tiny masks but 0.70 vs HQ masks. Test: upgrade teacher to
**GD-base + SAM2-large**, re-label ALL clips (harvest 740 + old 3,991 = 4,731 HQ pairs), retrain.
- **Result: held-out val IoU FLAT** — tiny 0.494 → 14%-HQ 0.508 → 100%-HQ **0.506**. Clean labels did not help.
- **Why the hypothesis failed:** the 0.70-vs-HQ figure was **train leakage** (evaluated on training clips). The
  clean held-out HQ-vs-HQ number is ~0.51, same as tiny. **Teacher-label quality was NOT the ceiling.**
- **What the ceiling actually is:** the tiny student generalizes to **right-SIZED but mis-LOCALIZED masks**
  (areaErr **1.4%**, Dice 0.62, IoU 0.50). Not fixable by labels or data (both exhausted) — it's a per-frame
  localization limit → points to a **temporal** student as the real lever.
- **Reframe for the paper:** pixel-IoU-vs-teacher-masks is a weak metric here (teacher isn't perfect GT). The
  project needs **area/posture + presence**, and area is accurate to 1.4% — so the model may already be adequate
  for its downstream ethological use. **Open rigor item: human-verified mask val + downstream-task eval** decide
  whether ~0.5 IoU actually matters. Good methodology lesson: *always check train/val leakage before trusting a
  "measure against cleaner labels" signal.*

### R4 — Negative results / ablations (keep for the paper)
- **Behavior classifier** (frozen CLIP feats → MLP): 45% val acc, *below* 50% majority baseline. Lesson: static pooled features can't classify behaviour — needs temporal/motion features.
- **Segmentation arch/aug ablations:** no gain from ch8→ch32, LR-ASPP, or strong augmentation — confirms the limit is data, not model.
- **235B as presence filter:** running 235B over 847 "verified" clips, **534 (63%) came back not-present** — the CLIP+motion extractor massively over-extracts (esp. Right_Left reflections). VLM is a far better presence filter than the detector.

---

## Methodology lessons (paper "what worked / what we learned")
- **Letterbox, not center-crop** for CLIP — aspect-ratio mismatch (not architecture) caused poor field performance.
- **Absolute motion (changed-pixel fraction), not per-video normalized** — normalization passes static videos (lamp flicker). Mask the burned-in timestamp region.
- **Teacher→student distillation** gives local, cheap, offline models (big VLM/SAM2 → tiny student).
- **Temporal consistency (SAM2 video propagation)** kills *transient* auto-label errors — but NOT *consistent* ones (a reflection present every frame). Gate on seed confidence + drop reflection camera.
- **Structured extraction > captioning** for behaviour: a caption is one field; posture/colour/context need a JSON re-prompt.

## R5 — Diverse-footage harvest (data-gen, in progress 2026-07-23)
- **Diagnosis:** whole corpus was 7 dates/one week → both students diversity-limited. Server actually holds
  ~6 collections / ~5 animals; **Nity alone ~209 days**. (Method note: server exposes crawlable HTML listings.)
- **Network finding (measured, Colab→server):** ~5 MB/s download; **parallelism barely helps** (1→5 streams =
  1.6×, near-total server-side cap) → single CPU box, 2–3 streams, NO GPU (network-bound; GPU idles).
  Stream-scan ~5 video-sec/s → stream + early-exit, never bulk-download.
- **Method:** `src/harvest_stream.py` (Modal CPU) — probe-first empty-skip (10 seek-frames; skip if
  `p_visible<0.5` everywhere) + **visibility-only gate** + 2 clips/video (60s spread) + early-exit. A/B (same
  8 vids): motion-gate 1 clip-video → visibility-gate **4 clip-videos** (motion gate was discarding
  still-but-visible octopus, which IS good seg/caption data). Detailed coverage ledger records probe points +
  `unscanned_sec` per video so skipped footage can be mined later.
- Full run: 1,769 Nity colour videos; projected ~6–12 h.

## R6 — Skeleton, tracking and kinematics (2026-08-13/15)
Downstream of segmentation: silhouette → anatomical graph (mantle/head/8 arms) → per-arm kinematics.
Code: `src/skeleton/`, `src/segment_to_skeleton.py`, `src/batch_skeleton_motion.py`. Full trail:
`src/SEGMENTATION_LOG.md`.

**Skeleton extraction phases** (frozen `data/skel_bench50`, 50 frames / 20 videos, model-mask input,
tip-correctness guarded). ⚠️ **All rows below are PRE-GATE and are being re-measured** — commit
`8343d2a` added anti-mess gates as unconditional defaults inside `select_arm_paths`, so even the
"baseline" row is no longer reproducible by today's code.
| phase | arms/frame | tip-match |
|---|---|---|
| baseline extractor | 2.82 | 0.876 |
| + selection floors (2.5×/0.30, prefix 0.70) | 3.65 | — |
| + prep (bin_thresh 96, spur width_factor 0.35) | 4.16 | 0.851 |
| + thin768 segmentation (768², Tversky β=0.8) | 4.64 | — |
| + SAM2 mask refine (offline) | 5.04 | 0.792 |
| clean human-mask ceiling (pre-gate) | 6.15 | — |
- Anatomical head (neck constriction on the mantle→crown line) replaced the 2nd-distance-peak head:
  "plausible" 9%→96% at the time; the same check later read 86% (refine, pre-gate) and **80%**
  post-gate. The "plausible" criterion is loose (head merely between mantle and crown) and is being
  replaced by pixel error against human head clicks (`data/skel_bench50/head_gt.json`, `src/skel_head_eval.py`).
- Measured NULLS: hysteresis on the seg probability field (+0.08 arms, noise — the student genuinely
  does not see the thin arms); zoom-2-pass segmentation (4.38 vs 4.60 — crops are OOD for the student).
- **Anti-mess gates (2026-08-15, `8343d2a`)**: unique-suffix (unshared portion ≥ max(2× root radius,
  30% of length)) + tip-thinness (tip clearance ≤ 0.55× root radius). Motivated by visible tangle
  (duplicate late-forking arms, stubs ending in the fat body). Effect on the `skel_bench_latest`
  harness (no tip guard, SAM2 refine): **4.80 → 3.48 arms/frame**, head-plausible 80%.
- ⚠️ **Provenance defect (recorded honestly):** the figure "≥6 arms rose 3/50 → 17/50" quoted in the
  tex came from a live UI header, not a logged artifact, and is **not reproducible**; the current
  on-disk post-gate value is 11/50. Do not cite 17/50.

**Tracking v2** (frozen 10-clip set, `src/skel_eval_tracking.py`, metrics `src/skeleton/track_metrics.py`):
| run | teleport | arms | verdict |
|---|---|---|---|
| baseline (centroid prior) | 14.85% | 5.1 | — |
| flow prior, unchanged gates | 15.71% | 4.9 | ❌ worse — a better prior admits more noisy detections |
| **flow + tightened gates (adopted)** | **14.27%** | 4.9 | ✅ |
| global tracklet association (2 tunings) | 15.07 / 14.35% | 4.2–4.3 | ❌ negative, kept opt-in |
- **occluded_frac = 41.7%**: nearly half of arm samples in naive tracking were evidence-free holds;
  `compute_motion` now emits rows only from `detected`/`fitted` samples (teleport-confident 16.5% >
  overall 14.3% — held nodes' fake stillness was flattering the average).

**Kinematics × behaviour cross-validation** (state-gated arm-tip speed vs VLM behaviour label):
resting 63 · crawling 101 · human-interaction 136 · exploration 139 · reaching 159 px s⁻¹.
⚠️ n = 41 clips only (crawling n=2, resting n=4), no significance test, speeds in crop-pixel units
(camera-distance confound unaddressed), and computed with the PRE-GATE detector.

## R7 — Frozen benchmark suite (2026-08-15)
`BENCHMARKS.md` + `src/benchmarks.py` — one runner, tagged results in `data/benchmarks.json`,
auto-generated LaTeX table. Three suites: **SEG-TEST** (122 human-mask frames from 5 held-out
videos + 19 negatives), **SKEL-50**, **TRACK-10**.
- **Metric fix:** arms/frame is *not* a score — the anti-mess gates improved the output while the
  count fell 27%. SKEL-50's headline is now **arm-tip F1** vs the human mask's protrusions (greedy
  1-1 within 5% of the diagonal, GT capped at 8), penalising both spurious and missed arms.
- **SEG-TEST head-to-head (same leak-free test, 2026-08-15):**
  | model | IoU mean | IoU median | area err | presence AUC |
  |---|---|---|---|---|
  | clean512tv (paper's current headline) | 0.6075 | 0.6661 | 1.07% | 0.718 |
  | **thin768** | **0.6415** | **0.7193** | 1.05% | **0.794** |
  → thin768 wins on every metric; **promoted to the paper's headline seg model 2026-08-15**
  (abstract/contributions/Sec. V now quote 0.642/0.719 + AUC 0.794 from this one model). This also fixes
  a rigor defect the paper review found: the abstract currently pairs clean512tv's IoU with a
  presence AUC measured on a *different* model (v3 aug-LR-ASPP), reading as one system.
- **SKEL-50 first run (thin768, no refine, post-gate):** tip-F1 **0.419** (precision 0.712,
  **recall 0.353**), arms/frame 3.24 → the gates are over-strict; gate grid in progress.
- **Gate frontier (2026-08-15, `src/skel_gate_grid.py`, corrected GT, 50 frames, no refine;
  artifact `data/skel_diag/gate_grid_result.json`)** — this is what the paper's Table II reports:
  | gates (uniq-scale, uniq-frac, tip-ratio) | P | R | F1 | dup | arms |
  |---|---|---|---|---|---|
  | off (0, 0, ∞) | 0.659 | 0.562 | 0.565 | 0.076 | 4.64 |
  | (1.0, 0.15, 0.85) | 0.673 | 0.535 | 0.550 | 0.038 | 4.28 |
  | **(1.5, 0.20, 1.00) SHIPPED** | 0.722 | 0.502 | 0.539 | 0.000 | 3.68 |
  | (2.0, 0.30, 0.55) 1st attempt | 0.760 | 0.468 | 0.520 | 0.000 | 3.24 |
  ⚠️ The **dup rate is defined by the same unique-suffix criterion the gates enforce**, so strictly
  gated rows are 0 by construction — informative mainly for the gates-off row. Gates-off maxes F1
  but restores the visible tangle; we ship the F1-suboptimal point because downstream tracking needs
  arm identities. **NOT claimed:** that the GT fix *changed the ranking* of configurations — for the
  only pair measured under both GTs the order held (old GT: shipped 0.441 > 1st attempt 0.419; new
  GT: 0.539 > 0.520). What is claimed is that pre-fix recall/F1 are incomparable with post-fix ones,
  so any old ranking had to be re-measured.
- **Kinematics n (recomputed 2026-08-15 from `behaviour_records.json`, same filter as
  `make_figures.py` Fig. 4):** 41 clips carry state-gated kinematics; **40** after dropping
  `behavior="uncertain"` — resting 4 · crawling 2 · human 13 · exploration 17 · reaching 4.
  The paper quotes n=40 (the plotted set). A 149-clip / 66-video recompute with video-level
  statistics is pending and will supersede these medians.

## Open rigor items (must close before paper claims)
- **No shared human-verified held-out test sets yet.** Segmentation has none; captioning has partial
  (`data/caption_training_set.json`). Every "A beats B" (e.g. seg gate vs CLIP gate) needs a head-to-head
  on ONE verified set. This is the DATA_PLAN Phase-D deliverable.
- **Footage diversity:** all 13,342 clips from **7 dates (one week, Feb 2026)** — the binding limit for
  both students. DATA_PLAN addresses it (harvest more distinct days).

## Figure/asset inventory (for the paper)
- `data/segmentation_demo/*.mp4` + `phase0_out/` — seg before/after (IR tool-bleed, colour bleed, reflection).
- `data/behaviour_dashboard.html` + artifact — activity budget, circadian, stimulus-response charts.
- `results/segmentation/*.log` — training curves / sweep logs.

## Pointers
Plans: `src/DATA_PLAN.md`, `src/SEGMENTATION_PLAN.md`, `src/TRAINING_PLAN.md`. Trails: `src/SEGMENTATION_LOG.md`.

## BLOCKED — VLM-250 reliability study (2026-08-15)
Code is complete and verified end-to-end locally (`src/vlm_reliability.py`: frames extracted,
detector-scored, disjoint frame set selected correctly), but every API call returns
**HTTP 401 `{"error":{"message":"User not found"}}`**. The key in `.env` is present and well-formed
(`sk-or-v1…`, 73 chars) but is rejected by OpenRouter — revoked/expired account, not a code fault.
**Unblocks with a fresh `OPENROUTER_API_KEY`; then just run**
`venv/bin/python3 src/vlm_reliability.py --run` (~$0.17, 250 clips, resumable).
This is the highest-value open rigor item: every headline behavioural result is grouped by labels
this extractor produced, and their reliability is still unmeasured.

## R7b — Segmentation training configuration (for the paper's reproducibility section)
Deployed model `octo_seg_thin768_lraspp.pt`, trained on Modal (A10G) via `src/modal_seg_train.py`:
- **Architecture** LR-ASPP / MobileNetV3-Large head, `base_ch=16`, **3.218 M parameters**
- **Input** 768×768 (`--in-size 768`); **batch** 8; **optimiser** Adam, **lr** 3e-4 with cosine schedule
- **Epochs** 60; **augmentation** "strong" (h-flip, affine rotate/translate/scale applied to image and
  mask in lock-step, brightness/contrast jitter ±25%, mild sensor noise)
- **Loss** focal Tversky (α=0.2, β=0.8 — β>α penalises false negatives, i.e. missed thin arms)
  + 0.5·BCE for stable pixel gradients
- **Data** 5,143 pairs / 183 source videos = human-verified masks + GD+SAM2 teacher labels
  (old-HQ 3,991 + harvest-HQ 740); **split BY SOURCE VIDEO**, with 5 test videos forced out of *all*
  training sources via `--holdout-videos` (leakage guard added after the incident in R3c)
- **Selection** best epoch by validation IoU

## R8 — Test-time temporal fusion: a NEGATIVE for masks, an unexpected WIN for presence (2026-08-15)
Motivation: R3c concluded the mask model fails by *mislocalizing* a correctly-sized blob and asserted
"a temporal student is the real lever" — an untested claim in the paper's limitations. Also a
reporting defect: every published IoU is single-frame, while the deployed skeleton path
(`segment_to_skeleton.py`, `EMA_ALPHA=0.45`) thresholds an EMA-smoothed probability map.
Method: `src/temporal_fusion.py` + `benchmarks.py --fusion {none,ema,flow}`; neighbours t±1,±2 warped
onto t with DIS optical flow, fused by per-pixel median. **Frame-alignment trap avoided:**
`seed_frame` indexes the labeller's `ffmpeg fps=2, scale='min(1024,iw)'` list, NOT raw video frames,
so neighbours are produced by re-running that identical extraction and asserting the regenerated
frame matches the stored labelled image (align_err 0.4–0.55 vs tolerance 12; **0/141 failures**).

SEG-TEST (122 human-mask frames, 5 held-out videos, 19 negatives), thin768:
| fusion | IoU mean | IoU median | area err | presence AUC |
|---|---|---|---|---|
| none (single frame — what the paper reports) | 0.6415 | 0.7193 | 1.05% | 0.794 |
| flow ±2 (DIS-warped median) | **0.5109** | **0.5505** | 1.06% | **0.9495** |

- **NEGATIVE (pre-registered kill criterion met, ≥+0.01 mean IoU required):** temporal fusion does
  **not** fix mislocalization — it makes masks materially worse (−0.131 mean, −0.169 median IoU),
  consistent with boundary blurring on a fast-deforming animal. The paper's limitation must be
  rewritten from "a temporal student is the real lever" to "**test-time** temporal fusion does not
  fix mislocalization" (a temporal *trained* student remains untested, but this removes the cheap
  evidence for it).
- **UNEXPECTED POSITIVE:** presence AUC **0.794 → 0.9495**. Fusion washes out single-frame
  hallucinations that are inconsistent across neighbouring frames, which is exactly the pipeline's
  dominant false-positive mode (reflections / empty tank). Body-area error is unchanged (1.05→1.06%),
  so mask *size* survives while boundary fidelity degrades — a good trade for a presence GATE and a
  bad one for morphology.
  ⚠️ CAVEAT: only **19 negatives** in the leak-free holdout, so this AUC has wide uncertainty; it
  should be re-measured on more negatives before being claimed as headline.

## R9 — Reflection robustness of the DEPLOYED model, measured for the first time (2026-08-15)
The paper reports presence AUC 0.794 for `thin768` and describes the system as reflection-robust.
Those are two different claims: the 0.794 came from **19 empty-TANK negatives on the same cameras as
the positives**; the *reflection* failure mode (Right_Left — the camera sees the room and a mirrored
human through the glass, and the CLIP detector fires at p_visible=1.0) was only ever measured for the
**v3 negatives model**, never for the deployed thin768. R9 closes that gap.

**Leakage assertion (verified, not assumed):** thin768's training set `/dataset_seg_thin768` =
**4,965 images, 0 of them Right_Left** (checked file-by-file on the Modal volume; the camera is
excluded by construction in `auto_segment.py` and absent from the human label set). Reflections are
therefore a leak-free negative source for this model. New: `src/reflection_negatives.py`,
`src/eval_reflection_presence.py`.

### Right_Left is NOT a pure-reflection camera — 10–20% of frames contain the real animal
42 frames reviewed at full resolution before scoring (indices 0-33 are one frame from each of the 34
 distinct Right_Left source videos; 34-41 add second frames).
**3/42 (7%) unmistakably contain the octopus** (animal spread on the glass with its own mirror image
beside it; arms with clearly resolved suckers), and **5 more are ambiguous → up to 19%**. Frames were
labelled conservatively: ambiguous frames are excluded from the negative set, never counted as empty.
- This revises the 2026-07-05 reading of the 235B result ("534/847 clips came back not-present, almost
  all Right_Left reflections → drop Right_Left"). Right_Left is **mostly** empty, not **purely** empty,
  so the standing "drop Right_Left" rule is discarding a real, if sparse, source of animal data.
- It also re-confirms the 2026-06 hard-negative lesson: an assumed-negative pool must be verified
  before it is trained or scored on. Assuming these 30 frames were empty would have injected a 10–20%
  label error straight into the headline metric.
- ⚠️ PROVENANCE: this review was done by an **AI vision model, not a human**. It is staged for human
  confirmation (`data/reflection_negatives/`, contact sheets + `ui/review_hardneg.py`). Treat the exact
  percentage as provisional; the qualitative finding (some frames plainly contain the animal) is not
  in doubt — see `data/reflection_negatives/pilot_1.jpg` #11 and `pilot_4.jpg` #24, #29.

### Presence separation, by negative type (never pooled), thin768 @ threshold 0.5
n for the CI is the number of **videos**, not frames; CIs are cluster-bootstrapped by source video.
| negative type | n | AUC | AUC CI95 (by video) | FP@recall .90 | FP@recall .80 | FP@area>=.01 | median neg area |
|---|---|---|---|---|---|---|---|
| empty tank (same cameras) | 19 frames | 0.7942 | [0.626, 0.932] | 0.316 | 0.210 | 0.316 | 0.0036 |
| reflection (Right_Left)   | 34 frames / 27 videos | **0.9214** | [0.871, 0.964] | 0.235 | 0.118 | 0.176 | 0.0035 |
(positive median mask area 0.0325; the empty-tank AUC reproduces the benchmark's 0.794 exactly, which
validates the harness.)

**CORRECTION (2026-08-15, same day).** I first wrote this up as "the assumed failure mode is
backwards — the model rejects reflections (0.921) better than the empty tank (0.794)". That
comparison does not hold up and is withdrawn. The empty-tank negatives are **19 frames from only 2
source videos, 18 of them from `2026-02-21/183003` alone** — effectively a single-video estimate.
Comparing it against a 27-video reflection estimate is not a like-for-like contrast, so the *ordering*
of the two AUCs cannot be asserted. Reported descriptively only, with no CI and no A-beats-B claim.

**What DOES survive, and it is the more useful finding:**
1. **The reflection failure mode is comfortably handled.** AUC **0.9214** across 27 source videos,
   CI95 [0.871, 0.964], FP at the deployed gate (area>=0.01) **0.176**. This is a properly-powered,
   leak-free measurement and it validates the paper's "reflection-robust" claim for the first time.
2. **The paper's published presence AUC of 0.794 is effectively a ONE-VIDEO number.** That is a defect
   in the benchmark, not a property of the model: 18/19 of its negatives come from one recording, so it
   is a near-meaningless population estimate and its CI [0.626, 0.932] is correspondingly useless. It
   must be either re-based on negatives drawn from many videos, or reported descriptively with the
   n=2-videos caveat stated. **This is the highest-value fix available to the presence section** and
   it was invisible until negatives were counted BY VIDEO rather than by frame.
3. Lesson, consistent with the leakage rule: **count n in videos at every stage, including the
   negatives.** We applied by-video discipline to training splits and to the kinematics statistics but
   never to the negative sets, and a headline benchmark number silently rested on one recording.

**Consequence for the R8 follow-up:** the referee's pre-registered early-stop was AUC(none) >= 0.93 on
reflections. Measured **0.9214** (stable: 0.9173 on the first 24 negatives, 0.9214 on 34 / 27 videos, so the
estimate is not an artifact of the smaller pilot) — just under the line, so the cycle is not killed, but the headroom
for fusion on this negative type is only +0.079, and the referee's second criterion (>= +0.05 AUC gain)
must clear that ceiling to count.
