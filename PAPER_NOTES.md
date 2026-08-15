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
  The paper quoted n=40 (the plotted set) until 2026-08-15; **superseded by R12** (146 clips /
  66 videos, video-level statistics). The n=40 medians are retained here only as the historical
  row and are referenced in the paper as the earlier, mildly optimistic estimate.

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

## R10 — CLIP detector vs mask area as a presence gate, head-to-head on ONE verified set (2026-08-15)
Closes the standing rigor item ("every 'A beats B' needs a head-to-head on one verified set"). The
paper's Sec. III-C claim that the detector is the weak presence filter rested on a single anecdote
(534/847 clips came back `octopus not present` in the 235B captioning run). The two gates had never
been scored against each other. New: `src/eval_presence_headtohead.py`; per-frame scores in
`data/presence_headtohead_frames.csv`.

**REFL-28 — the benchmark this had to be run on.** The detector was TRAINED on Right_Left frames, so
its training sessions must be dropped — and dropped from **both** arms, or the head-to-head commits the
sin it exists to fix. Leakage unit is the recording **session** (`date/segment`), not the camera: this
repo splits by session, and two cameras in one session are the same scene, lighting and animal state at
the same instant. Excluding only the *Right_Left* training sessions leaves 4 further sessions the
detector had already seen through another camera (my first run made exactly this error and got
33 frames; corrected → **28 frames / 22 videos**, dropping 6 frames from 5 sessions).
R9's REFL-34 number stands as a separately-scoped measurement (thin768 is Right_Left-free, so it needs
no exclusion); the REFL-28 row is reported beside it, not over it — segmenter AUC 0.9214 (34 fr/27 vid)
vs 0.9315 (28 fr/22 vid).

### Results — REFL-28, identical frame set for every arm
| arm | AUC | CI95 (cluster-bootstrap by video) | FP @ present-recall 0.90 |
|---|---|---|---|
| **mask area** (thin768) | **0.9315** | [0.884, 0.964] | **0.214** |
| CLIP detector `p_visible` | 0.8053 | [0.747, 0.864] | 0.679 |
| combined (rank-product) | 0.9186 | [0.884, 0.946] | 0.286 |

- **Paired ΔAUC (area − detector) = +0.1263, CI95 [+0.056, +0.198] — excludes 0.** Mask area is the
  better reflection gate, and now on a leak-free verified set rather than an anecdote.
- **The asymmetry makes this stronger, not weaker.** The detector is **in-domain** here — it trained on
  **1,519 Right_Left frames across 11 sessions**. The segmenter is **zero-shot**: not one Right_Left
  frame in its 4,965 training images. The zero-shot model beats the in-domain one by 0.126 AUC. Must be
  written as in-domain vs zero-shot wherever it is claimed.
- **Detector false-positive rate at its deployed threshold (p_visible ≥ 0.6): 32% of reflection frames
  and 68% of empty-tank frames.** This is the concrete number behind "the extraction over-extracts",
  replacing the 63%-not-present anecdote.
- **PRE-REGISTERED NEGATIVE — the combined gate is dead.** ΔAUC (combined − area) = −0.0129,
  CI95 [−0.032, +0.009], includes 0. Kill criterion met: the two signals are redundant, **mask area
  alone is the gate**. (One variant only, rank-product, fixed before running; a `min` variant was
  proposed and dropped as fishing on n=22 videos.)
- **Empty tank: descriptive only, no AUC and no CI** — 19 frames but **2 source videos, 18 from one**.
  Median mask area 0.0036 (neg) vs 0.0325 (pos); median p_visible 0.8179 (neg) vs 0.9989 (pos).

CAVEATS that must travel with any use of this: (a) the detector is scored **per frame at p≥0.6**, a
**proxy** — deployment applies that threshold to >50% of frames in a 20 s window, so this is not "the
deployed gate"; (b) the reflection labels are **AI-verified, not human-verified**, so this stays in
PAPER_NOTES and out of the .tex until human review of the 28 frames; (c) read-only study — no gate,
threshold or default was changed.

## R10 CAVEAT — selection bias in the reflection negative set (found by review, 2026-08-15)
**This qualifies R10's headline and must travel with it.** The reflection negatives were sampled from
`src/octopus_clips_verified/*/Right_Left_*.mp4` — clips the *extraction pipeline* selected, and that
pipeline fires only when the CLIP detector marks >50% of a 20 s window as visible at p>=0.6. So the
reflection negatives are, by construction, **enriched for frames the detector got wrong**. Both arms
are scored on the identical frames, but the *set itself* was chosen by a process that used one of the
two arms. The bias runs **against the detector and in favour of mask area** — i.e. in the direction of
R10's result.

I had this backwards in my own framing (I worried the pool would flatter the detector; it flatters the
segmenter). Consequences:
- R10's paired dAUC **+0.1263 [+0.056, +0.198] is an upper bound**, not an unbiased estimate, on the
  mask-area advantage over the detector on reflections. The *sign* is well supported (the detector
  fires on only 32% of these frames at p>=0.6, so the set is not purely its own false positives), but
  the magnitude is inflated by an unquantified amount.
- An unbiased version needs negatives drawn **detector-independently** — uniform random timestamps
  from whole source videos via input-seek, the way `src/harvest_stream.py` probes — not frames from
  clips the extractor already chose.
- The same latent bias sits in R9's REFL-34 segmenter-only numbers, though there it has no head-to-head
  to distort: it makes the reflection set *harder-than-random* for the detector and roughly
  representative for the segmenter.
- Nothing here is withdrawn; the claim is narrowed to "mask area is the better reflection gate, with the
  effect size an upper bound pending detector-independent sampling".

## R8/R9 CAVEAT — every presence number resting on the 19 empty-tank frames inherits the one-video problem
This includes **R8's fusion presence result** (AUC 0.794 -> 0.9685 ema / 0.9495 flow). Those negatives
are the same 19 frames from 2 recordings, so the fusion presence gain is a one-video observation and
cannot carry a CI either. The fusion *mask* results (IoU 0.642 -> 0.547 / 0.511) are unaffected: they
are computed on 122 positives across 5 held-out videos. Re-testing the fusion presence claim on a
properly-powered empty-frame set is a deliverable of the next cycle.

**Paper action taken:** the .tex now states the $19$ empty frames come from two recordings (18 from
one), reports $0.794$ descriptively, and attaches no confidence interval to it.

## R11 — LEAKAGE AUDIT of the headline segmentation model (2026-08-15): PASSES
Prompted by the discovery that all 5 SEG-TEST holdout videos appear in thin768's dataset manifest
(493 frames), which would invalidate every published IoU if those frames had been trained on.
**They were not.** Verified three independent ways, not asserted:
1. The training command (`/tmp/modal_train_thin768.log`) carries `--holdout-videos /data/holdout.txt`
   and the run prints `forced holdout: 5/5 holdout videos present -> excluded from train`.
2. The split was re-derived from scratch (manifest + `--sources human` filter + `RandomState(42)`
   shuffle + `val_frac 0.2` + forced holdout) and reproduces the logged frame counts **exactly**:
   train 3450 / val 1693. No holdout video appears in the training partition.
3. Presence in the *dataset* manifest is not presence in *training* — the manifest lists all pairs and
   the trainer partitions them by source video.
**Conclusion: the headline IoU 0.6415 / 0.7193 and everything derived from it are leak-free.**

**Paper action taken (2026-08-15):** Sec. III-F of the .tex states that all five SEG-TEST videos
appear in the dataset manifest but are excluded from training by the forced-holdout flag, that the
split was re-derived and reproduces the logged partition exactly (3450 train / 1693 val), and that
the audit also found and fixed a logging bug printing wrong video counts for a correct split.

Two by-products:
- **thin768's true training set is 142 source videos** (not 183 = the dataset, and not 147 = the logged
  figure). Enumerated to `data/thin768_train_videos.json` — this is the exclusion list any future
  negative set must be filtered against, since empty-tank negatives fall squarely in its domain.
- **Logging bug found and fixed** in `src/train_segmenter.py`: the split line printed
  `len(vids)-n_val / n_val`, which ignores videos added to val by `--holdout-videos`, so thin768 logged
  "train 147 / val 36" for a split that was actually 142/41. Frame counts were always correct, so the
  printed numbers were mutually inconsistent — which is precisely what made this audit look like a leak
  at first glance. Now prints the actual partition sizes.

## R8-FINAL — threshold sweep resolves the calibration confound; the negative HOLDS (2026-08-15)
R8 compared fusion arms at the shipped threshold 0.5. That is not a fair comparison: a per-pixel
median over warped neighbours suppresses any pixel not confidently octopus in most frames, so the fused
probability map is systematically shrunk relative to a single-frame map, and a fixed 0.5 handicaps it.
Publishing a negative on that basis would have been wrong. `src/fusion_threshold_sweep.py` caches one
probability map per frame per mode and sweeps the binarisation threshold, scoring **each arm at its own
best operating point**. SEG-TEST, 122 positives / 5 held-out videos / 19 empty-tank negatives:

| mode | best IoU | @ t | best presence AUC | @ t | best FP@recall .90 | @ t |
|---|---|---|---|---|---|---|
| **none** (single frame) | **0.6552** | 0.80 | 0.8192 | 0.80 | 0.316 | 0.10 |
| flow ±2 (DIS-warped median) | 0.5109 | 0.50 | 0.9521 | 0.70 | 0.105 | 0.25 |
| **median ±2 (unwarped CONTROL)** | 0.5527 | 0.55 | 0.9629 | 0.70 | **0.000** | 0.25 |
| ema α=0.45 (deployed config) | 0.5866 | 0.40 | **0.9763** | 0.60 | **0.000** | 0.50 |

**1. The mask negative HOLDS.** Best-vs-best, the strongest fusion arm still loses to single frame:
0.5866 vs 0.6552 = **−0.0686**. The confound was real (it narrowed the gap from −0.094 at t=0.5 to
−0.069 best-vs-best) but does not change the conclusion. Test-time temporal fusion does not fix
mislocalization; it degrades mask fidelity. The paper's limitation must read "**test-time** temporal
fusion does not fix mislocalization", and the claim "a temporal student is the real lever" loses its
cheap supporting evidence (a temporal *trained* student remains untested).

**2. MECHANISM ESTABLISHED — optical flow is not the mechanism, and is actively harmful.** The
unwarped control beats flow on **both** metrics (IoU 0.5527 vs 0.5109; AUC 0.9629 vs 0.9521). Plain
temporal averaging supplies the entire benefit; motion compensation subtracts from it, presumably
because DIS flow is unreliable on a deforming, low-texture animal in dim IR and warping errors smear
the map. Without this control the natural write-up would have been "optical-flow fusion improves the
presence gate" — true in isolation and wrong about why. **EMA — the cheapest arm, already deployed and
requiring no flow computation — is the best of the three.**

**3. The shipped binarisation threshold is mildly suboptimal.** Single-frame peaks at t=0.80
(IoU 0.6552) versus 0.6415 at the shipped 0.5: **+0.0137 IoU for free**, no retraining. Not yet applied
— changing it is a pipeline default and needs its own before/after on SKEL-50, since the skeleton
stage consumes these masks and a thinner mask at t=0.8 may cost arms.

**4. The presence gain survives best-vs-best** (0.8192 → 0.9763, +0.157) **but is measured on the
19 empty-tank frames = 2 source videos**, so it inherits the one-video problem and carries no CI.
Re-testing it on EMPTY-V2 is the pending deliverable.

**Paper action taken (2026-08-15).** Sec. V of the .tex now carries "A negative that needed a
control: test-time temporal fusion" — best-vs-best 0.5866 (ema) vs 0.6552 (single frame), the
unwarped-median control beating flow on both IoU (0.5527 vs 0.5109) and presence AUC (0.9629 vs
0.9521), and the explicit statement that the fusion presence gain is NOT headlined because it rests
on the same 19 two-recording empty frames. The limitation now reads "**test-time** temporal fusion
does not fix mislocalisation; a temporally *trained* student remains untested". Cut for space (still
true, still logged here): the +0.0137 IoU available at threshold 0.80 — noted in the .tex header.

## R12 — Kinematics × behaviour cross-validation, recomputed with video-level statistics (2026-08-15)
The paper's headline cross-validation (skeleton kinematics agree with the VLM's behaviour labels) rested
on **n=40 clips with crawling at n=2**, pooled at clip level — pseudo-replication, since several clips
come from one recording. Recomputed properly.

**Sample.** `src/kinematics_sample.py` drew a video-spread stratified sample: **146 clips / 66 distinct
source videos**, ~25 per behaviour class, ≤2 clips per video. Run as 2 shards
(`batch_skeleton_motion.py --shard i/n`, isolated outputs), merged by `src/merge_shards.py`, which
refuses to pool mixed configs — all 147 records carry one stamp:
`{ckpt: octo_seg_thin768_lraspp.pt, fps: 3.0, refine: false, sha: f456768}`.
**Statistics** (`src/kinematics_stats.py`): aggregated to one value per (video, class) before testing,
Kruskal–Wallis + ε², Holm-corrected Mann–Whitney for resting-vs-each, cluster-bootstrap CIs by video.
The two signals are independent: the skeleton pipeline never sees the behaviour label.

### RAW arm-tip speed (px/s), median [CI95 by video]
| behaviour | median | CI95 | videos |
|---|---|---|---|
| Resting / stationary | 53.05 | [31.6, 62.9] | 24 |
| Human / enrichment interaction | 90.22 | [69.0, 134.8] | 24 |
| Crawling | 91.11 | [79.4, 114.8] | 19 |
| Swimming / jetting | 107.29 | [92.2, 124.4] | 15 |
| Exploration / manipulation | 112.40 | [90.6, 132.0] | 25 |
| Reaching out of water | 141.05 | [127.8, 166.9] | 25 |

**Kruskal–Wallis H=33.18, p=3.5e-06, ε²=0.224** (N=132 video-class units, k=6). All five
resting-vs-X contrasts significant after Holm correction (p_holm 0.0086 → 5e-05).

### SCALE-INVARIANT speed (body-lengths/s = speed ÷ arm-spread)
Resting 0.17 [0.1,0.2] · Human 0.26 [0.2,0.5] · Reaching 0.31 [0.2,0.3] · Exploration 0.36 [0.3,0.5] ·
Crawling 0.36 [0.3,0.4] · Swimming 0.37 [0.3,0.4].
**H=21.40, p=6.8e-04, ε²=0.130**; all five contrasts still significant after Holm.

**PRE-REGISTERED KILL CRITERION (p>0.05 or ε²<0.06) NOT MET — the result stands** and is now
properly powered. It also survives in scale-invariant units, which matters because raw px/s is
confounded by apparent size (distance from camera); normalising by arm-spread removes that.

**Nuance worth reporting: `reaching out of water` is the FASTEST in raw px/s but only 3rd in
body-lengths/s.** Reaching is performed with an extended body, so a large part of its raw tip speed is
extended posture rather than faster motion. Reporting only raw px/s would have overstated it. Swimming
and crawling, by contrast, rise in the normalised ranking.

Versus the old n=40 figure (resting 63 → reaching 159 px/s): same direction, more conservative
magnitudes (53 → 141) — the small-sample version was mildly optimistic, not wrong.

CAVEAT: behaviour labels come from the VLM structured extractor, whose reliability study (VLM-250) is
still BLOCKED on a revoked OpenRouter key. This validates that kinematics track the labels, not that
the labels are correct. Speeds are px/s in crop space (no px→cm calibration).

**Paper action taken (2026-08-15):** R12 replaced the n=40 cross-validation everywhere in the .tex
(abstract, contribution 4, Sec. VI, Fig. 5, limitations). Fig. 5 is now a two-panel figure generated
by `OCEANS_2026/make_figures.py` **through `src/kinematics_stats.collect` on
`data/skeleton_motion_study.json`** — the same loader that produced `data/kinematics_stats.json`, so
the plotted medians are the published medians by construction; it plots per-(video,class) medians,
not clips. The old 63→159 px/s figures are cited in the paper only as the earlier, less conservative
clip-pooled estimate.

## R13 — EMPTY-V2: the presence benchmark repaired, and 0.794 was WRONG as well as under-powered (2026-08-15)
The paper's presence AUC of 0.794 came from 19 empty frames drawn from **2 recordings (18 from one)**.
EMPTY-V2 replaces it with a properly-powered, **detector-independent**, verified set.

**Construction** (`src/empty_negatives.py`). Frames are grabbed at **uniform random timestamps from
whole server videos** by input-seek — never from clips the extractor selected, because extracted clips
exist only where the CLIP detector fired, which enriches the set with that detector's own false
positives (the bias now recorded against R10). Excluded: thin768's 132 training sessions and the
detector's 32 sessions, matched on `(date, HHMM)`. All 120 frames reviewed at full resolution before
scoring: **8 (6.7%) unmistakably contain the animal, 7 more ambiguous (12.5% total)**; ambiguous frames
are excluded rather than assumed empty. Result: **105 verified negatives / 53 source videos.**

**Two sampling defects were caught and fixed before any number was computed** — both would have
produced a confident, wrong result:
1. *Single-date concentration.* The first run drew all 40 frames from 20 recordings on ONE date and ONE
   camera, because one directory listing supplied every recording before the loop advanced —
   reproducing exactly the concentration defect this benchmark exists to fix. Fixed with a per-listing
   cap and round-robin.
2. *Domain mismatch.* The second run crawled both Nity collections, which are **two different physical
   setups** — the 2026-02 lab tank (the positives' domain) and a 2025-09 collection in a different room
   with a different tank. Separating "2026 tank containing an octopus" from "2025 room containing none"
   would have measured SCENE DIFFERENCE and returned a flatteringly high AUC for the wrong reason.
   Fixed by matching the collection; the cross-setup frames are kept separately
   (`data/empty_negatives_crossdomain/`) as a distinct question (FP in an unseen environment).

### Results — thin768, threshold 0.5, CIs cluster-bootstrapped by source video
| negative set | n | videos | AUC | CI95 | FP@R.90 | FP@R.80 | FP@area>=.01 |
|---|---|---|---|---|---|---|---|
| **EMPTY-V2 (empty frames, multi-video)** | 105 | **53** | **0.9170** | [0.839, 0.962] | 0.171 | 0.086 | 0.143 |
| reflection REFL-34 | 34 | 27 | 0.9214 | [0.826, 0.966] | 0.235 | 0.118 | 0.176 |
| old SEG-TEST empty-tank | 19 | **2** | *descriptive only* | — | — | — | — |

**1. The published 0.794 was not merely under-powered — it was PESSIMISTIC.** Properly measured across
53 recordings the model separates empty frames from present frames at **0.917**, not 0.794. The old
figure was dominated by a single unusually hard recording. The paper's presence claim is stronger than
what it currently reports, and can now carry a confidence interval.

**2. NULL RESULT, and it settles the question I withdrew in R9.** Empty-frame AUC 0.9170 and reflection
AUC 0.9214 are statistically indistinguishable (CIs almost entirely overlapping). There is **no
measurable difference between the two failure modes** — neither "reflections are the dominant problem"
(the paper's original framing) nor "the failure mode is backwards" (my withdrawn claim) is supported.
Recording the null explicitly so neither framing returns.

**3. BUG FIXED in my own earlier statistics.** `areas_from_cache` set each positive's `video` to its
image FILENAME, so the cluster bootstrap treated 122 positives as 122 independent recordings when they
come from 5. That understates clustering and yields CIs that are too narrow *in the flattering
direction*. R9's reflection CI [0.871, 0.964] is corrected to **[0.826, 0.966]**; the point estimate
0.9214 is unchanged. Fixed in `src/eval_reflection_presence.py`. **Every cluster bootstrap must group
BOTH arms by true source video — grouping only the negatives is not enough.**

PENDING: re-test R8's fusion presence gain (0.794 -> 0.9685 ema) on EMPTY-V2 — it rests on the same
19 two-recording frames. Needs neighbour frames per negative, i.e. another server pass.
CAVEAT: labels are AI-verified, not human-verified — PAPER_NOTES only until human confirmation.

## R13-FINAL — EMPTY-V2 HUMAN-VERIFIED (2026-08-15). Now citable in the paper.
All **120/120** frames confirmed by a human via `ui/verify_negatives.py` (port 8020). Human labels are
stored in a separate `human` field; the model's labels stay in `review`, so agreement is measurable.

### Model-vs-human agreement: 102/120 = 85.0%
| my label | human said | n | direction |
|---|---|---|---|
| empty | **octopus present** | **9** | **contamination — I would have scored 9 animal-containing frames as negatives** |
| octopus present | empty | 2 | I over-called |
| ambiguous | octopus present | 6 | resolved |
| ambiguous | empty | 1 | resolved |

Final human set: **99 empty / 21 present / 0 ambiguous**. My proposed negative set of 105 was
**8.6% contaminated**. This is the third time in this project that an assumed-empty pool turned out to
contain the animal (166/232 in the 2026-06 hard-negative mining; 7-19% in the reflection pilot).
**An AI-verified negative set is not a substitute for a human one** — 85% agreement sounds high, but it
is the 15% that decides the number.

### Headline (HUMAN-verified, thin768, threshold 0.5, CI cluster-bootstrapped by source video)
| negative set | n | videos | AUC | CI95 | FP@R.90 | FP@area>=.01 |
|---|---|---|---|---|---|---|
| **EMPTY-V2 human-verified** | **99** | **53** | **0.9093** | **[0.833, 0.957]** | 0.182 | 0.152 |
| EMPTY-V2 as I had labelled it | 105 | 53 | 0.9170 | [0.839, 0.962] | 0.171 | 0.143 |
| old SEG-TEST empty-tank (paper's 0.794) | 19 | **2** | descriptive only | — | — | — |

**The paper's 0.794 should become 0.909 [0.833, 0.957] on 53 recordings.** Even after removing my
contamination the properly-powered figure is far above the published one, which was dominated by a
single hard recording.

### LESSON — contamination does not always deflate a metric; here it INFLATED it
Intuitively, octopus-containing frames scored as negatives should *hurt* the AUC. They did the
opposite (0.9170 -> 0.9093 when removed) because **7 of the 9 frames I missed also fell below the
deployed gate — the segmenter missed the animal in exactly the frames I did**. They therefore looked
like unusually clean negatives and flattered the score. When the reviewer and the model share a blind
spot, contamination masquerades as good performance. Do not assume label noise is conservative.

### Deployment statistic worth reporting (uniformly-sampled footage, not curated clips)
At the deployed gate (mask area >= 0.01): **11/21 (52%) of human-confirmed present frames fire**, and
**15/99 (15%) of confirmed empty frames fire**. Note these present frames are uniformly sampled, so
many show the animal small, dim or half-denned — a much harder recall test than SEG-TEST's curated
positives, and a more honest picture of what the gate does on raw footage.

### STILL AI-ONLY: the reflection set (R9 / R10)
`data/reflection_negatives/` (42 frames) has **0/42** human labels. Given 85% agreement and 8.6%
contamination on EMPTY-V2, R9's reflection AUC 0.9214 and R10's head-to-head dAUC +0.1263 must stay
**out of the paper** and be treated as provisional until the same pass is run on them.

## R14 — ALL presence results HUMAN-VERIFIED (2026-08-15). R9/R10/R13 are now citable.
Both staged negative sets fully human-labelled via `ui/verify_negatives.py`:
**EMPTY-V2 120/120** (83% model-human agreement) and **reflection 42/42** (88%).

### Final numbers — human-verified negatives, thin768 @ 0.5, CIs cluster-bootstrapped by source video
| negative set | n | videos | AUC | CI95 | FP@R.90 | FP@area>=.01 |
|---|---|---|---|---|---|---|
| **EMPTY-V2 (empty frames)** | 97 | 52 | **0.9070** | [0.826, 0.958] | 0.186 | 0.155 |
| **reflection** | 36 | 29 | **0.9064** | [0.806, 0.956] | 0.278 | 0.222 |
| old SEG-TEST empty-tank (paper's 0.794) | 19 | **2** | descriptive only | — | — | — |

**1. The paper's 0.794 becomes 0.907 [0.826, 0.958] across 52 recordings.** Human-verified,
detector-independent, multi-video. The old figure was a single-recording artifact.

**2. The NULL RESULT is now airtight: 0.9070 vs 0.9064.** Empty frames and reflections are equally
hard for the model — a near-exact tie on human labels. Neither "reflections are the dominant
false-positive source" (the paper's original framing) nor "the failure mode is backwards" (my
withdrawn claim) is supported. State the null; drop both framings.

**3. Head-to-head survives human verification** (human-verified reflection negatives, detector
training sessions excluded → 30 frames / 24 videos):
| arm | AUC | CI95 | FP@R.90 |
|---|---|---|---|
| mask area (**zero-shot** on this camera) | **0.9126** | [0.821, 0.962] | 0.267 |
| CLIP detector `p_visible` (**in-domain**, 1,519 Right_Left training frames) | 0.7989 | [0.598, 0.882] | 0.700 |

Paired **ΔAUC +0.1340, CI95 [+0.024, +0.308] — still excludes 0.** The wide CI (24 videos) is the
honest weakness; claim the ordering, not the magnitude. The R10 selection-bias caveat still applies:
these negatives come from extractor-selected clips, so the effect size remains an upper bound.

### My labelling erred in the FLATTERING direction on both sets
| set | my labels | human labels | shift |
|---|---|---|---|
| EMPTY-V2 | 0.9170 | 0.9070 | −0.0100 |
| reflection | 0.9214 | 0.9064 | −0.0150 |
Every AI-verified figure I produced was optimistic. The mechanism (documented in R13-FINAL) is a
**shared blind spot**: frames where I missed the animal are disproportionately frames where the
segmenter also missed it, so they masquerade as unusually clean negatives. **AI verification of a
negative set is systematically biased toward the model being evaluated** — it cannot substitute for
human labels, and the bias has a predictable sign.

### Bonus — intra-rater reliability, measured by accident
A UI bug (`first_unreviewed` wrapping to 0 on a completed set + the set dropdown resetting on reload)
sent a session intended for the reflection set back over EMPTY-V2, re-labelling all 120 frames. That
produced an unintended **test-retest**: the same human, the same frames, twice, independently.
**118/120 = 98.3% self-consistency** (differing only on `empty_0014`, `empty_0015`).
This bounds label noise: the human agrees with themselves at 98.3% while agreeing with the model at
83%, so **the model-human gap is model error, not rater instability**. Both UI bugs are fixed.

## R15 — VLM-250: the behaviour labels are only MODERATELY self-consistent (2026-08-15)
UNBLOCKED by a fresh API key. `src/vlm_reliability.py --run` re-ran the structured extractor over the
frozen 250-clip / 140-video sample using a **disjoint set of input frames** (detector ranks
N_KEEP..2*N_KEEP instead of the top N_KEEP). 250/250 succeeded, **$0.165**. Analysis:
`src/vlm_reliability_stats.py` → `data/vlm_reliability_stats.json`.

**What this measures: frame-sampling sensitivity — does a label survive being shown different clear
frames from the same clip. It is CONSISTENCY, not accuracy.** Consistency upper-bounds accuracy; a
perfectly consistent extractor can still be consistently wrong. Never write "accurate" for these.

| field | raw agreement | Cohen's κ | κ CI95 (by video) |
|---|---|---|---|
| **behavior (7-class)** | 0.652 | **0.552** | [0.472, 0.624] |
| posture | 0.684 | 0.510 | [0.423, 0.594] |
| location | 0.632 | 0.511 | [0.426, 0.591] |
| context | 0.756 | 0.585 | [0.482, 0.676] |
| activity | 0.752 | 0.592 | [0.507, 0.675] |
| present | 0.848 | 0.413 | [0.229, 0.579] |
| body_color | 0.868 | 0.744 | [0.658, 0.819] |
| color_or_texture_change | 1.000 | 1.000 | — **ARTIFACT, see below** |

**1. THE HEADLINE CAVEAT. Every behavioural result in this paper — activity budget, circadian
profile, human-presence stimulus response, kinematics × behaviour — is grouped by a label with
κ ≈ 0.55 ("moderate" on Landis–Koch).** Change which frames the model sees and roughly a third of
behaviour labels change. This must appear in the paper's limitations; it was previously unmeasured
and simply assumed.

**2. Rare classes are the least stable — and they are exactly the classes carrying the fewest videos:**
Exploration 72.4% · Resting 71.2% · Human-interaction 60.7% · Crawling 53.8% · Reaching 47.6% ·
**Swimming/jetting 42.9%**. So the classes with the widest kinematic CIs also have the least reliable
labels; treat per-class claims about swimming/reaching with corresponding caution.

**3. This STRENGTHENS R12 rather than undermining it.** Label noise that is independent of kinematics
causes regression dilution — it biases a group-difference test **toward the null**. Finding
KW p=3.5e-06, ε²=0.224 *despite* labels at κ=0.55 means the true separation is likely larger than
measured, not smaller. CAVEAT: this holds only if the mislabelling is independent of motion; if fast
clips are preferentially labelled "swimming", noise could inflate instead. Not currently testable.

**4. `color_or_texture_change` is a DEAD FIELD, and its κ=1.000 is an artifact.** Its value is
**100% determined by the greyscale gate** (IR clips forced to `uncertain` = 151; colour clips `none`
= 99) — the perfect agreement measures the determinism of a preprocessing rule, not model judgement.
More damning: across **99 colour clips the model reported a colour/texture change exactly zero times**.
The field carries no information and should be dropped from the schema or redesigned. The stats script
now detects and flags gate-determined fields rather than reporting them as excellent reliability.

**5. `present` has high raw agreement (84.8%) but low κ (0.413)** because the class is imbalanced
(~87% present). Report κ, not raw agreement, for this field. Abstention (`uncertain` behaviour) ran at
17.6% in condition B.
