# OCEANS 2026 — paper plan

Working document. **Stage 1 (now): put everything in.** Stage 2: crop to the hard limit.

## Venue constraints (hard)

- **4–6 pages including figures**, references excluded. No padding.
- U.S. letter, official IEEE conference template. We already use `\documentclass[conference]{IEEEtran}` ✅
- Remove the copyright string from the template footer before submitting.
- Online IEEE copyright transfer + conference registration required before upload.
- Oral vs poster is assigned by the committee on program balance, not quality.

**Current state:** `octopus_behaviour_pipeline_v2.tex` = **9 pages**, bibliography only ~30 lines, so
the body must lose ~35%. This is a restructure, not a trim.

## Framing decision — SETTLED

Rejected:
- ~~(A) A measured cascade that makes continuous monitoring tractable, with released data.~~ — reads as
  cost/throughput engineering. The funnel is *why the pipeline is possible*, not what the paper is about.
- ~~(B) A labelled dataset + benchmarks, and the pipeline that produced it.~~ — makes the pipeline read
  as plumbing and the science as a by-product.

**CHOSEN: a PIPELINE that turns raw aquarium video into BEHAVIOURAL UNDERSTANDING of the animal.**

The arc, and every section serves it:

1. **Problem** — continuous cephalopod behavioural monitoring is what a facility actually wants, and it
   is out of reach: manual scoring does not scale, and a VLM on everything is 372–1,030 h of wall-clock
   per animal and cannot run on-site.
2. **Pipeline** — a cascade of small local models with the 235B VLM as an *offline teacher only*. The
   funnel (6.7% of footage decoded) is the enabling fact, stated once, not the thesis.
3. **What it extracts** — presence, octopus silhouette/posture, a 6-class ethogram, one-sentence
   captions, arm kinematics.
4. **What we LEARN from it** ← the payoff, and the reason the pipeline is worth building: circadian
   structure, stimulus response, activity budget. This is the section that proves the system produces
   understanding rather than numbers.
5. **Released data + frozen benchmarks** so others can reproduce and extend.
6. **Where the understanding goes next** — limb-aware ethograms from the skeleton work.

**Consequence for the cuts:** the behavioural-findings section is NO LONGER the cut candidate — it is
essential. The "what does not work" catalogue drops from a section to a paragraph, and the detailed
ablations (full rung matrix, 3-way resolution isolation, seed-noise discipline) move to the companion.

**Overflow → arXiv companion / tech report** (full `PAPER_NOTES.md` R1–R35, all ablations, negative
results in full). The paper cites it.

---

# EVERYTHING AVAILABLE (stage 1 — the full inventory to cut from)

## §I Introduction + contributions

**The problem is tractability, not accuracy.**

| quantity | value |
|---|---|
| footage in the harvest ledger (one animal, colour cameras) | **892.5 h** |
| if sliced into 20 s clips | 160,650 clips |
| 5-pass VLM cost at $0.0007/clip/pass | $562 |
| 5-pass VLM **wall-clock** at the measured 13–36 clips/min | **372–1,030 h** |
| and the server also holds Heidi 155 d, a 2024 *O. vulgaris* 122 d, Maya 48 d, Eledone 1 d | ~3× more, untouched |

Cost is not the blocker — **throughput and locality are**. It cannot run on-site.

Claimed contributions:
1. A cascade whose every gate has a measured justification, several of which reversed the obvious choice.
2. Distillation of a 235B teacher into small models that run locally (Apple Silicon, no GPU server).
3. Released datasets + frozen video-level benchmarks.
4. A catalogue of what does **not** work.

## §II Related work

- HideAndSeg (nearest neighbour: octopus segmentation) — where we differ and where we do not beat it.
- Open-vocabulary detection (OWLv2), promptable segmentation (SAM2, GroundingDINO) as *teachers not detectors*.
- VLM captioning of animal behaviour; ethogram automation.
- Distillation / small-model deployment.
- **Trim target: 0.4 pp.**

## §III The cascade — gates and the measurements that set them  ← CORE, currently missing

### The funnel (measured on 1,769 videos)

| stage | outcome |
|---|---|
| videos in ledger | 1,769 (892.5 h) |
| probe-first (~10 seek frames/video) | **1,054 videos (60%) discarded without a full decode** |
| footage actually decoded | **59.9 h = 6.7%** |
| clips surviving extraction, sent to the VLM | 5,222 |
| of those: no animal | **42.7%** |
| of those: stationary | 16.7% |
| **behaviourally active** | **40.6%** |

### Gate justifications — each with the counterfactual

| gate | measurement | what it prevented |
|---|---|---|
| hard-negative mining | of 232 confident detector FPs, **166 actually contained the animal**, only 66 were true negatives | training on 232 "negatives" that were 72% animal — teaching the detector to call real octopus empty |
| camera-directional filter | `Right_Left` PRESENT wrong **45%**, ABSENT **100%** right; `Right_Top` ABSENT only **55%** right | blanket-excluding `Right_Left` and discarding the corpus's most reliable hard negatives |
| VLM presence gate | 235B returns "not present" on **63%** of detector-"verified" clips | trusting the trained detector over the VLM |
| soft targets | human agreement tracks vote margin: **0.726** unanimous / **0.864** at 4-of-5 / **0.426** at ≤3/5 | argmax labels teaching false confidence on 3–2 clips |
| motion gate (absolute, not normalised) | per-video normalised motion passes static videos — a flickering lamp normalises up to "motion" | a gate that fires on IR lamp flicker |
| truncated-clip guard | 147/6,945 clips (2.1%) are <15 s; one 3.6 MB clip is 0.49 s | a file-**size** check passing them; 8 reached the dataset, one in test labelled `Exploration` |

**Four of these reversed the choice we would otherwise have made.** That is the section's argument.

### 5-pass ensemble design
Interleaved-uniform sampling: pass *p* offsets by `(p-1)·step/n_passes`, so the 5 passes are disjoint
and their union tiles all 50 dense frames. Measured value: the majority vote differs from a single
pass-1 label on **705 clips (13.5%)**; 63.8% unanimous, 36.2% split.

## §IV Datasets and frozen benchmarks  ← promote

| artifact | n | labelled by | notes |
|---|---|---|---|
| ensemble behaviour labels | **5,222 clips** | 235B ×5 passes | presence + 7-class, **full vote distribution retained** |
| **captions** | **5,160 clips × 5 passes ≈ 25,000 instances** (4,678 real sentences) | 235B | 5 *disjoint* frame samplings per clip → caption-stability measurable |
| ethogram dataset v1 (frozen) | 4,665 clips / 204 videos | derived | video-level splits, soft targets, per-clip reliability weights |
| human behaviour labels | **456** | HUMAN | blind-by-construction UI; all `assisted` → agreement |
| human click-to-SAM2 masks | **513** (412 pos + 87 neg, ~35 videos) | HUMAN | |
| auto masks | 10,725 pairs | GD+SAM2 teacher | |
| SEG-TEST | 122 frames / 5 held-out videos + 19 empty-tank negatives | HUMAN masks | |
| SKEL-50 | 50 frames / 20 videos | HUMAN keypoints | |
| REFL-24 | reflection-rejection set | HUMAN | |
| structured behaviour records | 3,205 clips × 9 fields | 235B | posture, activity, location, context, colour |
| harvest ledger | 1,769 videos | — | per-video coverage: probe points, scanned/unscanned seconds, discard reason |

Rules stated in the paper: splits **by source video, never by clip**; frozen sets never regenerated
to suit a result; negatives of different kinds never pooled; holdout videos excluded from *every*
training source.

## §V-A Ethogram classifier  ← NEW, the biggest addition

6 classes in one head — `No octopus` is a **class**, not a pre-filter, because that is how it deploys.
Merges: `Crawling`+`Swimming/jetting` → `Locomotion` (teacher cannot separate them; 5/40 errors were
Swimming→Crawling, all one direction). Dropped `Colour change/defensive` (1–3 clips corpus-wide).

### Progression

| model | test macro-F1 | accuracy |
|---|---|---|
| majority class | 0.1004 | 43.1% |
| pooled CLIP → linear (reproduces the *previous failed* classifier) | 0.4000 | — |
| CLIP + BiGRU | 0.5298 | — |
| 3-backbone ensemble | 0.6172 | 71.4% |
| **5-member ensemble (+ mask crops)** | **0.6648** | **75.4%** |
| fusion (val-selected) | 0.6514 | — |

### The representation finding (+0.087, the largest single lever)

| backbone | rung 1 | rung 2 | rung 3 |
|---|---|---|---|
| CLIP ViT-B/32 | 0.5096 | 0.5368 | 0.5298 |
| DINOv2-base | **0.6006** | 0.5772 | 0.5781 |
| VideoMAE-base | 0.5883 | **0.6057** | 0.6016 |

Robust to the selection rule: the **worst** rung of either new backbone beats CLIP's **best**, no overlap.
Not clearly about *time* — DINOv2 (an image model) is within noise of VideoMAE, whose best rung pools
over time and discards order. Claim: "a better self-supervised representation", not "video sees motion".

### Animal pixel resolution is the operative variable (3-way isolation)

| variant | frame retained | animal size in a 224² input | val | test |
|---|---|---|---|---|
| centre-crop (processor default) | 43% | preserved | 0.5784 | 0.5772 |
| letterbox | **100%** | **shrunk** | 0.5738 | 0.5895 |
| **mask-guided crop** (~3.1× magnification) | animal only | **maximised** | **0.6183** | **0.6465** |

Letterbox keeps every pixel but shrinks the animal; centre-crop discards half the frame but preserves
animal size; **they tie**. Only magnifying the animal helps. Median mask bbox longest side is 0.216 of
the frame, so backbones were seeing the animal at ~48×48 px inside 224².

### Per-class, 5-member ensemble (740 test clips / 34 videos)

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| No octopus | **0.926** | 0.897 | 0.911 | 319 |
| Resting / stationary | 0.710 | 0.597 | 0.648 | 119 |
| Exploration / manipulation | 0.662 | 0.667 | 0.664 | 141 |
| Locomotion (crawl/swim) | 0.644 | 0.731 | 0.685 | 52 |
| Reaching out of water | 0.571 | 0.754 | 0.650 | 69 |
| Human / enrichment interaction | 0.436 | 0.425 | 0.430 | 40 |

Macro precision 0.658 · macro recall 0.678 · weighted precision 0.761.

### Teacher gap (vs 456 human labels, 251-clip population)
teacher 72.5% / macro-F1 0.657 · single-CLIP student 60.6% / 0.492 · **3-backbone student 66.9% / 0.576**.
Disagreements: human backs teacher 49:19 → **36:22** after the representation work. Half the gap closed.
**Confound: the UI hint is the teacher's own verdict and all labels are `assisted`, so teacher-vs-human
is inflated and the true gap is smaller.**

## §V-B Segmentation student

- SEG-TEST: **IoU 0.6415 mean / 0.7193 median, area error ~1%**, presence AUC 0.794.
- The 0.85 IoU bar is unmet, but **area error ~1%** and everything downstream (presence, posture,
  masked motion) is area-based → **pixel-IoU is the wrong metric for this model's job**.
- What worked, leak-free: **256²→512² input** (at 256² the median octopus is ~40×40 px, tentacles
  1–2 px) + **Focal-Tversky loss** → 0.466 → 0.608; then 768² → 0.6415.
- **Volume beat purity:** 290 clean human frames → 0.505; 92%-auto blend of 3,450 frames → 0.608.
- Temporal fusion is a **trade**: EMA costs 0.10 IoU, buys **+0.17 presence AUC** (0.794 → 0.9685);
  optical flow worse on both, so plain smoothing is the mechanism.
- Teacher vs human (per-frame): teacher 0.374 vs student 0.6415; but where GroundingDINO clears its
  own 0.60 gate (21/122 frames) the **teacher wins 0.726 vs 0.657** — high-precision/low-recall teacher,
  uniformly competent student. Distillation turned a sparse high-quality signal into dense coverage.

## §V-C Caption student

Qwen3-VL-2B + LoRA r16/α32, 3,066 train / 392 val. Held-out val: base emb-sim 0.702 / rougeL 0.269 →
**LoRA 0.834 / 0.455**. Deployed 4-bit MLX, **1.7 GB, ~3 s/caption on a 16 GB Mac**, no CUDA.

## §VI What does not work  ← NEW

1. **Head capacity/architecture ≈ 0.** hidden {256,512,1024} × dropout {0.3,0.5}: val gain +0.0017 vs
   seed std 0.0040. **More capacity actively hurts** — 1024 is the worst config on both backbones.
   Test spread across the grid is 0.033 while val resolves 0.002.
2. **Upsampling ≈ loss weighting** (+0.0023 vs seed std 0.0026).
3. **Less balancing keeps winning.** `BALANCE=none` is best on test macro-F1 (0.6103), accuracy (0.704)
   *and* on the weakest class (`Human` 0.44 vs 0.37) — balancing hurt the class it was meant to protect.
   Not adopted: val ranks it worst, so taking it would be test-set selection.
4. **Feature-space augmentation is negative** — mixup {0.2,0.4}, noise {0.05,0.1}, and the combination
   all lose, monotonically with strength, with val and test agreeing.
5. **Teacher-label quality is NOT the segmentation ceiling** — RETRACTED hypothesis. The supporting
   evidence (0.49→0.70) was train leakage; clean held-out is flat: 0.494 → 0.508 → 0.506.
6. **OWLv2 has no viable operating point.** AUC 0.759 (so not uninformative), but at its own optimum it
   passes 30% of negatives while losing 32% of animals; any threshold keeping ≥95% of animals passes 70%.
7. **Mask geometry features:** +0.043 raw, but **all of it on videos the segmenter trained on**
   (+0.064 seen vs −0.025 unseen), and null on top of the crop. Not claimable.
8. **Motion channels bought +0.006** despite AUC 0.714 in isolation — real but redundant with appearance.
9. **Pooled-CLIP behaviour classifier failed** (45% acc, per-class F1 ≈ 0) — needs temporal/motion, or
   rather, as we later found, a better representation.

## §VII Skeleton, kinematics, and limb-usage  ← reframed as capability + released data + future work

- **Capability:** mask → Zhang–Suen thinning → Dijkstra arm paths → mantle/head/8-arm graph;
  per-node `detected|fitted|occluded` state so kinematics use only evidence-backed samples;
  optical-flow prior with tightened acceptance gates when the prior validates.
- **Released:** SKEL-50 (50 human-keypoint frames / 20 videos; **tip-F1 0.539**, precision 0.722,
  recall 0.502, 3.68 arms/frame) and per-clip kinematics (tip/mantle speed, arm-spread, occluded
  fraction) merged into the behaviour records.
- **Measured levers:** best-frame seeding **+3.67 median arms** (2.17→5.83) — the big one; thin-preserving
  mask prep +0.17 arms; tracking v2 cut teleport 14.85%→14.27% and exposed **occluded_frac 41.7%** (half
  of arm samples were evidence-free holds, now excluded).
- **Future work (the direction, not a result):** per-limb usage statistics → **limb-aware ethograms**,
  distinguishing "exploration with one arm" from whole-body exploration, which the 6-class sheet cannot
  express.
- **Caveats that travel:** speeds are px/s in crop space (no px→cm calibration); arms are
  silhouette-limited (~1.7 arms/frame below human GT).

## §VIII Behavioural findings  ← CUT CANDIDATE (goes to the companion)

3,083 present clips: activity budget 41% exploration / 33% resting / 14% human-interaction / 9%
reaching / 2% crawling / 1% swimming. Exposure-normalised circadian: ~1–5% overnight → **45% peak at
17:00**, 13:00–19:00 plateau, dawn bump 05–06 h. **Human presence nearly doubles motion (0.045→0.095)
and lifts arousal 0.46→0.68.** Colour: dark_red_brown 16% at baseline vs 6% during human interaction.
Caveats: `context="enrichment_object"` fires on 66% of clips (permanent tank toys) so it means "object
present", not "active enrichment"; presence gate ~66% dirty upstream, so *rate contrasts* are robust
but absolute levels shift after a detector retrain.

## §IX Limitations — state plainly

- **Single animal, single tank.** Everything is Nity. Heidi (155 d), Maya (48 d), a 2024 *vulgaris*
  (122 d) are on the server and untouched — cross-animal generalisation is unmeasured.
- **All 456 human labels are `assisted`** (the model's answer was on screen) → they measure
  **agreement, not accuracy**. The blind round on the 34 reserved test videos has not been run.
- **~6 val-based selections on 35 val videos**, and val/test disagreed on three of them (class
  balance, rung choice, fusion-vs-ensemble). Reported numbers are softer than their error bars suggest.
- Segmentation: IoU 0.6415 against a 0.85 target; IR (35% of corpus) unusable by the colour-trained model.
- `Human / enrichment interaction`: 40 test clips on 7 videos — its per-class figure is not reliable.
- Skeleton speeds uncalibrated (px/s, not cm/s).

## §X Conclusion + data availability

Datasets, frozen benchmarks, weights, and the companion report. Ethical statement (observational
footage of a captive animal, no intervention).

---

# STAGE 2 — cropping plan (revised for the pipeline→understanding framing)

Target 6.0 pages, 1 figure + 3 tables.

| § | section | pp | notes |
|---|---|---|---|
| I | Introduction + contributions | 0.6 | problem = continuous behavioural monitoring; funnel as the enabling fact in 2 sentences |
| II | Related work | 0.4 | HideAndSeg; teachers-not-detectors; ethogram automation |
| III | Pipeline and its gates | 1.3 | **Fig 1** cascade w/ survival numbers; **Tab 1** gates + the measurement that set each. Keep the 4 reversing gates only |
| IV | Local models | 1.3 | ethogram 0.6 (progression + per-class) · segmentation 0.4 (IoU + area-err + "wrong metric") · caption 0.3 (LoRA gain + 1.7 GB/3 s) |
| V | **What the pipeline reveals** | 1.2 | **the payoff.** activity budget, circadian (45% peak @17:00), human presence doubles motion 0.045→0.095 and lifts arousal 0.46→0.68, colour-by-context. WITH the caveats (enrichment_object 66%; rate contrasts robust, absolute levels not) |
| VI | Datasets + frozen benchmarks | 0.6 | **Tab 3** release inventory + the 4 protocol rules |
| VII | Skeleton → limb-aware ethograms (future) | 0.3 | capability + SKEL-50 + the direction |
| VIII | Limitations | 0.4 | single animal; assisted labels = agreement not accuracy; val/test disagreement; IR unusable |
| IX | Conclusion + data availability | 0.3 | |
| | **total** | **6.4** | trim 0.4 from II/IV in drafting |

**Demoted to a paragraph** (inside §IV): what does not work — head capacity ≈ 0 and *more capacity
hurts*; augmentation negative; teacher-label quality NOT the segmentation ceiling (retracted, leakage);
OWLv2 has no viable operating point. One sentence each, no table.

**Moved to the companion:** full rung matrix · 3-way resolution isolation · mask-geometry leakage
result · upsampling-vs-weighting · motion-channel redundancy · skeleton tracking ablations · temporal
fusion trade · 30B-vs-235B captions · teacher-vs-human mask comparison.

**Figure 1** must carry §III visually — cascade stages with survival counts — so §III can stay at 1.3 pp.

## Open decisions
1. ~~Drop behavioural findings?~~ **NO — it is the payoff section.** Settled.
2. Blind human round on the 34 reserved test videos — run it, or ship the limitation as written?
3. Companion venue: arXiv preprint, or tech report linked from the data release?
4. Trim the last 0.4 pp from §II or §IV during drafting.
