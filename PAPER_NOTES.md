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
