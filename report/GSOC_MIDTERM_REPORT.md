# GSoC Mid-Term Evaluation Report
**Project:** Cephalopod Behavioral Analysis — detecting & captioning the octopus "Nity" (*O. vulgaris*) in aquarium footage
**Organization:** Catrobat
**Contributor:** Siddharth Raj
**Period:** 2026-05-16 → 2026-07-08

---

## 1. Goal

Build an end-to-end pipeline that ingests live aquarium camera footage, detects when the octopus is present, extracts short behavioral clips, and automatically captions each clip with a natural-language description and a behavior (ethogram) label — producing a curated, labeled dataset of octopus behavior.

---

## 2. Pipeline Overview

```
Remote footage  →  Motion gate  →  Octopus detector  →  20s clip extraction  →  VLM captioning  →  Human review
 (server)          (absolute       (CLIP + MLP probe)     (ffmpeg)              (+ ethogram label)   (UI)
                    changed-pixel)
```

Everything is consolidated into a clean, self-contained package under `src/`.

---

## 3. What Was Accomplished

### Detection — CLIP + MLP probe (working approach)
- Built an octopus presence classifier: **CLIP ViT-B/32 (frozen) → MLP probe (512→256→64→2)**, classifying each frame as `visible` vs `hidden`.
- **Key finding:** an earlier attempt at zero-shot CLIP scoring was found unreliable (camera-bias baselines) and abandoned; the trained probe replaced it.
- **Critical fix — preprocessing:** switched from CLIP's default center-crop (which discards 33–44% of a 16:9 frame) to **letterbox padding**. Aspect-ratio mismatch — not model architecture — was the root cause of poor field performance.
- **Hard-negative mining with label verification:** mined the model's confident false positives, then *independently verified* them rather than trusting the labels. Of 232 candidate frames, 166 actually contained the octopus (model was right) and only 66 were genuine hard negatives. Folding in verified negatives cut the hidden false-positive rate from 24% → 3% while holding visible recall ~0.97.
- Current production model: `clip_mlp_hardneg_v2.pt` (~96.8% acc on a hardened test set).
- Training dataset assembled: **~3,970 visible / ~5,626 hidden** labeled frames.

### Motion detection
- Found and fixed a significant bug: the original per-video **normalized** motion gate let static videos through (a flickering lamp normalizes up to "motion").
- Replaced with an **absolute changed-pixel-fraction** method that also masks the ticking-clock timestamp region so it isn't counted as motion.

### Clip extraction
- Consolidated the whole extraction flow into one script (`extract_octopus_clips.py`): per video it runs the octopus detector + absolute motion gate, slides a non-overlapping 20s window, and keeps windows that are octopus-visible AND moving. This replaced a fragile 3-notebook chain (extract → verify → audit).
- Produced **847 candidate behavioral clips** in the curated index, with per-clip metadata (camera, source timeline, scores).

### Captioning (VLM)
- Built two captioners that write both a one-sentence `caption` and a 7-class `ethogram_label` per clip:
  - **Qwen3-VL-30B** teacher on Colab/GPU (vLLM).
  - **Qwen3-VL-235B** via the OpenRouter API — runs locally, no GPU.
- Input quality improvements for the dim IR footage: CLAHE contrast enhancement, higher resolution, best-frame selection, and skipping empty clips.
- **Key finding (data quality):** running the 235B model over all 847 clips revealed **534 (63%) had no octopus** — the extractor massively over-extracts, mostly `Right_Left` reflections the detector fires on. Actionable takeaway: drop the `Right_Left` camera and use the VLM as a stronger presence filter than the detector.

### Ethogram
- Reduced the behavior sheet from 19 → **7 classes** (6 originals had zero clips), with a `maps_from` mapping preserving the originals. All labeling/training now uses the 7-class sheet.

### Distillation (student models)
- **Caption student:** QLoRA fine-tune of Qwen2.5-VL-3B distilling the teacher captions.
- **Behavior classifier:** attempted (frozen CLIP feats → MLP) but **failed** (below majority-class baseline) — an informative negative result: behavior classification needs temporal/motion features, not static pooled ones.

### Tooling & infrastructure
- Five FastAPI review/labeling UIs (caption review, blind A/B labeling, base-vs-LoRA comparison, hard-negative review).
- Security hardening: removed hardcoded server credentials from code, moved to a gitignored `.env` loaded via `server_creds.py`.
- Packaged a clean, portable `src/` pipeline plus a shareable deliverable branch (`octopus-pipeline-src`).

---

## 4. Key Results (headline numbers)

| Metric | Result |
|---|---|
| Octopus detector accuracy | ~96.8% (hardened test set) |
| Hidden false-positive rate | 24% → **3%** after verified hard negatives |
| Visible recall | ~0.97 |
| Labeled training frames | ~3,970 visible / ~5,626 hidden |
| Candidate clips extracted | 847 |
| Ethogram classes | 7 (from 19) |

---

## 5. Lessons Learned
- **Preprocessing beat architecture** — letterboxing fixed field performance without changing the model.
- **Always verify labels before training** — 166 of 232 "negatives" were actually positives.
- **Absolute > normalized** for motion gating on near-static footage.
- **The VLM is a better presence filter than the detector** — 63% of "verified" clips were empty, revealing over-extraction.

---

## 6. Remaining Work (second half)
- Drop `Right_Left` and re-run extraction using the VLM as a presence filter to produce a clean clip set.
- Finalize and evaluate the LoRA caption student against the teacher.
- Explore temporal/motion features for behavior classification (the static-feature approach failed).
- Optionally add the 166 confirmed back-angle frames as `visible` training data.

---

*Repository note: despite the `sentiment-analysis` repo name, this is the cephalopod video-analysis project.*
