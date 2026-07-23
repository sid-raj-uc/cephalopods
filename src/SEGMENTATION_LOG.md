# Octopus Segmentation — Running Work Log

Detailed chronological log of the tiny-segmenter effort (teacher auto-labeling → student training).
Newest entries at the bottom. Numbers are honest, including the failures. See `SEGMENTATION_PLAN.md`
for the design and `AGENTS.md` for the durable summary.

Compute: GPU box **`amera-vllm-a100`** (A100-40GB, 10.32.0.7), reached by SSH from `amera-siddharth`
(CPU-only). Env: `~/seg-venv`. Dataset + weights live on the A100 during the run, pulled back at the end.

---

## 2026-07-22/23 — Session 1

### Phase 1a — balanced sampler (`src/sample_seg_clips.py`)
- Joined the 3,986 on-disk clips to behaviour labels in the index; dropped `octopus not present`
  (311) + Right_Left (reflections) + 33 not-in-index. Colour-first.
- Selected **1,824 present colour clips** (Right_Front 604 / Right_Back 540 / Right_Right 680),
  water-filled across behaviours. Manifest: `src/dataset_seg/sample_v1/sample_manifest.json`.

### Env setup on A100
- Installed ffmpeg + venv (torch cu124, transformers, opencv, `sam2` built `SAM2_BUILD_CUDA=0` — no nvcc).
- Warmed HF cache: GroundingDINO-tiny + SAM2.1-hiera-tiny. Setup script `scratchpad/a100_setup.sh`.
- Transfer: one-time rsync of the 1,824 clips (~10.5 GB) over the internal VPC to `~/seg_clips/`.

### Phase 1 — auto-labeling (`auto_segment.py`, GroundingDINO+SAM2 teacher)
- Sharded 3× by camera (parallel, ~9.5 GB GPU). ~13 s/clip.
- **Data-quality finding:** 504 of 731 Right_Right source clips are **0-byte on disk** (pre-existing
  extraction bug, not a transfer issue). auto_segment correctly rejected them as `no_frames`.
- Result: **1,104 clips accepted → 4,412 (image, mask) pairs** across **77 source videos**.
  Front 1,912 / Back 1,864 / Right 636. Mask area: median 2.9%, healthy (spot-checks clean).
  Rejected 243 low-confidence (reflections/ambiguous), which is the seed-conf gate working.

### Phase 2 — train tiny segmenter (`src/train_segmenter.py`)
- Split BY SOURCE VIDEO (62 train / 15 val), BCE+Dice, IoU@0.5. Aug: flip + brightness only (weak).
- **From-scratch TinyUNet sweep (256², 40 ep):**
  | base_ch | params | best val IoU |
  |---|---|---|
  | 8  | 0.121 M | 0.398 |
  | 16 | 0.483 M | 0.438 |
  | 32 | 1.927 M | 0.474 |
- **LR-ASPP MobileNetV3 (ImageNet-pretrained, 256², 60 ep):** train loss → 0.18, **val IoU 0.447**.

### Diagnosis (why v1 stalls at ~0.47, bar is 0.85)
- Cleaning big-GT masks (area>0.20) barely moved val (0.474→0.480) → **not** label noise.
- **Train IoU 0.684 / Val IoU 0.474** → a **generalization gap**, not underfitting.
- Failure mode on the 31% of val frames with IoU<0.3: GT is a normal ~3% octopus, model predicts a
  normal ~4% blob, **but in the wrong place** (only 10% predict empty) → *mislocation*.
- From-scratch U-Net and pretrained LR-ASPP hit the **same** ceiling → bottleneck is
  **data diversity + augmentation, not architecture**. Only 62 train videos; aug was flip+brightness only.
- **Decision:** highest-leverage cheap fix = strong spatial augmentation (affine/translate/scale) to
  teach position-invariance. Then, if short, add IR Right_Top (1,391 clips) for video diversity.
  (No `octopus_clips_auto` on this box; colour clips essentially exhausted.)

### Experiment: strong augmentation (affine/translate/scale + flips + color jitter + noise)
- U-Net ch32 + aug: **0.469** (was 0.474) — no change.
- LR-ASPP + aug: **0.473** (was 0.447) — +0.026, marginal.
- **Verdict: aug does NOT close the gap.** LR-ASPP still drives train loss to ~0.18 while val
  stays ~0.47 → confirms a genuine video-diversity generalization gap, not something aug fixes.

### Per-video val IoU (aug LR-ASPP) — diversity gap confirmed
- Range **0.00 → 0.82** across the 15 val videos, broad gradient (worst: 116-frame Right_Back @0.28;
  best 0.66–0.82). Not one pathological video → the model does ok on train-like videos, fails on
  dissimilar ones. Classic limited-diversity overfitting (only 62 train videos). macro 0.455 / micro 0.473.

### Decision: add IR Right_Top data (more video diversity + it's a real deployment camera)
- Colour clips are exhausted; the available diversity is IR Right_Top (1,391 clips). Right_Top is the
  BIGGEST deployment camera, so an IR-capable model is wanted (not a compromise) — supersedes the
  earlier colour-only-v1 call given the generalization evidence.
- Sampled **653 present IR clips** (`sample_seg_clips.py --cameras Right_Top --target 800`), rsynced to
  the A100, distributed round-robin into 4 shards, auto-labeling in parallel (~35 min).
- NEXT: merge IR pairs + colour v1 → dataset v2; retrain aug LR-ASPP on v2; eval PER-CAMERA (check IR
  doesn't tank colour, and measure IR quality — watch for the known IR over-segmentation on bright tools).

### Remaining
- [ ] Merge → v2 dataset (colour + IR), retrain aug LR-ASPP, per-camera eval.
- [ ] Pick best model; update segment_octopus for the chosen arch.
- [ ] Pull weights + dataset + diagnostics; scoped-delete A100 seg files; commit + update AGENTS.md.
