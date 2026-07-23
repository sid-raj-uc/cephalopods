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

### IR auto-labeling — FAILED the quality bar (confirms plan's IR caution)
- 653 present IR clips → only **87 accepted (13%)**: GroundingDINO isn't confident on greyscale IR,
  so the seed-conf gate rejects 86%.
- Accepted IR masks **over-segment**: area median 8.5% / mean 14.7% / 27.5% >20% (vs colour 2.9%) —
  SAM2 grabs bright metal tools/pipes. `auto_segment.py` lacks the Phase-0 IR fix (point/negative prompts).
- Hard-filtered to octopus-range area (0.012–0.13) → only 189 of 345 IR pairs survived → v2 = 4,412
  colour + 189 IR. v2 retrain (aug LR-ASPP): val IoU ~0.47 — IR neither helped nor hurt colour.
- **Verdict: IR unusable without the Phase-0 IR fix. Defer to v2 as the plan said.**

### Presence-gate eval — THE deployment test (`scratchpad/eval_presence.py`) — v1 FAILS
- Ran best colour model (aug LR-ASPP) on 300 present val frames vs held-out negatives
  (60 Right_Left reflections + 60 octopus-absent colour clips).
- Mask-area medians: **present 0.027, reflection 0.039, absent 0.017**.
- **AUC (area separates present vs neg) = 0.496 — essentially RANDOM.** vs reflection 0.418 (worse than
  chance — reflections get BIGGER masks than real octopus). Threshold sweep: reflection-FP ≥ present-recall
  at every threshold. **v1 is NOT a usable presence gate.**
- **ROOT CAUSE (key insight): the model was trained ONLY on octopus-present frames** — every one of the
  4,412 masks contains an octopus, so it learned to ALWAYS emit an octopus-shaped blob, including on
  reflections/empty tank. It was never shown a negative. This also explains the "blob in the wrong place"
  mislocation and why arch/aug/IR couldn't move val IoU.

### Decision: add NEGATIVE (empty-mask) frames to training → v3
- Extract frames from reflection + octopus-absent clips, pair with EMPTY masks, add to training so the
  model learns "no octopus → no mask". This directly targets the presence-gate goal (and should sharpen
  localization). Keep the eval's seg_neg set held-out (distinct clips). Retrain, re-run presence eval.

### v3 built + training
- v2 (colour+189 IR) final: **val IoU 0.492** — marginal vs colour-only, IR confirmed not helpful.
- Built **v3 = 5,800 pairs** (4,412 positives + 1,388 empty-mask negatives from 350 reflection/absent
  clips, 24% neg). Training aug LR-ASPP, 60 ep (~65 s/ep — bigger set + CPU-bound aug).
- **Infra note (bug hit + fixed):** the background waiter scripts used `pgrep -c -f build_v3.py` /
  `train_segmenter.py`, which **match pgrep's own command line** (self-match) → count always ≥1 → the
  chained automation stalled at step 1 and never launched v3, and an earlier waiter never fired.
  Fix: `pgrep -cf "[t]rain_segmenter"` (bracket trick). Relaunched clean.
- v3 val IoU will NOT be comparable to v1 (val now contains easy negatives) — the metric that matters
  is the **presence-eval AUC** on the held-out seg_neg set.

### v3 RESULT — negatives fix the presence gate (the deployment win) ✅
v3 (aug LR-ASPP, 4,412 pos + 1,388 neg). Presence eval on held-out negatives (distinct clips):

| metric | v1 (no negs) | **v3 (with negs)** |
|---|---|---|
| AUC present vs all-neg | 0.496 (random) | **0.860** |
| AUC vs **reflections** | 0.418 | **0.991** |
| AUC vs absent (empty tank) | 0.575 | 0.725 |
| reflection mask area (median/mean) | 0.039 / 0.065 | **0.000 / 0.000** |

At op-threshold area≥0.01: **present-recall 0.88, reflection-FP 0.00**, absent-FP 0.52.
**Reflections — the #1 extraction false-positive (esp. Right_Left) — are essentially solved:** the
model emits ZERO mask on glass reflections. This is the mask-gate payoff the plan wanted, and it beats
the CLIP gate (which fires at p=1.0 on the same reflections).

### Honest bottom line
- **Deployment payoff (reflection-rejecting presence gate): achieved.** v3 is usable to clean extraction.
- **Mask pixel-quality bar (IoU 0.85): NOT met** — present-frame masks ~0.5 IoU (ok for coarse
  body-area / masked-motion, not precise). Ceiling is data diversity (62 colour videos), not model/aug.
- **Empty-tank "absent" discrimination (0.725): decent, improvable** with more absent negatives.
- Key insight that unlocked it: **train with negatives** (v1's fatal flaw was positives-only).

### Deliverables pulled back to the repo (before A100 cleanup)
- `weights/seg/octo_seg_*.pt` — all 8 checkpoints (sweep + aug + v2 + **v3 = deployable**).
- `data/dataset_seg/{v1,v3}` — the 4,412-pos mask dataset (+ v3 negatives). [gitignored, local]
- `results/segmentation/` — all train/eval logs + diagnostic overlays.
- `src/eval_presence.py`, `src/build_v3.py` — eval + negatives-dataset scripts.

### A100 cleaned up (2026-07-23)
- All segmentation artifacts deleted from `amera-vllm-a100` (home 31G → 28K): dataset_seg, seg_clips*,
  ir_shard*, seg-venv, weights_seg, all scripts + logs, and the HF/torch/pip/.nv caches I created.
- Left intact (not ours / shared): default dotfiles, `~/.ssh` (SSH access key), and the apt system
  packages (ffmpeg, python3-pip, python3-venv). GPU idle, 189G free. All artifacts already pulled to the repo.

### Recommended next (needs more data — user offered)
- More DISTINCT colour videos (diversity, not volume) → raises present-mask IoU + absent-case AUC.
- A small human-verified mask val set (~100–200) for trustworthy numbers.
- Implement the Phase-0 IR fix (point/negative prompts) before using IR.
- Then wire `segment_octopus` (area≥~0.01 gate) into extraction and A/B vs the CLIP gate.
