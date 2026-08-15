# Benchmark suite

Three **frozen** benchmarks. Every improvement claim in this project — and every number in the
OCEANS 2026 paper — is measured here, before and after. Run them all with one command:

```bash
venv/bin/python3 src/benchmarks.py --suite all --tag <run-name> --latex
```
Results append to `data/benchmarks.json` (keyed by tag); `--latex` writes
`OCEANS_2026/assets/benchmarks_table.tex` for the paper.

## Non-negotiable rules
1. **Frozen sets.** The frame/clip lists are committed and must not be regenerated to suit a result.
2. **Split by SOURCE VIDEO, never by frame.** Frames from one recording are near-duplicates; a
   frame-level split leaks. We shipped this bug once (an apparent 0.49→0.70 gain evaporated under a
   video-level holdout) — see `SEGMENTATION_LOG.md`.
3. **The holdout videos are excluded from every training source**, not just the one being trained.
4. **Report the metric that can't be gamed** (see the tip-F1 note below), and report negatives.

---

## SEG-TEST — segmentation quality (leak-free, human-verified)
| | |
|---|---|
| Data | `data/dataset_seg_human/` — 412 human masks + 87 empty-mask negatives, 35 source videos |
| Test split | 5 held-out source videos → **122 frames** (+19 negatives), excluded from all training |
| Labels | human, via click-to-SAM2 (`ui/seg_label.py`, port 8015): a human clicks the octopus, SAM2 masks it, the human accepts/corrects |
| Metrics | mask **IoU** (mean, median); **body-area error** (%) — the quantity the behaviour analysis actually consumes; **presence AUC** — does mask area separate present frames from empty ones |

Holdout videos: `2026-02-21/150002`, `2026-02-21/183003`, `2026-02-22/153002`,
`2026-02-22/190003`, `2026-02-23/170003`.

## SKEL-50 — per-frame skeleton quality
| | |
|---|---|
| Data | `data/skel_bench50/frames.json` — **50 frozen frames** across 20 source videos, each with image + human mask + model mask |
| Metrics | **arm-tip F1** (headline), tip precision, tip recall, head error (body radii), arms/frame (descriptive only) |
| Ground truth for tips | the ≥8 strongest protrusions of the **human** mask (`finger_tips`), capped at 8 (biological maximum); a predicted tip matches a GT protrusion within 5% of the image diagonal, greedy 1-1 |
| Head GT | human clicks on the eyes via `ui/skel_static_viewer.py` (port 8019) → `data/skel_bench50/head_gt.json`; head error is reported in **body radii** so it is pose/scale independent |

> **Why tip-F1 and not "arms per frame".** Arm count is not a score: *fewer* can be *better*.
> Anti-mess quality gates removed ~1.3 duplicate/tangle arms per frame and the count fell
> 4.80 → 3.48 while the output got visibly cleaner. Precision alone is equally gameable (emit one
> obvious arm). F1 against the human mask's protrusions penalises **both** spurious arms and missed
> arms, so it is the number to optimise and to publish. Arm count stays as a descriptive statistic.

## TRACK-10 — temporal tracking quality
| | |
|---|---|
| Data | 10 frozen clips (`EVAL_CLIPS` in `src/skel_eval_tracking.py`) spanning all 6 behaviour classes, 3 colour cameras, 3 dates |
| Metrics | **teleport rate** (per-node steps jumping > 4× that node's median step — proxy for identity swaps), **teleport-confident** (same, restricted to evidence-backed samples), **occluded fraction** (share of arm samples that are evidence-free holds), **coverage**, **fragmentation**, in-mask fraction, arm-count stability |

> **Why occluded-fraction matters.** Naive tracking looked smooth because 42% of arm samples were
> held with no evidence. Kinematics are computed only from `detected`/`fitted` samples; this metric
> keeps that honesty visible.

---

## Current results
See `data/benchmarks.json` (authoritative, tagged per run) and the summary table in
`PAPER_NOTES.md`. Reproduce any row with:
```bash
venv/bin/python3 src/benchmarks.py --suite seg,skel,track --ckpt weights/seg/<model>.pt --tag <name>
```
