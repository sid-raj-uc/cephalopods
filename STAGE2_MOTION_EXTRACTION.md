# Stage 2 — Motion Detection + Clip Extraction  (PARKED 2026-08-20)

Review document. Every claim names the file it came from. **Status: parked** — recorded here so the
findings are not lost; no code or paper changes made.

Scope: the two components between detection and everything downstream —
**(a) motion detection**, **(b) the AND gate that turns per-second scores into 20 s clips.**
Stage 1 (the visibility detector) is in `STAGE1_DETECTION.md`.

---

## 1. Motion detection — `src/motion_detector.py`

Two functions live in this file and **only one is safe**:

| function | behaviour | use? |
|---|---|---|
| `scan_motion()` | normalises by `scores.max()` **per video** (line ~104) | **NO.** A flickering lamp in an otherwise static video is scaled up to "motion". The original shipped bug. Still present in the file. |
| **`scan_motion_area()`** | **absolute** changed-pixel fraction: `(diff > pix_thresh).mean()` | **YES** — the correct method |

- `pix_thresh = 25` grey-levels per pixel to count as "moved".
- **Timestamp masking**: `diff[int(h*0.88):, int(w*0.60):] = 0` zeroes the burned-in bottom-right
  datetime, so the ticking clock is not counted as motion.
- `scan_motion_area` was **added alongside** `scan_motion`, not as a replacement — hence the footgun.

## 2. The clip-finding gate — `src/extract_octopus_clips.py`

```
non-overlapping 20 s windows; keep a window when
    >50%  of its frames have p_visible >= 0.60     (MIN_VISIBLE_FRAC, VIS_THRESH)
AND  mean absolute motion >= 0.008                 (MOTION_THRESH)
```
Extraction is then an ffmpeg byte-range copy (no re-encode).

| param | value | line |
|---|---|---|
| `CLIP_LEN` | 20 s | 63 |
| `MIN_VISIBLE_FRAC` | 0.50 | 64 |
| `VIS_THRESH` | 0.60 | 65 |
| `MOTION_THRESH` | **0.008** | 66 |
| `MOTION_PIX` | 25 | 68 |
| `CAMERAS` | includes **`Right Left`** | 59 |

**Keep rate: 13,342 clips from 89,758 candidate 20 s windows = 14.9%.**
(Candidates computed from `src/octopus_clips_processed.json`: 1,117 videos x `n_frames` at 1 fps,
~499 h of scanned footage.)

---

## 3. How was `MOTION_THRESH = 0.008` selected? — **THERE IS NO SELECTION DATA**

Searched exhaustively:

| looked in | found |
|---|---|
| the code | one inline comment: *"raised from 0.005: 0.005 let IR-noise/reflection false positives through, esp. Right_Left"* |
| `data/motion_debug/` | **4 PNGs** (2026-06-26): `area_motion_compare`, `motion_heatmap`, `per_frame_motion`, `source_motion_timeline`. Plots only — **no numbers saved** |
| `PAPER_NOTES.md` | **no mention of the motion threshold** |
| `BENCHMARKS.md` | **no mention** |
| `src/SEGMENTATION_LOG.md` | **no mention** |
| scripts | `MOTION_THRESH` appears in 13 files — **all consumers. No sweep or eval script exists.** |

**Conclusion: the threshold that defines the entire corpus was chosen by eyeballing plots, and it
never entered the benchmark ledger.** BENCHMARKS.md's own rule is that every number in the paper is
measured by `src/benchmarks.py` on frozen sets; this parameter is not a "claim", so it slipped through
— yet it is more consequential than most claims, because it is a **hard floor on what data exists**.

### There are also no recorded RULES for choosing an operating point
The project has strong explicit rules for benchmark *sets* (frozen; split by source video; negatives
never pooled; report negatives) and **none** for choosing a threshold. The three gate parameters were
each set once:
- `MOTION_THRESH` — changed once, by inspection (0.005 → 0.008)
- `MIN_VISIBLE_FRAC = 0.50` — **never varied**
- `CLIP_LEN = 20 s` — **never varied**

The project demonstrably knows how to do this properly: **`src/fusion_threshold_sweep.py`** exists
precisely because "a fused median map is not calibrated like a single-frame map; comparing at a fixed
0.5 would fake a result either way." That reasoning applies verbatim here and was not applied.

---

## 4. What IS measured about Stage 2

- `data/clips_motion_audit.json` (8 fields/clip: `mean_area`, `max_area`, `frac_frames_gt1pct`,
  `survive`) + `data/clips_motion_survivors.txt` (**116 survivors**) — the exp30 re-audit. This is a
  **consistency** re-check (does the clip still pass absolute motion), **not** a correctness check.
- The only evidence on the gate's **precision** is the VLM verdict on the 847-clip set:
  **534 (63%) came back `octopus not present`**, overwhelmingly `Right_Left`. That is an AI label,
  not human, and it is a single data point on one camera-skewed subset.

## 5. The gap that blocks a proper sweep

**All ~900 hand labels in this project are FRAME-level.** Nothing labels *"should this 20 s window
have been kept?"* So today we can measure neither the gate's precision nor its recall honestly.

A defensible sweep needs window-level labels on a sample that **spans the 0.005–0.02 motion band and
includes windows the current gate REJECTS.** Sampling only kept windows measures precision alone and
repeats the conditioning trap that makes the 232 mined hard negatives unusable for scoring the
detector (they were selected at `p_visible >= 0.70`, so that column spans only 0.81–1.00).

## 6. Consequences decided at this stage

- **The corpus is left-truncated at the motion gate.** Across the 3,205 behaviour-record clips:
  `min mean_motion 0.00720`, `p1 0.00811`, `median 0.02892` — **nothing below 0.005 exists.**
  Whatever that costs downstream is decided here, by a parameter with no measured basis.
- `Right Left` is still in `CAMERAS` despite the 63% not-present finding, and despite
  `auto_segment.py` excluding that camera by construction.
- The harvester deliberately went the other way: `harvest_stream.py` sets `REQUIRE_MOTION = False`
  ("still-but-visible octopus is good seg/caption data"), so **the harvest and the main corpus were
  built under different gates** and are not interchangeable.

## 7. Open items for Stage 2 (when unparked)

1. Build window-level labels spanning the motion band, **including rejected windows** — the
   prerequisite for everything else here.
2. Sweep `MOTION_THRESH` and `MIN_VISIBLE_FRAC` against those labels; record in `data/benchmarks.json`.
3. Decide `Right Left`: drop from `CAMERAS`, or justify keeping it.
4. Delete or clearly quarantine `scan_motion()` so the normalisation bug cannot be reintroduced.
5. Note that changing any gate parameter **changes which clips exist**, so it invalidates the current
   corpus rather than improving it in place — must be a deliberate, versioned re-run.

---

## NOT part of Stage 2 — parked separately, do not lose
While tracing the motion gate I found a defect in the **downstream circadian statistic** (Stage 4+,
`src/analyze_behaviour.py`): `present_rate` divides VLM-confirmed present clips (from the 3,205
processed) by **all 13,342 index clips**, so the 10,137 never-processed clips count as absent by
omission. Result: `Pearson r(processing coverage, published rate) = 0.9997`. Recomputing with the
correct denominator (89,758 candidate windows) keeps the pattern — peak still 17:00, 13:00–19:00
plateau, dawn bump — but the magnitude falls from **13.4x to 4.7x** (peak 45%→34.4%, overnight
~1–5%→7.3%; 4.2x excluding `Right_Left`). Not acted on. Belongs to the behavioural-analysis stage.
