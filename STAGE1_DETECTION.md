# Stage 1 — Octopus Detection (visibility gate)

Review document. Every number names the file it comes from so it can be checked.
Scope: **detection only.** Mask area / segmentation presence is Stage 2 and is deliberately
excluded except where noted as deferred.

Date: 2026-08-20

---

## 1. What the detector is

| | |
|---|---|
| Backbone | CLIP ViT-B/32, **frozen** |
| Head | MLP probe `mlp_256_64` (512 → 256 → 64 → 2), classes `hidden` / `visible` |
| Weights | **`weights/clip_mlp_hardneg_v2.pt`** |
| Preprocessing | **letterbox pad-to-square**, NOT CLIP's centre-crop (crop discards 33–44% of a 16:9 frame) |
| Training data | letterbox + 66 verified hard negatives + ~1.6k mined IR-noise/reflection hard negatives, class-weighted |
| Shipped threshold | `p_visible >= 0.60`, and a 20 s window is kept when **>50% of its frames** pass |

**Verified in code** — the same checkpoint is used everywhere in the live pipeline:
`extract_octopus_clips.py`, `harvest_stream.py`, `caption_openrouter.py` (frame selection),
`local_pipeline.py` and `extract_behaviour_records.py` (both inherit via
`from caption_openrouter import load_detector`), and `eval_presence_headtohead.py` (so the published
head-to-head evaluated the **shipped** model, not an older one).
Only `phase2/.../exp26_remote_scan.py` still loads `clip_mlp_best.pt` — a training-frame harvester,
not the pipeline, and deliberate per AGENTS.md.

---

## 2. Hand-labelled detection data we already have

| set | frames | breakdown | source |
|---|---|---|---|
| **EMPTY-V2** | **120** / 60 videos | 97 `empty`, **23 `octopus_present`** | `data/empty_negatives/index.json` |
| **REFL** (Right_Left) | 42 of 150 staged | 36 `empty`, 6 `octopus_present`, **108 never labelled** | `data/reflection_negatives/index.json` |
| hard negatives | 232 | 166 octopus, 66 true negatives | `data/hard_negatives/review_decisions.csv` |
| seg masks (Stage 2) | 513 | 412 masks, 87 negatives | `data/dataset_seg_human/manifest.jsonl` |

≈900 hand-labelled detection items. **No new labelling is needed to evaluate the gate.**

### Two assets currently unused
- **23 human-confirmed positives inside EMPTY-V2.** Sampled at uniform random timestamps over whole
  videos, so they are *unbiased* positives — the arm needed to address the paper's stated
  "upper bound" caveat, which exists because positives elsewhere come from extractor-selected clips.
- **108 REFL frames staged but never labelled.** Finishing them takes the reflection negative set
  from 36 → up to ~140 frames (29 → 34 videos) and tightens every reflection CI. Needs a human;
  `ui/verify_negatives.py` already exists for it.

---

## 3. Results

### 3a. NEW — trained probe vs zero-shot CLIP
`src/eval_zeroshot_vs_probe.py` → `data/zeroshot_vs_probe.json`

Set: EMPTY-V2's **120 human-labelled frames / 60 videos** (23 present / 97 empty). Sampled at
uniform random timestamps over whole source videos → **detector-independent**. Leak-free: the
sampler excludes thin768's 142 training videos *and* the CLIP detector's training sessions.

| arm (identical 120 frames) | AUC | CI95 | FP@R.90 |
|---|---|---|---|
| **trained probe** `clip_mlp_hardneg_v2` | **0.7450** | [0.564, 0.890] | 0.856 |
| **zero-shot CLIP**, same backbone, 5+5 prompt ensemble | **0.4500** | [0.259, 0.643] | 0.959 |

**Paired ΔAUC = +0.2950, CI95 [+0.069, +0.528], cluster-bootstrapped by source video — excludes 0.**

- Zero-shot is **at chance**, and its median score is **higher on empty frames (0.182) than on frames
  containing the animal (0.129)**.
- Both arms share the frozen backbone **and** the letterbox preprocessing, so the gap isolates the
  **probe**, not a different feature extractor.
- Zero-shot was given a deliberately generous 5-octopus + 5-empty prompt ensemble (prompts stored in
  the output JSON) so the margin is not a strawman artefact.
- **Conclusion:** on this footage CLIP supplies features and the discriminative ability is
  essentially all supervised. This is the first *measured* number behind "zero-shot CLIP abandoned
  as unreliable".

### 3b. Context — other detector numbers on record
| number | set | comparable? |
|---|---|---|
| **96.8%** accuracy | internal detection test split | **No baseline, self-selected.** Not comparable to any AUC above. |
| AUC **0.7989** [0.598, 0.882], FP@R.90 0.700 | 30 human-verified reflection frames / 24 videos (R14) | vs mask area 0.9126 — that is a Stage-2 comparison |

### 3c. Deployment-facing numbers (measured, NOT yet in the paper)
At the **shipped** threshold `p_visible >= 0.60`, on the unbiased EMPTY-V2 frames:

| | value |
|---|---|
| recall on unbiased positives | **0.609** |
| FP rate on human-verified empty frames | **0.175** |

So at the threshold we actually ship, the detector **misses ~39% of frames where a human can see the
animal**, when those frames are random moments rather than curated clearly-visible ones.

---

## 4. What must NOT be used

**The 232 mined hard negatives cannot score the detector.** They were selected *because* the model
was confidently positive (`p_visible >= 0.70`), so on that set:

```
p_visible range: 0.81 – 1.00   (214 of 232 sit at exactly 1.00)
```

Any AUC computed there (it gives 0.7805) measures residual variation inside the model's own confident
region — a **selection artifact**, not performance. The set is still valid for what it was built for:
mining hard negatives, and evaluating a model that did *not* select it (e.g. OWLv2).

**Paper wording to fix.** v2 says OWLv2's "scores never separated the two classes". The stored scores
in that CSV give **AUC 0.7589** — real rank-ordering. The defensible claim is about the *threshold*:
at 0.10 it fired on 231/232, so no usable operating point was found. As written the sentence is
contradicted by our own released data.

---

## 5. Paper status

**IN the paper** — §III-A *Visibility Detection*, "What the probe adds over CLIP alone" (line ~311):
the zero-shot comparison, ΔAUC, prompt-ensemble note, and the 23-positive caveat.
Cost: paper went **8 → 9 pages** (0 errors, 0 overfull). Page 8 was already nearly full.

**NOT in the paper** (verified by grep — neither string appears):
- the **0.609** shipped-threshold recall (§3c) — omitted for space, not principle. It is the
  detector's own number at the detector's own threshold, so §III-A is its natural home.
- the deferred Stage-2 number: mask area **0.7064** [0.544, 0.854] on the same 23 unbiased positives
  vs the **0.907** the paper reports. Same negatives, same model, same threshold — only the positives
  differ (curated human-masked "definitely present" frames vs random moments). Not a contradiction,
  a harder question, and the one deployment faces. **This set cannot rank probe vs mask area** — their
  CIs overlap heavily; it only separates both from zero-shot.

---

## 6. Open items for Stage 1

1. **Decide** whether the 0.609 recall goes into §III-A. (Recommend yes — one sentence.)
2. **Fix** the OWLv2 "never separated" sentence (§4 above). Cheap, and it is checkable by a reviewer.
3. **Finish the 108 staged REFL frames** — needs a human, tightens every reflection CI.
4. **Use the 23 unbiased positives** as a second positive arm in the head-to-head, to address the
   stated "upper bound" caveat.
5. **Page budget:** confirm the OCEANS 2026 limit. v1 was deliberately held at 7 pages; v2 is now 9.
   T2 (demote the activity budget) and T3 (retire the enrichment contrast) both *remove* text.

## 7. Caveats that travel with every number here
- **23 positives** in EMPTY-V2 → all intervals wide; claim orderings, not magnitudes.
- **Frame-level**, whereas the shipped gate acts on 20 s windows (>50% of frames). Per-frame is a proxy.
- **Do not compare 0.745 to 96.8%** — different metric *and* a harder, unbiased set.
- Cluster bootstraps group **both** arms by source video (R9's correction: grouping only the
  negatives understates clustering and flatters the result).
