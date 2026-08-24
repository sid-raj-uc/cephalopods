# AGENTS.md — Cephalopod / "Nity" Octopus Detection Project

Guidance for AI agents working in this repo. Read this before touching the
detection pipeline. Despite the repo name (`sentiment-analysis`), this is a
**cephalopod video-analysis project**: detecting the octopus "Nity" in aquarium
camera footage, classifying whether the octopus is **visible** vs **hidden**, and
extracting behavioral clips.

## Environment

- **Python**: use the project venv at `venv/bin/python3`. Do NOT use homebrew/system python.
- **Jupyter kernel**: a registered kernel named `octopus-venv` points at `venv/bin/python3`.
  Execute notebooks with:
  ```
  cd phase2 && WANDB_SILENT=true python3 -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=octopus-venv <nb>.ipynb
  ```
  (Clip-pipeline notebooks moved to `phase2/octo-clip-extraction/` — pass that path to the
  `<nb>.ipynb` argument. They discover the repo root by walking up, so any cwd under the repo works.)
- **Device**: Apple Silicon (MPS). MPS works for CLIP. **OWLv2 hangs on MPS — use `device="cpu"`** for it.
- **`clip` import shim** (OpenAI CLIP needs `packaging` shimmed into `pkg_resources`):
  ```python
  try:
      import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
      pkg_resources.packaging = packaging
  except Exception: pass
  import clip as clip_lib
  ```
- **Server (footage source)**: `repo.octopus-intelligence.org`. Credentials are **no longer
  hardcoded** — they live in the gitignored repo-root `.env` as `OCTOPUS_USER` / `OCTOPUS_PASS`
  and are loaded via `server_creds.py` (`from server_creds import USER, PASS`; zero-dependency
  .env parser). Scripts add the repo root to `sys.path` first. Do NOT paste the password back
  into source. (NOTE: the old plaintext creds remain in git history; rotating them server-side
  is the only real fix for that.)

## The model: CLIP + MLP probe

- **Architecture**: CLIP ViT-B/32 (frozen) → MLP probe `mlp_256_64` (512 → 256 → 64 → 2).
  Classes: `hidden` (no octopus) vs `visible` (octopus in frame).
- **Training notebook**: `phase2/exp18_clip_mlp.ipynb`. Trains the probe on cached CLIP
  features over `data/frames/{visible,hidden}/`.
- **CRITICAL — preprocessing is letterbox, not crop.** CLIP's default `CenterCrop` is
  destructive (drops 33–44% of a 16:9 frame). We pad-to-square instead:
  ```python
  def letterbox(img, size=224, fill=(128,128,128)):
      w,h = img.size; s = size/max(w,h); nw,nh = max(1,round(w*s)), max(1,round(h*s))
      img = img.resize((nw,nh), Image.BICUBIC)
      cv = Image.new("RGB",(size,size),fill); cv.paste(img, ((size-nw)//2,(size-nh)//2)); return cv
  # use: preprocess(letterbox(Image.open(p).convert("RGB")))
  ```
  Aspect-ratio mismatch (not architecture) was the root cause of poor field performance.
  We deliberately did NOT switch to a ResNet — the fix was preprocessing.
- **CRITICAL — feature cache gotcha**: CLIP features are cached at
  `data/frames/clip_features.npz`, keyed on file paths NOT on the transform. **If you change
  preprocessing, `rm data/frames/clip_features.npz` before retraining** or you'll train on stale features.
- Do not use augmented data (`augment_hidden.py` output) in training — current models are trained without it.

### Weights (`weights/`)

> **USE `clip_mlp_hardneg_v2.pt` for the clip-extraction pipeline** (set 2026-06-27). It is the
> latest model — letterbox + 66 original hard negs + ~1.6k mined IR-noise/reflection hard negatives
> (the false positives from the 0.005-era Right_Left clips), class-weighted. In A/B vs
> `clip_mlp_best.pt` it cut the hidden false-positive rate 24%→3% while holding visible recall ~0.97.
> NOTE: `extract_octopus_clips.py` uses it; `exp26_remote_scan.py` (frame harvester) and
> `exp28_verify_clips.py` (audit) still load `clip_mlp_best.pt` — switch those only on instruction.
> Do not change a script's active model without an explicit instruction.

All are CLIP ViT-B/32 + `mlp_256_64` probes unless noted. **Headline accuracies are NOT
directly comparable** — each was scored on a different test set (different preprocessing /
label cleanliness), so a higher number does not mean a better model.

| File | Use | Preprocessing / data | Acc | Trust |
|------|-----|----------------------|-----|-------|
| `clip_mlp_hardneg_v2.pt` | **LATEST — clip-extraction pipeline uses this** (66 + ~1.6k mined hard negs, class-weighted) | letterbox + ~1.7k hard negs | 96.8% | ✅ |
| `clip_mlp_best.pt` | prior default; still used by exp26 scanner + exp28 audit (letterbox + 66 verified hard negs) | letterbox + 66 hard negs | 96.3% | ✅ |
| `clip_mlp_letterbox_v1.pt` | clean letterbox baseline before hard negs (kept for A/B) | clean letterbox | 96.9% | ✅ |
| `clip_mlp_hardneg_unverified.pt` | trap: 232 UNVERIFIED labels (166 were actually octopus) — inflated | letterbox + 232 unverified | 97.2% | ❌ |
| `clip_mlp_crop.pt` | old destructive squish/crop model | CenterCrop (drops 33–44%) | 96.8% | ❌ deprecated |
| `clip_linear_best.pt` | old **linear** probe (not MLP), superseded | — | 88.8% | ❌ |

- Checkpoint dict keys: `state_dict`, `clip_model`, `arch`, `feat_dim`, `label_map`, `test_acc`.

## Motion detection — `phase2/octo-clip-extraction/motion_detector.py`

- **BUG (do not reintroduce)**: the original per-video **normalized** motion (`motion/motion.max()`)
  passes static videos — a flickering lamp gets normalized up to "motion". See memory
  `motion_gate_normalization_bug.md`.
- **Use the absolute method**: `scan_motion_area(source, fps=1.0, pix_thresh=25, mask_timestamp=True)`
  returns the **absolute changed-pixel fraction**. It masks the timestamp region
  (`diff[int(h*0.88):, int(w*0.60):] = 0`) so the ticking clock isn't counted as motion.
  This was ADDED alongside, not replacing, the old `scan_motion()`.
- **Location**: `motion_detector.py` now lives in `phase2/octo-clip-extraction/` (with the rest of
  the clip pipeline). It is still imported by non-pipeline scripts that stayed in `phase2/`
  (`exp16_motion_timeline.py`, `exp22_auto_scan.py`, `run_aquarium_analysis.py`) — they add
  `phase2/octo-clip-extraction` to `sys.path` and `import motion_detector` (the folder name has
  hyphens, so it is NOT importable as a package — always use sys.path + flat import, never
  `from octo-clip-extraction.x import`).

## Clip extraction & verification pipeline

All of these live in **`phase2/octo-clip-extraction/`**. Each script computes the repo root as
`Path(__file__).resolve().parents[2]`; notebooks discover it by walking up for `data/`+`weights/`.

- **`phase2/octo-clip-extraction/extract_octopus_clips.py` — THE clip extractor (use this).**
  The clean, consolidated pipeline: octopus detection (`clip_mlp_hardneg_v2.pt`, letterbox) + motion
  detection (`scan_motion_area`, the correct ABSOLUTE method) + 20s clip extraction, in one script.
  Per video it makes two 1 fps ffmpeg passes (octopus, then motion via `scan_motion_area`), slides a
  non-overlapping 20s window, and keeps a window when **>50% of frames are octopus-visible
  (`p_visible ≥ 0.6`, `--vis-thresh`) AND mean absolute motion ≥ `--motion-thresh`** (default 0.008,
  raised from 0.005 because the 0.005–0.008 band was IR-noise/reflection false positives, esp. on
  Right_Left). Extracts via ffmpeg
  byte-range copy. Outputs clips to `data/octopus_clips_verified/{date}/{segment}/...` (extract_clip
  skips paths that already exist, so existing verified clips are not overwritten), index
  `data/octopus_clips_verified.json` (the clip index — one entry per clip: clip_path, camera,
  video_url, video_timeline, start/end, scores), ledger `data/octopus_clips_processed.json`. Flags: `--limit`, `--date`,
  `--motion-thresh`, `--visible-frac`.
  **This supersedes the exp27 + exp28 + exp30 chain** — because both gates are correct here, clips
  come out clean in one pass, so exp28/exp30 are no longer required (keep them only as optional audits).
- `phase2/octo-clip-extraction/exp26_remote_scan.py` — a different tool: harvests training *frames*
  (visible/hidden) to `data/scanned_frames/`, not clips. streams remote videos (ffmpeg HTTP → image2pipe), **motion-gates
  then classifies**. Per video it first runs `scan_motion_area` (per-second absolute changed-pixel
  fraction), then streams frames through the letterbox filter
  `scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:-1:-1:color=gray`. Each frame is
  annotated with its motion fraction; with the gate ON (default) frames below `--motion-thresh`
  (default 0.005, `pix_thresh=25` — matches exp30) are marked `static` and are NOT classified or
  saved. Moving frames are classified with **`weights/clip_mlp_best.pt`** (NOT the global
  `letterbox_v1` default — set explicitly for this scanner per instruction, 2026-06-26) and saved to
  `data/scanned_frames/{visible,hidden}/`. Flags: `--no-gate`, `--motion-thresh`, `--motion-pix`.
  Per-video JSON now carries `model`/`motion_gate`/`n_static` + per-frame `motion`; the summary CSV
  gained `n_classified,n_static,frac_static,motion_mean` (old file auto-rotated to
  `scan_summary_pre_motion.csv`). Tracks done videos in `data/processed_videos.json` (the canonical
  "already processed" ledger — always update it).
- `phase2/octo-clip-extraction/exp27_octopus_clips.ipynb` — **SUPERSEDED** by `extract_octopus_clips.py`.
  Old clip extractor; gated on the buggy per-video **normalized** motion (`motion/motion.max()`), so it
  over-extracts. Kept for reference only — do not use for new extraction.
- `phase2/octo-clip-extraction/exp28_verify_clips.py` — optional audit: re-runs the octopus check over
  extracted clips. Writes `data/clips_verify_audit.json` (NOT `octopus_clips_verified.json` — that is now
  the live clip index written by `extract_octopus_clips.py`). No longer required in the main flow.
- `phase2/octo-clip-extraction/exp30_audit_clip_motion.py` — optional audit: re-audits clips with `scan_motion_area`;
  writes `data/clips_motion_audit.json` + `data/clips_motion_survivors.txt`. No longer required in the main
  flow (the consolidated extractor already gates on absolute motion).
- `phase2/octo-clip-extraction/exp29_motion_debug.ipynb`, `phase2/octo-clip-extraction/exp31_saliency.ipynb` — forensics: false-motion debug,
  and occlusion saliency (what pixels make the model say "octopus").
- `phase2/octo-clip-extraction/caption_octopus_clips.ipynb` — **Colab/GPU** captioner+classifier
  (Qwen3-VL-30B AWQ via vLLM, A100). Reads `octopus_clips_verified.json`, samples frames from each
  clip, and writes back onto each clip entry: a one-sentence `caption` AND an `ethogram_label` chosen
  from `data/ethogram_list.json` (19 behaviors) — or `octopus not present` for both if no octopus
  (plus `captioned_at`, `caption_model`). Parsing validates the label against the list with a keyword
  fallback (`match_ethogram`) for near-misses. **Resumable** — skips entries that already have a
  `caption`; saves the JSON after every clip. To run: zip the `octopus_clips_verified/` folder to
  `octopus_clips_verified.zip` + copy `octopus_clips_verified.json` and `ethogram_list.json` to Drive
  (`MyDrive/GSOC-Catrobat/`), caption on Colab, then copy the updated JSON back into `data/`. Successor
  to `phase2/exp23_caption_colab.ipynb` (flat clip dir + separate captions json; this writes into the index).
- `phase2/octo-clip-extraction/caption_octopus_clips_v2.ipynb` — **improved captioner** (use this over v1).
  Same Qwen3-VL-30B/vLLM base, but fixes the low-quality captions by improving the VLM *input* on
  the dim IR footage: (1) **CLAHE brightness/contrast** enhancement per frame; (2) **higher res**
  (~768px vs 512); (3) **best-frame selection** — scores candidate frames with `clip_mlp_hardneg_v2.pt`
  and sends only the top-`N_KEEP` where `p_visible` is highest, in time order; (4) **skips no-octopus
  clips** (if no frame's `p_visible` ≥ `PRESENT_MIN`, default 0.5, auto-labels `octopus not present`
  and skips the VLM — catches hallucinations on empty clips); (5) lets Qwen answer **`uncertain`**
  instead of force-guessing. Writes `caption`/`ethogram_label` + `caption_pipeline="v2-enhanced"` +
  `max_p_visible`. **Resumable & non-destructive**: skips clips already done by v2 AND clips
  `review=="approved"`. Needs `clip_mlp_hardneg_v2.pt` on Drive too. `PRESENT_MIN` is tunable (raise
  = stricter; too high can skip dim/partial real octopus).
- `ui/review_captions.py` — **caption review UI** (FastAPI, port 8005). Plays each clip from local
  disk (`data/octopus_clips_verified/`) next to its `caption` + `ethogram_label`; approve/reject,
  edit the caption (textarea) and label (dropdown of `ethogram_list.json` labels). Writes review
  status + edits back into `octopus_clips_verified.json` per clip (`review`, `reviewed_at`), saved
  after every action (resumable). Keys: A approve, R reject, ←/→ nav, U clear, S save edits.
  Run: `venv/bin/python3 ui/review_captions.py` → http://localhost:8005.

## Hard-negative mining — the verification lesson (IMPORTANT)

**Always verify labels before training on them.** We mined the model's confident-visible
false positives as hard negatives. Workflow:
1. Harvested 232 confident-visible frames (p≥0.70) from Right_Back survivor clips → staged in
   `data/hard_negatives/` (filenames include clip stem to avoid collisions:
   `hardneg_p{p}_{date}_{segment}_{stem}_f{i:02d}.jpg`).
2. **Did NOT assume they were octopus-free.** Ran independent verification:
   - OWLv2 (`google/owlv2-base-patch16-ensemble`, CPU) → `data/hard_negatives/_detector_verify.json`.
     **OWLv2 was useless as an auto-filter**: at thr 0.10 it "detected" octopus in 231/232 frames;
     Its scores are NOT uninformative (AUC **0.759** vs the human labels, measured 2026-08-22 from
     `review_decisions.csv`) — but the overlap leaves no usable operating point: at its own best
     threshold (0.23) it still passes 30% of the negatives while losing 32% of the real animals, and
     any threshold keeping ≥95% of animals passes 70% of negatives. Do not trust OWLv2 alone for
     this, and do NOT repeat the older claim that "scores never separated the classes" — that was
     wrong; the correct claim is "weak ranking signal, no viable operating point".
   - Human review UI: `ui/review_hardneg.py` (port 8004, one image at a time, keyboard bindings
     O/1=octopus, N/0=no-octopus, Space/→ skip, ← back, U clear, F full-res). Decisions →
     `data/hard_negatives/review_decisions.csv`.
3. Result: of 232, **166 actually contained the octopus** (model was right), **66 were genuine
   hard negatives** (no octopus). Only the **66** were folded into `data/frames/hidden/` and the
   model retrained → 96.3% (lower headline acc than before = harder, more honest test set).
- Open follow-up (not yet done): optionally add the 166 confirmed-visible Right_Back frames as
  `visible` training data for better back-angle coverage.

## `src/` — canonical self-contained pipeline (use this going forward)
The clean, portable pipeline lives in **`src/`** (extracted from `phase2/octo-clip-extraction/`).
A bare copy of `src/` runs standalone (verified): all paths are `HERE`-relative, and it bundles
its own `clip_mlp_hardneg_v2.pt` (detector), `server_creds.py`, `ethogram_list_v2.json`, and the two
state JSONs. Contents:
- `extract_octopus_clips.py` — the extractor (clips land in `src/octopus_clips_verified/`, index in
  `src/octopus_clips_verified.json`, ledger `src/octopus_clips_processed.json`).
- `motion_detector.py` — `scan_motion_area`.
- `caption_octopus_clips.ipynb` — **Colab** captioner, teacher = Qwen3-VL-30B (vLLM).
- `caption_openrouter.py` — **local** captioner via the **OpenRouter API** (see below).
- `train_caption_student.ipynb` — LoRA fine-tune of Qwen2.5-VL-3B (caption student).
- `TRAINING_PLAN.md` — the **caption-student plan** (locked 2026-07): caption-ONLY distillation of
  the Qwen3-VL-235B teacher into a **Qwen3-VL-2B** student (QLoRA), loss = LM cross-entropy on the
  caption tokens only. **Incremental = retrain from base on a versioned cumulative snapshot** (Option
  A; "continue when more clips arrive" = rebuild the bigger snapshot + retrain — NOT resume-on-new,
  to avoid catastrophic forgetting). Ethogram label deliberately NOT trained here (its own model later).
- `build_caption_dataset.py` — **dataset builder** (local, run after captioning). Selects present +
  captioned + local clips (caption source `caption_235b`>`caption`, drops "octopus not present"),
  CLIP-dedups within source video (reuses `clip_embeddings.npz`), splits train/val BY SOURCE VIDEO,
  and writes best-N CLAHE frames (identical to teacher input) → `src/dataset/vN/` (frames + train.jsonl
  + val.jsonl + snapshot.json). Flags `--version --dedup-thresh --val-frac --n-frames --caption-keys`.
- `train_caption_student_qwen3vl.ipynb` — **the Colab QLoRA notebook (use this)**. Loads
  `Qwen/Qwen3-VL-2B-Instruct` 4-bit via `AutoModelForImageTextToText` (fallback `MODEL_ID` =
  Qwen2.5-VL-3B), phase-0 smoke test, trains on a `build_caption_dataset.py` snapshot zip (uploaded to
  Drive `caption-student/`), saves the adapter to Drive, evals base-vs-LoRA (emb-sim + rougeL) on the
  held-out val split.
- `local_pipeline.py` — **optimized local pipeline module + CLI** (`python local_pipeline.py <video> --camera X`).
  Same gates/clips/frames as `extract_octopus_clips.py` + the MLX student, but two speedups: **(A) single-decode
  scan** — decodes the video ONCE and feeds both the CLIP octopus classifier (pad→224²) and the motion
  detector (cv2 stretch→224² grey, timestamp-masked) from that one stream (naive thread-parallel decode is
  SLOWER — video decode is CPU-bound + already multithreaded, so two concurrent decodes just contend). Cuts
  scan ~219s→130s on a 30-min video, verified byte-identical clip set (89 windows). **(B) caption reuses the
  scan's per-second p_visible** to pick best-N frames instead of re-extracting dense frames + re-running CLIP
  per clip (equivalent captions; small win since MLX generation dominates ~5–12s/clip and is the real cost —
  batching is the next lever). Exposes `process_video(video, out_dir, M, on_stage=, on_clip=)` for the UI.
- `local_video_to_captions.ipynb` — **fully-local end-to-end demo notebook (video in → captions out)**.
  Takes ONE local video (set `VIDEO_PATH`, Run All), runs the same extraction gates as
  `extract_octopus_clips.py` (octopus CLIP+MLP + `scan_motion_area`, 20s windows, >50% visible & motion≥0.008)
  but reads a **local file** (no server/creds), then captions each clip with the **local MLX 4-bit student**
  (`models/qwen3vl2b_caption_v1_mlx_4bit`). Reuses `caption_openrouter`'s exact frame prep (dense→score→top-6
  CLAHE) so the VLM input matches training. Outputs to `local_pipeline_out/` (clips + `<stem>_captions.json`);
  does NOT touch the shared index. ~a few min to scan a 30-min video + ~3s/clip to caption on Apple Silicon.
  Verified end-to-end 2026-07-17. See [[caption-student-mlx-4bit]].
- `.env.example`, `requirements.txt`, `README.md`. `.env` (real creds) is gitignored — never commit it.
- **Deliverable branch `octopus-pipeline-src`** (orphan, on GitHub): only `src/` + `weights/` + a root
  README — the clean shareable package.

## Behaviour analysis — structured extraction → dashboard (the project's actual goal)
The goal is **behaviour/affect understanding of Nity, not a better captioner**. The captioner is
effectively done (v1 student trained on ~all present clips). PoCs (2026-07-20) established the direction —
see memories [[structured-extraction-unlocks-behaviour]] and [[footage-colour-camera-dependent]]:
- **Treat a caption as ONE field in a structured behavioural record**, then do ethology on the aggregate.
- **Colour is camera-gated**: Right_Back/Right_Front are colour (100%), Right_Top is pure IR (0%),
  Right_Left/Right_Right ~10-12%. Only ~42% of present clips can carry a colour signal. Colour *change*
  is NOT reliably measurable (whole-frame variance is movement-confounded) — needs animal segmentation (future).
- **Affect model = arousal (motion + posture-spread + activity) + state (location/context response)**;
  static colour is a secondary channel on colour cameras. Framed as arousal/behavioural-state, NOT emotion.

The pipeline (in `src/`, all outputs under gitignored `data/`):
- **`src/extract_behaviour_records.py`** — structured JSON extraction over all present-on-disk clips via
  OpenRouter Qwen3-VL-235B. Reuses `caption_openrouter.py` frame prep (CLAHE + best-N by `clip_mlp_hardneg_v2`).
  Per clip emits `{present, behavior(7-class), posture, activity, location, context, body_color,
  color_or_texture_change, confidence}` with **snap-to-list validation**; **colour fields gated per-clip**
  (greyscale/IR clips told to leave colour `uncertain`, detected by BGR channel divergence <6). Parallel
  (6 workers, CLIP inference lock-guarded for MPS), **resumable** (skips clip_paths already in output),
  cost-tracked. Writes `data/behaviour_records.json` (keyed by clip_path) — **does NOT touch the index**.
  Full run 2026-07-20: 3,205 clips, $2.22, ~0.0006/clip. (Note: OpenRouter throttles ~36→13 clips/min under
  sustained load; a few clips 429-fail and are recovered by re-running — resumable.)
- **`src/analyze_behaviour.py`** — aggregates `behaviour_records.json` (joins the index for absolute clock
  hour) → `data/behaviour_stats.json`: activity budget, exposure-normalized circadian (present ÷ all
  extracted windows/hour), stimulus response by context, colour-by-context, per-camera. Transparent arousal
  rubric `0.6*activity + 0.4*posture-spread` (edit in-file).
- **`src/render_behaviour_dashboard.py`** — renders stats → `data/behaviour_dashboard.html` (self-contained,
  theme-aware, inline SVG; validated dataviz palette). Artifact fragment → `behaviour_dashboard_artifact.html`.

**Findings (3,083 present, full run):** activity budget 41% exploration / 33% resting / 14% human-interaction
/ 9% reaching-out / 2% crawling / 1% swimming. Circadian: visible-activity rate ~1-5% overnight → **45% peak
@17:00** (13:00-19:00 plateau) + dawn bump 05-06h. **Stimulus response: human presence nearly doubles motion
(0.045→0.095) and lifts arousal 0.46→0.68.** Colour (colour cameras): dark_red_brown most common at baseline
(~16%) vs during human interaction (~6%). CAVEAT: `context="enrichment_object"` fires on ~66% of clips (tank
has permanent toys/pipes) — it means "object in tank", not "active enrichment"; the clean stimulus contrast is
none vs human_present. Presence gate still ~66% dirty upstream — absolute levels shift after detector retrain,
but the rate/response *contrasts* are robust.

**Next (GPU): distill the 235B structured extractor into the local Qwen3-VL-2B student** — same QLoRA recipe
as the caption student, JSON targets instead of a sentence, using `behaviour_records.json` as the training set.

## Captioning: 30B teacher, 235B via API, and the comparison
Two captioners, both write a one-sentence `caption` + a 7-class `ethogram_label`:
- **Colab / Qwen3-VL-30B** (`src/caption_octopus_clips.ipynb`, vLLM, A100).
- **Local / Qwen3-VL-235B via OpenRouter** (`src/caption_openrouter.py`) — `qwen/qwen3-vl-235b-a22b-instruct`,
  no GPU, just `OPENROUTER_API_KEY` in `.env`. Per clip: CLAHE-enhance frames → score with
  `clip_mlp_hardneg_v2` → skip if no frame `p_visible ≥ PRESENT_MIN` (0.5) → send top-N clearest frames
  to the API. Flags `--index`, `--clips-root`, `--cap-key`, `--etho-key`, `--limit`. Resumable.
- **30B vs 235B comparison** lives in `data/octopus_clips_verified.json`: `caption` = 30B,
  **`caption_235b`** = 235B (written via `--cap-key caption_235b`, non-destructive).
- **KEY FINDING (2026-07-05):** running 235B over all 847 "verified" clips, **534 (63%) came back
  `octopus not present`** — the extraction massively over-extracts, almost all `Right_Left` reflections
  the CLIP detector fires on at `p_visible=1.0`. Takeaways: **drop `Right_Left` from `CAMERAS`**, and the
  VLM is a far better presence filter than the detector. (Right_Top/Back/Front are the real den angles.)

## Ethogram — now 7 classes
`data/ethogram_list_v2.json` (7 behaviors, reduced from the 19 in `ethogram_list.json` — 6 of which had
zero clips). Each has a `maps_from` list folding in the originals. All captioning/labeling/training uses
the 7-class sheet. Order: Resting / Exploration/manipulation / Crawling / Swimming/jetting /
Reaching out of water / Human/enrichment interaction / Colour change/defensive (+ `octopus not present`).

## Octopus segmentation — tiny mask model (in progress, 2026-07-21)
**Goal:** pixel-level octopus masks to (a) **clean up extraction** (mask-based presence beats the
CLIP gate on reflections; motion *inside the mask* ignores IR-lamp flicker) and (b) **enrich the
behavioural record for free** (body area = posture-spread, octopus-only colour, colour-change,
masked motion — the exact signals the affect model was missing). Plan: **`src/SEGMENTATION_PLAN.md`**.
**Design constraint (hard): smallest model that still gives good masks** — single class, low-res
input; GroundingDINO/SAM2 are the *teacher/auto-labeler ONLY*, never deployed.

- **Two-model teacher→student design.** Teacher (offline, slow): **GroundingDINO-tiny** (box) →
  **SAM2** (mask). Student (deployed, tiny, fast): a compact U-Net / LR-ASPP distilled from the
  teacher masks. Both teacher models run via `transformers` (`IDEA-Research/grounding-dino-tiny`,
  `facebook/sam-vit-base`) / the `sam2` package (`facebook/sam2.1-hiera-tiny`) — **already cached
  locally**, no compile needed. GroundingDINO deformable-attention is unstable on MPS → keep it on
  **CPU** (SAM2 runs on MPS/CUDA).
- **Phase 0 DONE — recipe validated (before/after on 4 cameras).** Per-frame box→SAM alone
  over-segments (IR: grabs bright metal tools; colour: bleeds into background from the loose box).
  **Fix = SAM2 *video propagation* seeded by GroundingDINO's most-confident frame** (temporal
  consistency) + largest-connected-blob cleanup + area-continuity check. Results: IR tool-bleed fixed
  (mask area 11.8%→6.5%), colour background-bleed fixed (15.5%→5.8%), clean colour no-regression,
  and the **reflection camera (Right_Left) is correctly rejected by low seed confidence** (~0.50 on a
  reflected human vs 0.74–0.89 on a real octopus). **Lesson: temporal consistency kills *transient*
  errors but not *consistent* ones (a reflection present every frame) — so gate on seed confidence
  AND drop Right_Left.** Demo before/after: `data/segmentation_demo/` + scratchpad `phase0_out/`.
- **`src/auto_segment.py` — the auto-labeler (Phase 0/1).** clips → `(frame.jpg, mask.png)` training
  pairs. Recipe: GroundingDINO per sampled frame → seed = argmax confidence; **GATE reject clip if
  best conf < `MIN_SEED_CONF` (0.60)**; SAM2 propagate both directions; keep largest blob; drop frames
  whose area is out of range or jumps >3× the clip median; emit `N_PER_CLIP` (4) clean frames + a
  `manifest.jsonl`. Device auto (cuda→mps→cpu), **resumable** (skips clips in the manifest), excludes
  Right_Left by default. Output `src/dataset_seg/vN/{images,masks}/`. Flags `--clips-root --out
  --cameras --limit`.
- **Local speed measured: ~4–5 min/clip** on the Mac (GroundingDINO on CPU ~2–3 s/frame is the
  bottleneck; SAM2 ~1.2 fps on MPS) → ~150 h for the ~2,168 colour clips. **So data extraction runs on
  GPU** (20–50× faster, SAM2 native CUDA) — see the A100 workflow below (~13 s/clip on the A100).
- **Clips available to label:** Right_Front 861 / Right_Right 731 / Right_Back 576 (colour) + Right_Top
  1391 (IR) + Right_Left 427 (reflection, excluded) = 3,986 verified clips (+2,380 in `octopus_clips_auto`).
- **`src/sample_seg_clips.py` — balanced-subset sampler (Phase 1a, stdlib-only, runs on any box).**
  Joins on-disk clips → index behaviour labels, drops `octopus not present` + Right_Left, water-fills a
  `--target` count evenly across behaviours (over-samples rare Swimming/Colour-change) then round-robins
  cameras. Writes `src/dataset_seg/sample_v1/sample_manifest.json`; `--stage` copies clips for upload.
  Colour-first default (Right_Front/Back/Right). Present colour pool = **1,824 clips**.
- **`src/train_segmenter.py` — the tiny-segmenter trainer (Phase 2).** Compact `TinyUNet` (4-level,
  width set by `--base-ch`: 8→0.13M / 16→0.5M / 32→2M params) on the `(image,mask)` pairs. Split **BY
  SOURCE VIDEO** (`date/segment`, no leakage), BCE+soft-Dice, eval IoU@0.5 / Dice / area-err vs the
  **0.85 colour bar**. Sweep `--base-ch` for the IoU-vs-size curve. Saves `weights/octo_seg_<ver>_ch<n>.pt`.
- **`src/segment_octopus.py` — inference module (Phase 2 deliverable + Phase 3 gate).**
  `OctoSegmenter(ckpt).segment(frame) -> (mask, area_frac)` (largest-blob cleanup). CLI saves an overlay.
- **A100 remote workflow (2026-07-22):** the auto-labeler runs on the GPU box **`amera-vllm-a100`
  (10.32.0.7, us-central1-f)**, reached by SSH key `~/.ssh/id_ed25519` from `amera-siddharth` (this box's
  service account lacks GCP compute/storage IAM, so gcloud/gsutil can't drive it — plain SSH + rsync over
  the internal VPC instead). Setup script installs venv (torch cu124 + transformers + `sam2` built with
  `SAM2_BUILD_CUDA=0`, no nvcc on box + ffmpeg). Sharded 3× by camera to use the GPU (~9.5 GB, models tiny),
  merged after. The `_C`-import warning from sam2 is benign (skips optional hole-filling). Smoke test:
  clean masks, area 3–10%, seed conf 0.75–0.89.
- **Phase 1-2 RUN DONE on the A100 (2026-07-23).** Auto-labeled 1,824 colour clips →
  **4,412 (image,mask) pairs / 77 videos** (`data/dataset_seg/v1`). Trained the tiny segmenter (sweep +
  aug + IR + negatives). Full trail in **`src/SEGMENTATION_LOG.md`**; logs/overlays in `results/segmentation/`;
  weights in `weights/seg/` (local, gitignored). Headline results:
  - **Mask pixel-IoU bar (0.85) NOT met.** At this point it plateaued **~0.47** across
    TinyUNet(ch8/16/32), LR-ASPP(pretrained), strong augmentation, and +IR, and was diagnosed as a
    video-diversity gap (62 train videos; train 0.68 / val 0.47; fails by *mislocating* a right-sized
    blob). **⚠️ DO NOT QUOTE 0.47 AS THE CURRENT RESULT — it is ~0.15 stale.** That number is a
    *training-time val IoU against auto-labelled TEACHER masks*. The current model scores
    **IoU 0.6415 mean / 0.7193 median against HUMAN masks** on the frozen SEG-TEST suite
    (`data/benchmarks.json` tag `paper_current`, ckpt `octo_seg_thin768_lraspp.pt`). Always quote the
    frozen-suite number; see the "Current segmentation results" block below.
  - **IR (`Right_Top`) unusable as-is:** GroundingDINO low-confidence on greyscale (13% clip acceptance) +
    SAM2 over-segments bright tools (mask area median 8.5% vs colour 2.9%). Needs the Phase-0 IR fix first.
  - **DEPLOYMENT WIN — the presence gate works once you train with NEGATIVES.** v1 (positives-only) was a
    random presence detector (AUC 0.50, fired on reflections). **v3 = 4,412 pos + 1,388 empty-mask negatives
    → AUC 0.86 overall, 0.99 vs reflections** (reflection mask area → 0.000). At area≥0.01: 88% present-recall,
    **0% reflection-FP**. This beats the CLIP gate on the Right_Left/reflection false-positives. Deployable
    model: `weights/seg/octo_seg_v3_lraspp.pt` (LR-ASPP, 3.2M params). New scripts: `src/eval_presence.py`
    (presence AUC eval), `src/build_v3.py` (negatives dataset). `segment_octopus.py` now loads either arch.
- **Diversity retrain DONE on Modal (2026-08-07).** Attacked the plateau's root cause (only 62 training
  videos) with the diverse-footage harvest: **530 clips / 276 videos / 149 dates** on the Modal volume →
  auto-labeled on an A10G (**178 accepted / 732 pairs**, 345 low-conf recoverable) → merged with old v1 and
  retrained. **old(62 vid)=0.468 → new-only(100 vid, 732 pairs)=0.245 (overfits, too few frames) →
  merged(176 vid, 5,144 pairs)=0.494** (best, on a HARDER diverse-date val -- again a TEACHER-mask val
  number, not comparable to the 0.6415 human-mask figure above). New app **`src/modal_seg_train.py`**
  (A10G; `autolabel`+`train`, computes on the volume so no clip transfer; `--ds` accepts a comma-sep list →
  symlink-merge). Best model: **`weights/seg/octo_seg_merged_lraspp.pt`**. Full trail in SEGMENTATION_LOG.md.
  **Refined diagnosis: the ceiling is now teacher-label quality, NOT data** — merged val plateaus flat at ~0.49
  with no overfitting (loss keeps falling), so a student can't beat the noisy GD+SAM2 masks it learns from.
- **TEACHER vs HUMAN masks MEASURED (2026-08-20) — `src/eval_teacher_masks.py`, PAPER_NOTES R19.** The
  gap is closed: on SEG-TEST's 122 human-mask frames the **per-frame zero-shot teacher scores IoU 0.374
  vs the student's 0.6415** (paired Δ −0.2675 [−0.313, −0.136], clustered by source video). The student
  reproduced its published 0.6415 exactly, validating the harness. **But the conditional split is the
  real result:** when GroundingDINO clears its own `MIN_SEED_CONF` 0.60 gate (only 21/122 frames) the
  **teacher WINS, 0.726 vs 0.657**; it finds nothing at all on 25% of frames and 83% fall below the gate.
  The teacher is **high-precision/low-recall, the student uniformly competent** — distillation turned a
  sparse high-quality signal into dense coverage. **So "teacher-label quality is the ceiling" survives and
  sharpens: the student's 0.6415 sits ~0.08 under the teacher's 0.726 operating-point quality, so more
  clips will not move the plateau.** Do NOT quote 0.374 as the quality of the labels the student trained
  on — those used SAM2 propagation from the most-confident frame and are better; measuring those needs
  122×40 = 4,880 GD calls (~3.4 h locally) and has not been run.
### Current segmentation results — QUOTE THESE, from the frozen suite (`data/benchmarks.json`)
SEG-TEST = 122 human-drawn mask frames / 5 held-out videos + 19 empty-tank negatives.

| tag / ckpt | mask IoU mean | median | area err | presence AUC |
|---|---|---|---|---|
| `paper_current` / **`octo_seg_thin768_lraspp.pt`** (headline) | **0.6415** | **0.7193** | 1.05% | 0.794 |
| `clean512tv` / `octo_seg_clean512tv_lraspp.pt` | 0.6075 | 0.6661 | 1.07% | 0.718 |
| temporal fusion EMA (`fuse_ema`) | 0.5471 | 0.600 | 1.00% | **0.9685** |
| temporal fusion flow (`fuse_flow`) | 0.5109 | 0.5505 | 1.06% | 0.9495 |

SKEL-50 (50 frozen frames / 20 videos): tip-F1 **0.539** (precision 0.722, recall 0.502), 3.68 arms/frame.

- **The 0.85 bar is unmet but pixel-IoU may be the wrong metric for the goal.** AREA ERROR IS ~1%, and
  the downstream needs are presence + body-area (posture) + masked motion — all area-based. Boundary
  quality (thin tentacles) is the weak axis and nothing downstream reads boundaries.
- **Temporal fusion trades mask for presence:** EMA costs 0.10 IoU and buys **+0.17 presence AUC**
  (0.794 → 0.9685). Optical flow is worse on both, so plain smoothing — not motion compensation — is
  the mechanism. Use EMA when the output is a presence gate, single-frame when it is a mask.
- **RETRACTED — "teacher-label quality is the ceiling" was WRONG** (2026-08-08). The evidence for it
  (0.49 → 0.70 with HQ labels) was **train leakage**: those frames were in the model's train set. Clean
  held-out is FLAT across teacher quality: tiny 0.494 → 14%-HQ 0.508 → 100%-HQ 0.506. Upgrading the
  teacher (GroundingDINO-base + SAM2-large) bought nothing. Do not restart that chain.
- **What actually worked, measured leak-free:** input resolution **256² → 512²** (at 256² the median
  octopus is ~40×40 px and tentacles are 1–2 px, i.e. unrepresentable) plus **Focal-Tversky loss**
  (β>α punishes the under-segmentation that symmetric Dice+BCE ignores) → **0.466 → 0.608**, then 768²
  → 0.6415. Also: **volume beat purity** — 290 clean human frames give 0.505, a blend of 8% human +
  92% auto over 3,450 frames gives 0.608.
- **Leakage guard:** `--holdout-videos` forces the test videos out of ALL training sources. The first
  human-val comparison was contaminated because `old_hq` shared source videos with the human labels.
- **Next:** wire the `segment_octopus` area-gate (≥~0.01) into `extract_octopus_clips.py` /
  `local_pipeline.py` — it beats the CLIP gate on reflections and is still not deployed. Phase-0 IR fix
  for the ~1,391 IR clips (the colour-trained model over-segments bright tools on IR, so IR is
  currently excluded downstream). A temporal student, since per-frame mislocalisation is the failure
  mode. NOT more teacher labels or more clips — both exhausted and measured.

## Skeleton / pose pipeline — mask → anatomical graph → kinematics (2026-08-13)
Downstream of segmentation: **`src/skeleton/`** (adapted from the external Skeleton repo; critical
`Skeleton_best` NameError + head-edge mask-constraint + degenerate-head bugs fixed) turns a binary
octopus silhouette into a **mantle/head/8-arm graph** (Zhang–Suen thinning → Dijkstra arm paths →
mask-constrained splines). Fast thinning needs `opencv-contrib-python` (installed; replaces
opencv-python — same 4.13 version, superset).
- **`src/segment_to_skeleton.py`** — clip → seg masks (EMA-smoothed) → FIXED union-bbox crops →
  skeleton. `segment_masks(..., keep_small=W)` also returns small colour frames; `grey_crops()` builds
  the aligned grey crops the flow prior needs.
- **`src/skeleton/multi_frame.py` — the tracker.** `tracked_sequence(crops, greys=..., seed="best",
  method="chain")`: per-frame detect once → seed at the best-resolved frame → bidirectional chain via
  `temporal_fit` with a **per-node optical-flow prior** (DIS + fwd-bwd check; when the prior validates
  the acceptance gates TIGHTEN 0.6× — a better prior with unchanged gates measurably admitted MORE
  noise). Every node carries **`state: detected|fitted|occluded`**; `compute_motion` only emits rows
  from evidence-backed samples (occluded holds never become motion) + per-node median+gaussian
  trajectory smoothing. `method="global"` (tracklet linking) exists but measured WORSE — see log.
- **Skeletonization phases (measured):** thin-preserving mask-prep (+0.17 arms), **best-frame seeding
  +3.67 median arms (2.17→5.83, the big lever)**. Tracking v2 on a fixed 10-clip eval
  (`src/skel_eval_tracking.py`, metrics `src/skeleton/track_metrics.py`): baseline teleport 14.85% →
  flow2 14.27% (adopted); **occluded_frac 41.7%** = ~half of arm samples were evidence-free holds,
  now excluded from kinematics. Full trail in `SEGMENTATION_LOG.md`.
- **`src/batch_skeleton_motion.py`** — batch: clips → per-clip kinematics summary (tip/mantle speed,
  activity, arm-spread, occluded_frac) → `data/skeleton_motion.json` + `--merge` writes a `kinematics`
  block into `behaviour_records.json`. IR (Right_Top) clips auto-skip (colour-trained seg model).
- **`src/seg_skeleton_pipeline.py` + `ui/seg_skeleton_viewer.py` (8017)** — one video → 3 synced
  overlay mp4s (raw seg / smoothed seg / skeleton+trails). **`ui/skel_diag_viewer.py` (8018)** —
  phase-results browser (reads `data/skel_diag/summary.json` + chart, meta-driven).
- **Caveat:** speeds are px/s in crop space (no px→cm calibration); arms are silhouette-limited
  (mask quality costs ~1.7 arms/frame vs human GT — `src/skel_diagnostic.py`).

## Diverse-footage harvest — the data-gen fix (in progress, 2026-07-23)
**Why:** both students are footage-diversity-limited — the whole corpus was **7 dates (one week)**.
Plan: `src/DATA_PLAN.md`. Running results ledger: `PAPER_NOTES.md`.
- **The server has WAY more than we used.** It exposes HTML directory listings (crawlable). 6 collections /
  ~5 animals: **Nity ~209 days** (`O-vulgaris-Nity-2025-9-17--` 157d + `-2026-2-20--` 52d) + others
  (Heidi 155d, a 2024 vulgaris 122d, Maya 48d, Eledone 1d). Creds in `.env` as `OCTOPUS_USER`/`OCTOPUS_PASS`.
- **Network reality (measured on Colab):** server download is **~5 MB/s and PARALLELISM BARELY HELPS**
  (1→5 streams = only 1.6×; near-total server-side cap). So: **one CPU box, 2–3 stream workers, no GPU**
  (GPU idles on a network-bound job — costs 10–50× for zero speedup). Stream-scan is viable
  (~5 video-sec/s), so we **stream + early-exit, never bulk-download** (30 days of full videos = ~1 TB).
- **`src/harvest_stream.py` — THE harvester (self-contained).** Crawls listings for the Nity colour
  cameras (Right front/back/right; excludes Right_Left reflections + IR Right_Top), and per video:
  **probe-first** (`N_PROBES`=10 cheap input-seek frames; if none reach `p_visible≥PROBE_THRESH` 0.50,
  skip the full scan) → else stream 1fps CLIP+MLP (`clip_mlp_hardneg_v2.pt`, letterbox) → **VISIBILITY-only
  gate** (`REQUIRE_MOTION=False` — still-but-visible octopus is good seg/caption data; motion still
  recorded) → keep 2 windows spread `SPREAD_SEC`=60s apart → **early-exit at 2**, extract via ffmpeg
  byte-range from the URL. Validation A/B (same 8 vids): motion-gate 1 clip-video/2 clips → **visibility-gate
  4 clip-videos/8 clips**. Sampling for VIDEO diversity (`MAX_SEG_PER_DAYCAM`=3). Device auto.
- **ONE resumable ledger** `harvest_ledger.json` keyed by `video_url` = the tracker AND a **detailed
  coverage report** per video: `duration`, `probe_points [[t,p]…]`, `probe_max_p`, `status`
  (`clips`/`probed_empty`/`scanned_empty`/`failed`), `coverage`, `scanned_sec`, **`unscanned_sec`**,
  `discard_reason` — so any skipped/partial video can be mined for MORE later. Clips also emitted in
  `octopus_clips_verified.json` entry format (`harvest_clips_index.json`) for merge. Resumable (skips
  video_urls already in ledger).
- **`src/modal_harvest.py` — Modal app (CPU, `sidraj` profile).** Image = ffmpeg+git / torch+torchvision+
  `openai-clip`+`setuptools<81` (needed: latest setuptools drops `pkg_resources` which openai-clip imports).
  Secret `octopus-creds`, Volume `octopus-harvest-vol` (periodic `vol.commit` = resumable across timeouts).
  Run: `MODAL_PROFILE=sidraj modal run --detach src/modal_harvest.py --workers 2 --max-scan-sec 400`
  (**must be `--detach`** — a foreground `modal run` dies if the local client's gRPC heartbeat drops).
  Fetch: `modal volume get octopus-harvest-vol /harvest ./harvest_dl`.
- **Full run launched 2026-07-23:** ~209 Nity colour days = **1,769 videos**, probe-first + visibility gate.
  Projected ~6–12 h (probe-first cut it from ~37 h; now bounded by probe-seek latency + the ~5 MB/s cap).
- **`src/colab_speedtest.py`** — the server bandwidth / stream-scan / parallel-speedup probe.

### cbox — the always-on harvest box (2026-08-19)
Second harvest campaign runs on **`cbox` = SSH host `claude-box`** (Oracle free-tier A1, `ubuntu@129.158.193.16`,
key `~/.ssh/oracle_box`). Checkout at **`/home/ubuntu/project/cephalopods`** (same repo/branch `sid-dev`,
venv at `./venv`, torch 2.12.1 CPU — `cuda False`). Naming pattern: `abox`/`vbox`/`cbox` in `~/.ssh/config`.
- **Box reality: aarch64, 2 cores, 11 GB RAM, ~13 GB free disk.** Measured **CLIP ViT-B/32 = 7.9 frames/s**,
  so a full 1,796-frame scan ≈ 227 s of pure CLIP. **Extra workers do NOT multiply scan throughput** (only
  2 cores — they split it); they only hide probe/network latency. Use `--workers 2` (also respects the
  server-throttling rule: concurrency 2–3).
- **`harvest_stream.py` reads creds from ENV, not `.env`** — it does not call `server_creds.py`. Always run as
  `set -a; . ./.env; set +a; ./venv/bin/python3 -u src/harvest_stream.py ...`. cbox's `.env` holds ONLY
  `OCTOPUS_USER`/`OCTOPUS_PASS` (0600, gitignored); the OpenRouter/W&B keys were deliberately not copied.
- **Seed the ledger before running.** Copy the prior `harvest_ledger.json` into the new `--out` dir; the
  harvester skips any `video_url` already in it, so past work is never redone. Raising `--max-seg-per-daycam`
  still yields NEW urls on already-visited dates, so deeper sampling stays available.
- **Run 2 target: `O-vulgaris-Nity-2026-2-20--/`** — the untouched second Nity collection, crawls clean
  (3 colour cams, **52 dates 2026-02-20 → 2026-04-12**, ~429 videos at 3 seg/day-cam). Extends the record
  ~5 weeks past the analysed corpus (which stops 2026-03-07). Run **uncapped** (`--max-scan-sec 0`), unlike
  Modal's 400 s cap — probe-first discards ~59% cheaply and early-exit bounds productive videos, whereas the
  cap only ever inspects the first 6.7 min of a 30-min video. Launch detached:
  `tmux new-session -d -s harvest "... > /tmp/harvest_cbox.log 2>&1"`; poll that log.
- **Still untouched on the server** (exact listing names): `GP11-Heidi-menu/`, `Heidi-additional-videos/`,
  `O maya 2025-05_2025-06/`, `O vulgaris 2023-10_2024_08/`, `O eledone 2024-10_2024-11/`. These are
  **different animals** — a cross-animal generalization claim, not an extension of Nity's time-series.
- **THREE TRAPS hit on the first cbox launch (2026-08-19) — check all three before any new box/campaign:**
  1. **`cv2` is imported LAZILY inside `scan_stream` (line ~137), so a missing opencv fails ONLY after a
     video passes the probe** — i.e. it silently destroys exactly the promising videos while
     `probed_empty` ones sail through. First run: 3/13 `failed` with `ModuleNotFoundError: No module
     named 'cv2'`, and all 3 had `probe_max_p >= 0.5`. Nothing imports cv2 at module scope, so the
     harvester starts up perfectly fine. `opencv-python-headless` is now in `src/requirements.txt`
     (cbox has cv2 5.0.0). **Always smoke-test a video that PASSES the probe**, not just any video.
  2. **Resume skips by PRESENCE, not status** (`plan = [it for it in plan if it[3] not in ledger]`), so a
     `failed` entry is skipped *forever* and its video is never retried. After fixing any crash-type bug,
     **purge `status=="failed"` from the ledger before restarting** or the losses are permanent.
  3. **The ledger cannot see the original corpus.** It is keyed by `video_url`, but the 1,117-video
     original corpus was processed from LOCAL downloads (`data/aquarium/full/<date>/<segment>/<Cam>.mp4`),
     so those keys are absent and the harvester happily re-scans the same physical recordings. Measured:
     **48 of the 426 planned videos were the exact same (date, segment, camera) already processed** —
     the 2026-2-20 collection covers 2026-02-20..04-12 and the corpus covers 2026-02-20..03-07. Fix used:
     join the plan against `src/octopus_clips_processed.json` on `(date, segment, camera)` and pre-seed
     those urls into the ledger as `status="skipped_already_analysed"` with a `discard_reason`, which the
     existing skip logic then honours (38 seeded; the other 10 had already been probed). Re-running them
     would have produced near-duplicate clips and quietly undermined any "diverse footage" claim.
- **Measured end-to-end scan rate on cbox: ~2.9 frames/s** (network + decode + CLIP together, vs 7.9 f/s
  for CLIP alone), so a full 1,796-frame scan is ~10 min, not the ~4 min pure-CLIP arithmetic suggests.
  A productive video costs less thanks to early-exit (self-test: 2 clips, 256 s scanned, 159 s wall).
- **Two-box split of ONE collection — `--date-min` / `--date-max` / `--date-reverse` (added 2026-08-19).**
  The plan is built ONCE at startup, so seeding a ledger mid-run does **not** stop a running box from
  reaching those videos — the split must be in the plan itself. Give box A `--date-max D` and box B
  `--date-reverse --date-min D+1` and they work toward each other over provably disjoint footage
  (unit-tested with the network stubbed: zero overlap, union == unfiltered plan). Dates are ISO so
  string compare is chronological. Also seed box B's ledger from box A's current ledger, since A may
  already have covered part of B's window (it iterates **camera-major**: all dates of Right Back, then
  Right Front, then Right Right — so its frontier is a camera, not a date).
  Live split 2026-08-19: cbox `--date-max 2026-03-20` (146 videos, 2 workers) + Mac
  `--date-min 2026-03-21 --date-reverse` (1 worker, MPS), out dir `data/harvest_local`.
- **Is a second box worth it? Measured, not assumed.** The Mac pulled **1.73 MB/s single-stream while
  cbox was running**, and cbox did **not** slow down (recent-20 rate 1.07 vs 0.93 videos/min average) —
  so the server had headroom. But at the ~0.425 MB/s footage bitrate that bandwidth feeds only ~4
  video-seconds/s, i.e. **~4 frames/s — barely above cbox's 2.9**. A second box is therefore
  **network-bound, not compute-bound**: expect ~+50%, NOT 2×, and MPS mostly idles. Keep TOTAL streams
  at 3–4 across all boxes (memory `server-throttling-sustained-load`: the server collapses to ~4 KB/s
  under sustained multi-worker load), which is why the Mac runs 1 worker, not 2.
- **`pkill -f harvest_stream` / `pgrep -f harvest_stream` MATCHES YOUR OWN SSH COMMAND LINE.** `pgrep`
  gives a false "STILL RUNNING"; `pkill` actually killed the shell that was starting the new tmux
  session, so the relaunch silently died and the old log made it look like nothing happened. Kill by
  PID, or grep a string that cannot appear in your own command (`ps -eo pid=,args= | grep "venv/bin/
  python3 -u src/harvest_stream" | grep -v "grep\|bash -c"`), and always verify via `/proc/<pid>`.
- **`src/merge_harvest.py` — pull results back.** rsync the run dir home, then merge into the canonical pair
  `data/harvest_ledger_all.json` + `data/harvest_clips_index.json` (keyed by `video_url`; a record is only
  overwritten by a strictly more informative one — real status beats `failed`, more clips beats fewer).
  **It deliberately does NOT pool into `data/octopus_clips_verified.json`** — that index backs the paper's
  frozen benchmark sets, and harvested clips use a different sampling regime (visibility-only gate, 2/video),
  so merging would silently change reported denominators. There is no flag to do it — pooling is a
  deliberate separate step that must be followed by re-running `src/benchmarks.py`.

## Benchmarks — the frozen suite (READ `BENCHMARKS.md` BEFORE CLAIMING ANY IMPROVEMENT)
Every improvement claim, and every number in the OCEANS 2026 paper, is measured by **`src/benchmarks.py`**
on frozen sets; results append to `data/benchmarks.json` keyed by `--tag`, and `--latex` regenerates
the paper's table. Suites: **SEG-TEST** (122 human-mask frames / 5 held-out videos + 19 empty-tank
negatives), **SKEL-50** (50 frozen frames / 20 videos, headline = arm-tip F1), **TRACK-10** (10 clips),
**REFL-24** (reflection rejection, `src/eval_reflection_presence.py`). Rules: frozen sets are never
regenerated to suit a result; splits are **by source video, never by frame**; holdout videos are excluded
from *every* training source; negatives of different kinds are **never pooled**; report negatives.
- **`src/temporal_fusion.py`** — test-time fusion of the segmenter probability map, modes
  `none|ema|flow|median` (`median` = the unwarped control for `flow`). NOTE the alignment trap:
  `seed_frame` indexes the labeller's `ffmpeg fps=2, scale=min(1024,iw)` list, NOT raw video frames, so
  neighbours are regenerated with that identical extraction and asserted to match before use.
- **`src/fusion_threshold_sweep.py`** — caches one probability map per frame per mode, then sweeps the
  binarisation threshold, so arms are compared at their OWN best operating point (a fused median map is
  not calibrated like a single-frame map; comparing at a fixed 0.5 would fake a result either way).
- **`src/reflection_negatives.py`** — samples + stages Right_Left frames for review. **Right_Left is not
  a pure-reflection camera: ~10–20% of its frames contain the real animal**, so frames must be reviewed
  before being scored as negatives, and ambiguous ones excluded rather than assumed empty.
- Measured so far: temporal fusion **hurts mask IoU** (0.642 → 0.547 ema / 0.511 flow) but **helps
  presence AUC a lot** (0.794 → 0.969 ema / 0.950 flow); plain EMA beats optical flow on both, so motion
  compensation is not the mechanism. thin768 rejects reflections (AUC 0.917) *better* than it rejects the
  empty tank (0.794) — the assumed failure mode is backwards. Full trail in `PAPER_NOTES.md` R8/R9.

## Distillation students
- **Behavior classifier** (`train_behavior_student.py`, local): frozen CLIP feats (mean+max pooled) → MLP,
  copies the v2 labels. **Failed** — 45% val acc, *below* the 50% majority-class baseline; per-class F1≈0
  on everything but presence. Lesson: static pooled features can't classify behavior — it needs
  **temporal/motion features**, not more labels.
- **Caption student** (`train_caption_student.ipynb`, Colab): QLoRA fine-tune of Qwen2.5-VL-3B distilling
  the teacher captions; `demo_video_to_captions.ipynb` runs base vs LoRA on a fresh video.
  - **Trained adapter (v1, DONE 2026-07-15):** `Qwen3-VL-2B` + LoRA r16/α32, 3066 train / 392 val,
    576 steps. Eval (50 held-out val): base emb-sim 0.702 / rougeL 0.269 → **LoRA 0.834 / 0.455**.
    Adapter on Drive `GSOC-Catrobat/caption-student/lora_out/qwen3vl2b_caption_v1(.zip)`.
  - **Local 4-bit deploy (MLX, runs on the 16GB Mac, ~3s/caption, no CUDA):** merge adapter→fp16 base
    then convert with mlx-vlm. Recipe (`scratchpad/merge_adapter.py` + `mlx_vlm.convert -q --q-bits 4
    --q-group-size 64`). Outputs `models/qwen3vl2b_caption_v1_merged/` (fp16, 4.3G) and
    `models/qwen3vl2b_caption_v1_mlx_4bit/` (**1.7G, the deliverable**). bitsandbytes NF4 is CUDA-only
    so it does NOT run on Mac — MLX is the Apple-Silicon 4-bit path (`mlx-vlm` 0.6.3 supports `qwen3_vl`).
    **Two config patches needed** (transformers 5.x vs mlx-vlm drift, else convert fails): in the merged
    `config.json`, (1) add flat `text_config.rope_theta` + `rope_scaling` from `text_config.rope_parameters`;
    (2) set `vision_config.model_type` `qwen3_vl_vision`→`qwen3_vl`. Test with `mlx_vlm.load`+`generate`.

## Review / labeling UIs (FastAPI, local)
- `ui/review_captions.py` (8005) — approve/reject/edit caption+label, writes into the index.
- `ui/compare_captions.py` (8007) — v1 vs v2 caption A/B (local video).
- `ui/label_captions.py` (8008) — **blind** A/B labeling (v1/v2 shown as anonymous Options A/B) → builds
  `data/caption_training_set.json` (human ground truth; `caption_source` = v1/v2/human).
- `ui/compare_base_lora.py` (8009) — base vs LoRA captions, streams clips from the server.
- `ui/review_hardneg.py` (8004) — hard-negative frame review.
- `ui/local_pipeline_app.py` (8010) — **local pipeline UI (video → clips → captions with timeline)**. Pick a
  suggested server video (ranked by # present-octopus clips from the index) or type a local video path; runs
  `src/local_pipeline.py` (one job at a time, models loaded once) and streams each extracted clip inline next
  to its caption + `mm:ss-mm:ss` timeline. Endpoints: `/api/suggestions`, `POST /api/run`, `/api/status`,
  `/clip`. Run: `venv/bin/python3 ui/local_pipeline_app.py` → http://localhost:8010.
- `ui/demo_player.py` (8011) — **demo player: full video LEFT, synced captions RIGHT**. Pure viewer (no models):
  reads pre-processed local demos from `data/demo_videos/` (each `*.mp4` + `*_captions.json` made by
  `local_pipeline.process_video(..., save_clips=False)`). Tabs pick a video; the 30-min video plays on the left
  (served via range-capable `FileResponse` → seekable) while its captions list on the right — click a caption to
  jump the video there, and the active caption highlights + auto-scrolls as it plays. The `data/demo_videos/` set
  is ~5–6 good-motion segments (ranked by index `mean_motion`). Also has an **⬆ Upload a video** button:
  `POST /api/upload` saves the file into `data/demo_videos/` and runs the pipeline on it (lazy-loads models on
  first upload, one job at a time; `save_clips=False`), polled via `/api/upload_status`; the finished video shows
  up as a new tab with its captions. Needs `python-multipart`. Run: `venv/bin/python3 ui/demo_player.py` →
  http://localhost:8011.

## Public release repo — `sidraj000/octopus-behaviour`
The paper's companion release is a **separate public repo** (`github.com/sidraj000/octopus-behaviour`,
local checkout `~/Documents/my-projects/octopus-behaviour`): code + ethogram/caption labels + the paper;
models and the big datasets on Drive. Apache-2.0 (code) / CC-BY-4.0 (labels).
- **NEVER publish `AGENTS.md` there.** It is an internal runbook: it named the footage-server password
  in plaintext and the address + SSH key of a compute box. It *was* pushed there on 2026-08-23 and had
  to be scrubbed with a history rewrite (see the credential note below). Publish `docs/PAPER_NOTES.md`
  as the experimental record instead.
- Sanitize box hostnames/IPs out of anything else published (`SEGMENTATION_LOG.md` named the A100 box).

## Credentials
**The footage-server password leaked publicly on 2026-08-23** (in `AGENTS.md`, pushed to the public
release repo for ~20 min; scrubbed by history rewrite, but GitHub still serves orphaned commits by SHA,
so treat it as compromised). **It must be rotated server-side** — a rewrite is not a fix. Until then
assume `repo.octopus-intelligence.org` creds are public.

`.env` (repo root, gitignored) holds `OCTOPUS_USER`/`OCTOPUS_PASS` (footage server) and
`OPENROUTER_API_KEY` (captioning). Never commit any of these; only `.env.example` files are tracked.
Notebooks read creds via getpass/env; scripts via `server_creds.py` / env.

## Dataset layout
- `data/frames/{visible,hidden}/` — training frames (~3970 visible / ~5.6k hidden after 2026-06 mining) + manifest.
- `data/octopus_clips_verified/{date}/{segment}/{Camera}_{start}-{end}.mp4` — curated behavior clips.
- `data/scanned_frames/`, `data/hard_negatives/`, `data/saliency/`, `data/motion_debug/` — pipeline outputs.

## Conventions
- Notebooks are often built programmatically via builder scripts in the scratchpad
  (`json.dump` + `compile()` syntax-check per cell) rather than hand-edited JSON.
- Right-side cameras (Right_Front, Right_Back, etc.) are the relevant angles for Nity's den.
- Long-running scans/retrains: run in background, log to `/tmp/<name>.log`, poll the log.

## Working agreement (IMPORTANT)
- **After every meaningful chunk of work, commit.** Don't leave a pile of unrelated
  changes uncommitted — group a coherent change, write a clear message, and commit it.
- **Keep this file (`AGENTS.md`) up to date.** Whenever something changes that a future
  agent needs to know — new active model/weights, a new script or notebook, a pipeline
  step, a bug+fix, a changed default, a verification result — add or update it here in the
  same session. `CLAUDE.md` is a symlink to this file, so one edit updates both. Treat
  `AGENTS.md` as the living source of truth, not just first-time onboarding docs.

## Persistent memory
Cross-session facts live in
`~/.claude/projects/-Users-siddharthraj-Documents-my-projects-sentiment-analysis/memory/`
(index: `MEMORY.md`). Key entries: project status, motion-gate bug, ethogram extraction,
VLM captioning, detection pipeline, clustering results. **Note**: CLIP-based octopus
*detection* (zero-shot CLIP scores, exp06 classifier) was invalidated 2026-06-13 — unreliable;
the CLIP+MLP *probe* above is the working approach.
