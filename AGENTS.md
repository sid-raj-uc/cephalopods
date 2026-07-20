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
     scores never separated the classes. Do not trust OWLv2 alone for this.
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

## Credentials
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
