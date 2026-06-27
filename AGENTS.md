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

> **USE `clip_mlp_best.pt` ALWAYS, FOR NOW.** It is the latest model — letterbox
> preprocessing + 66 verified hard negatives folded into `hidden/` — and the current
> default for all inference/clip-extraction work. Do not switch the active model
> without an explicit instruction.

All are CLIP ViT-B/32 + `mlp_256_64` probes unless noted. **Headline accuracies are NOT
directly comparable** — each was scored on a different test set (different preprocessing /
label cleanliness), so a higher number does not mean a better model.

| File | Use | Preprocessing / data | Acc | Trust |
|------|-----|----------------------|-----|-------|
| `clip_mlp_best.pt` | **DEFAULT — use this** (letterbox + 66 verified hard negs; honest, harder test set) | letterbox + 66 hard negs | 96.3% | ✅ |
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
  The clean, consolidated pipeline: octopus detection (`clip_mlp_best.pt`, letterbox) + motion
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
- `phase2/octo-clip-extraction/caption_octopus_clips.ipynb` — **Colab/GPU** captioner (Qwen3-VL-30B
  AWQ via vLLM, A100). Reads `octopus_clips_verified.json`, samples frames from each clip, and writes
  a one-sentence `caption` field back onto each clip entry (also `captioned_at`, `caption_model`).
  **Resumable** — skips entries that already have a `caption`; saves the JSON after every clip.
  To run: zip the `octopus_clips_verified/` folder to `octopus_clips_verified.zip` + copy
  `octopus_clips_verified.json` to Drive (`MyDrive/GSOC-Catrobat/`), caption on Colab, then copy the
  updated JSON back into `data/`. Successor to `phase2/exp23_caption_colab.ipynb` (which used a flat
  clip dir + separate captions json; this writes into the index instead).

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

## Dataset layout
- `data/frames/{visible,hidden}/` — training frames (currently ~3970 visible / 3982 hidden) + manifest.
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
