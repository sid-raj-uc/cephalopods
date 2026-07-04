# src/ — octopus clip-extraction pipeline

The clean, canonical extraction pipeline. Given remote aquarium videos, it finds
20-second clips where the octopus is **visible AND moving**, extracts them, and
records them in a JSON index.

## Files
- **`extract_octopus_clips.py`** — the pipeline. Per video: octopus detection
  (CLIP ViT-B/32 + MLP probe, letterbox) + motion detection (`scan_motion_area`,
  absolute changed-pixel fraction) at 1 fps, slide a non-overlapping 20s window,
  keep it when **>50% frames have `p_visible ≥ 0.6` AND mean motion ≥ 0.008**,
  extract via ffmpeg byte-range copy.
- **`motion_detector.py`** — `scan_motion_area()` motion detection.
- **`octopus_clips_verified.json`** — the **clip index** (one entry per extracted
  clip: video_url, time range, scores, clip_path). Extraction fills the clip
  metadata; **captions/labels are filled in later by a separate captioning script.**
  Re-running the extractor appends new clips and preserves existing entries' captions.
- **`octopus_clips_processed.json`** — the **processed-video ledger** (which videos
  have already been scanned, so they're never reprocessed). Resumable.
- **`caption_octopus_clips.ipynb`** — the **separate captioning step** (Colab/GPU,
  Qwen3-VL-30B via vLLM). Fills `caption` + `ethogram_label` (7-class
  `ethogram_list_v2.json`) into `octopus_clips_verified.json`. Enhances frames (CLAHE),
  picks the clearest frames via `clip_mlp_hardneg_v2` `p_visible`, skips no-octopus
  clips, allows `uncertain`. Resumable; skips clips already captioned or `review==approved`.
  Run on Colab: upload the JSON + clips zip + `ethogram_list_v2.json` + the detector
  weight to Drive, caption, copy the JSON back into `src/`.
- **`train_caption_student.ipynb`** — the **caption-student training** step (Colab/GPU).
  LoRA (4-bit QLoRA) fine-tune of `Qwen2.5-VL-3B-Instruct` that distills the teacher
  captions in `octopus_clips_verified.json` into a small student model; saves the LoRA
  adapter (zipped) back to Drive. This is the final "train the LoRA adapter" step.

## Pipeline order
1. `extract_octopus_clips.py` → clips + `octopus_clips_verified.json` metadata (local).
2. `caption_octopus_clips.ipynb` → fill `caption` + `ethogram_label` (Colab, Qwen3-VL-30B teacher).
3. `train_caption_student.ipynb` → LoRA-fine-tune the small Qwen2.5-VL-3B student on those captions (Colab).

## Depends on (kept at repo root, not copied here)
- `weights/clip_mlp_hardneg_v2.pt` — the octopus detector.
- `server_creds.py` + `.env` (`OCTOPUS_USER`/`OCTOPUS_PASS`) — footage-server creds.
- `data/octopus_clips_verified/{date}/{segment}/*.mp4` — where extracted clip mp4s land.

## Run
```bash
venv/bin/python3 src/extract_octopus_clips.py --limit 5          # smoke test
venv/bin/python3 src/extract_octopus_clips.py --date 2026-02-20  # one day
venv/bin/python3 src/extract_octopus_clips.py                    # all unprocessed
```
Flags: `--limit`, `--date`, `--motion-thresh`, `--visible-frac`, `--vis-thresh`.
