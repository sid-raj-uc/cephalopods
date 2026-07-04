# Octopus Clip-Extraction & Captioning Pipeline

Turn continuous aquarium camera footage of an octopus (*Octopus vulgaris*, "Nity")
into short, behavior-labeled, captioned clips — automatically.

Given remote videos, the pipeline finds 20-second windows where the octopus is
**both visible and moving**, extracts them, captions them with a vision-language
model, and distills that into a small student model.

## Pipeline (3 stages, in `src/`)

| Stage | File | Runs on | Output |
|-------|------|---------|--------|
| 1. **Extract** clips | `src/extract_octopus_clips.py` | local (CPU/MPS) | clips + `octopus_clips_verified.json` |
| 2. **Caption** clips | `src/caption_octopus_clips.ipynb` | Colab GPU (Qwen3-VL-30B teacher) | `caption` + `ethogram_label` in the JSON |
| 3. **Train student** | `src/train_caption_student.ipynb` | Colab GPU (Qwen2.5-VL-3B + LoRA) | LoRA caption adapter |

### How extraction works
- **Octopus detection** — CLIP ViT-B/32 (frozen) + a small MLP probe
  (`weights/clip_mlp_hardneg_v2.pt`), letterbox preprocessing, per-second `p_visible`.
- **Motion detection** — `scan_motion_area`: the absolute changed-pixel fraction with
  the burned-in timestamp masked out (robust to IR-lamp flicker).
- **Window gate** — keep a non-overlapping 20s window when **>50% of frames have
  `p_visible ≥ 0.6` AND mean motion ≥ 0.008**, then ffmpeg byte-range copies it.

## `src/` — self-contained
Bundles everything it needs to run standalone: the detector weight, `server_creds.py`,
`motion_detector.py`, and the two JSON state files.

```bash
cd src
pip install -r requirements.txt        # torch, openai-clip, numpy, pillow  (+ ffmpeg on PATH)
cp .env.example .env                   # then fill OCTOPUS_USER / OCTOPUS_PASS
python3 extract_octopus_clips.py --limit 5
```
Flags: `--limit`, `--date`, `--motion-thresh`, `--visible-frac`, `--vis-thresh`.

## `weights/` — model checkpoints
CLIP ViT-B/32 + `mlp_256_64` probes (visible vs hidden octopus). **Use
`clip_mlp_hardneg_v2.pt`** — letterbox + mined IR-noise hard negatives, the model the
pipeline uses. Others are earlier baselines kept for reference.

## Data files
- `octopus_clips_verified.json` — the clip index (one entry per extracted clip:
  source URL, time range, scores; captions filled in by stage 2).
- `octopus_clips_processed.json` — processed-video ledger, so videos are never
  re-scanned (resumable).

## Credentials
Footage-server creds are read from `.env` (`OCTOPUS_USER` / `OCTOPUS_PASS`) or
environment variables — **never hard-coded**. `.env` is gitignored; only
`.env.example` is committed.
