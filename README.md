# Octopus Clip-Extraction & Captioning Pipeline

Turn continuous aquarium camera footage of an octopus (*Octopus vulgaris*, "Nity")
into short, behavior-labeled, captioned clips — automatically.

Given a video, the pipeline finds 20-second windows where the octopus is
**both visible and moving**, extracts them, and captions them with a
vision-language model. The teacher VLM's captions were distilled into a small
**local caption student** (Qwen3-VL-2B, MLX 4-bit) that runs on a Mac — so the
whole video → captions loop works offline, no GPU server needed.

## Quickstart — the captioning UI (upload a video, get captions)

```bash
pip install -r src/requirements.txt   # + ffmpeg on PATH
./download_model.sh                   # auto-fetches the right caption student for your platform
python3 ui/demo_player.py             # → http://localhost:8011
```

Click **⬆ Upload a video**: the full pipeline runs (CLIP+MLP octopus detection +
motion gating → 20s windows → caption student), and the video appears as a tab —
it plays on the left with its captions synced on the right. Click a caption to jump
to that moment.

### Runs on any platform — two caption backends, auto-selected

The caption student runs on Mac, Linux, Windows, CPU or GPU. `local_pipeline.py`
picks the backend automatically (override with `--backend mlx|hf`):

| Platform | Backend | Model (fetched by `download_model.sh`) | Notes |
|----------|---------|----------------------------------------|-------|
| **Apple Silicon** (arm64 macOS) | `mlx` | MLX 4-bit, `models/qwen3vl2b_caption_v1_mlx_4bit/` (~1.7 GB) | fastest laptop-local path, ~3–5 s/caption |
| **Linux / Windows / CUDA / CPU / Intel Mac** | `hf` | LoRA adapter, `models/qwen3vl2b_caption_v1_lora/` (~67 MB) | base `Qwen3-VL-2B-Instruct` auto-downloads from HF Hub; 4-bit via bitsandbytes on CUDA, fp16 on GPU, fp32 on CPU |

On CUDA also `pip install bitsandbytes` for 4-bit inference. The extraction stages
(CLIP+MLP detector, motion) already run on any device (`mps → cuda → cpu`).

## Layout

| Dir | What |
|-----|------|
| `src/` | the pipeline: extraction, captioning, dataset building, student training |
| `ui/` | the captioning UI apps (see `ui/README.md`) |
| `weights/` | CLIP+MLP octopus-detector checkpoints |
| `models/` | the caption student (created by `download_model.sh`, not committed) |

## Pipeline (in `src/`)

| Stage | File | Runs on | Output |
|-------|------|---------|--------|
| 1. **Extract** clips | `src/extract_octopus_clips.py` (server) / `src/local_pipeline.py` (local file, single-decode fast path) | local (CPU/MPS) | clips + `octopus_clips_verified.json` |
| 2. **Caption** clips (teacher) | `src/caption_openrouter.py` (Qwen3-VL-235B via OpenRouter, no GPU) or `src/caption_octopus_clips.ipynb` (Qwen3-VL-30B, Colab GPU) | local / Colab | `caption` + `ethogram_label` in the JSON |
| 3. **Build dataset** | `src/build_caption_dataset.py` (select + dedup + split + best-N CLAHE frames) | local | `src/dataset/vN/` snapshot |
| 4. **Train student** | `src/train_caption_student_qwen3vl.ipynb` (Qwen3-VL-2B, QLoRA) | Colab GPU | LoRA caption adapter |
| 5. **Run student locally** | `src/local_pipeline.py` / `src/local_video_to_captions.ipynb` (MLX 4-bit) | Mac (Apple Silicon) | captions, ~3 s/clip |

The locked training plan is `src/TRAINING_PLAN.md` (caption-only distillation,
retrain-from-base on versioned cumulative snapshots). Trained v1 eval on held-out
val: embedding-similarity 0.702 → **0.834**, rougeL 0.269 → **0.455** vs the base model.

### How extraction works
- **Octopus detection** — CLIP ViT-B/32 (frozen) + a small MLP probe
  (`weights/clip_mlp_hardneg_v2.pt`), letterbox preprocessing, per-second `p_visible`.
- **Motion detection** — `scan_motion_area`: the absolute changed-pixel fraction with
  the burned-in timestamp masked out (robust to IR-lamp flicker).
- **Window gate** — keep a non-overlapping 20s window when **>50% of frames have
  `p_visible ≥ 0.6` AND mean motion ≥ 0.008**, then ffmpeg byte-range copies it.
- `src/local_pipeline.py` decodes the video **once** and feeds both detectors from
  that single stream (~1.7× faster), then reuses the scan's per-second scores to pick
  the best frames for captioning.

## `src/` — self-contained
Bundles everything it needs to run standalone: the detector weight, `server_creds.py`,
`motion_detector.py`, the 7-class ethogram (`ethogram_list_v2.json`), and the two JSON
state files.

```bash
cd src
pip install -r requirements.txt
cp .env.example .env                   # OCTOPUS_USER / OCTOPUS_PASS (footage server, optional)
                                       # OPENROUTER_API_KEY (teacher captioning, optional)
python3 extract_octopus_clips.py --limit 5          # server videos
python3 local_pipeline.py /path/to/video.mp4        # a local file → clips + captions
```

## `weights/` — model checkpoints
CLIP ViT-B/32 + `mlp_256_64` probes (visible vs hidden octopus). **Use
`clip_mlp_hardneg_v2.pt`** — letterbox + mined IR-noise hard negatives, the model the
pipeline uses. Others are earlier baselines kept for reference.

## `models/` — the caption student (GitHub release `caption-student-v1`)
Qwen3-VL-2B fine-tuned (QLoRA) on the teacher captions. Not committed; fetch with
`./download_model.sh` (auto-picks by platform, or pass `mlx` / `lora` / `both`):
- `models/qwen3vl2b_caption_v1_mlx_4bit/` — MLX 4-bit merged model (~1.7 GB), Apple Silicon.
- `models/qwen3vl2b_caption_v1_lora/` — LoRA adapter (~67 MB) for the cross-platform
  transformers backend; the base `Qwen3-VL-2B-Instruct` downloads from the HF Hub.

## Data files
- `octopus_clips_verified.json` — the clip index (one entry per extracted clip:
  source URL, time range, scores; captions filled in by stage 2).
- `octopus_clips_processed.json` — processed-video ledger, so videos are never
  re-scanned (resumable).

## Credentials
Footage-server creds are read from `.env` (`OCTOPUS_USER` / `OCTOPUS_PASS`) or
environment variables — **never hard-coded**. `.env` is gitignored; only
`.env.example` is committed. `OPENROUTER_API_KEY` (teacher captioning) works the
same way. The UI and local pipeline need **no credentials at all** for local files.
