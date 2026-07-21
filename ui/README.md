# Captioning UI

Two FastAPI apps. Both run the full local pipeline (`src/local_pipeline.py`:
CLIP+MLP octopus detection + absolute-motion gating → 20 s windows → MLX 4-bit
caption student). Install deps from `src/requirements.txt`, fetch the model with
`../download_model.sh`, and make sure `ffmpeg` is on PATH.

## `demo_player.py` — upload a video → synced captions (port 8011)

```bash
python3 ui/demo_player.py    # → http://localhost:8011
```

- **⬆ Upload a video** — runs the whole pipeline on it (one job at a time; models
  load lazily on the first job) and adds it as a tab.
- The video plays on the **left**; its captions list on the **right**, each with a
  `mm:ss–mm:ss` timeline. Click a caption to jump the video there; the active
  caption highlights and auto-scrolls during playback.
- Processed videos + `*_captions.json` are stored in `data/demo_videos/` and appear
  again on restart (processing runs once per video).

This is the main demo: video in → captions out, fully local, no credentials.

## `local_pipeline_app.py` — clips + captions browser (port 8010)

```bash
python3 ui/local_pipeline_app.py    # → http://localhost:8010
```

Point it at a **local video path**, or pick a **suggested footage-server video**
(ranked by known octopus activity from the clip index; needs `OCTOPUS_USER`/`PASS`
in `src/.env`). Streams each extracted 20 s clip inline next to its caption and
timeline as the pipeline produces them.
