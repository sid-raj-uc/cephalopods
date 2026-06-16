# Nity Annotation UI

A browser-based tool for reviewing motion-scanned aquarium videos and saving behavioral captions.

## Setup

```bash
pip install fastapi uvicorn requests
python3 ui/server.py
# open http://localhost:8000
```

Requires `data/ethogram_index.json` to exist with at least some entries that have `motion_timeline` populated (run `phase2/exp16_motion_timeline.py --all` first).

## Workflow

1. **Browse cameras** — click camera tabs to switch between views of the same event. The active tab is highlighted blue.
2. **Mark good views** — click the `·` / `✓` toggle on each tab to mark which cameras have a useful view for this event. Multiple cameras can be marked. These are saved with the annotation.
3. **Use the timeline** — the bar below the video shows:
   - **Green blocks** — motion windows detected across cameras (darker = more cameras agreed)
   - **Orange line** — the original ethogram-recorded timestamp for the active camera
   - **Amber line** — live playhead
   - Click anywhere on the timeline to seek the video.
4. **Capture timestamps** — play or scrub to the moment of interest, then click **⏺ Start** / **⏺ End** to snap the current time into the form fields.
5. **Write a caption** — the caption box pre-fills with the ethogram event label; edit it to describe what Nity is doing in that window.
6. **Submit** — click **Submit & Next** to save and auto-advance. Click **Skip** to move on without saving.

## Re-annotating

Navigate back with **← Prev** to revisit a completed entry. The form will reload with the previously saved start, end, caption, and camera marks. Resubmit to overwrite.

## Data

Annotations are saved in-place to `data/ethogram_index.json` under each entry:

```json
"annotation": {
  "start": "16:50",
  "end": "18:16",
  "caption": "Octopus reaches toward screen during video call",
  "cameras_used": ["Right Back", "Right Top"],
  "annotated_at": "2026-06-15T03:53:10"
}
```
