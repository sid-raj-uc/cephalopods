"""
Nity Annotation UI — FastAPI backend + single-page frontend.

Usage:
    python3 ui/server.py
    open http://localhost:8000

Features:
- Side-by-side video players for each available camera
- Motion timeline bar with clickable seek
- Time range + caption form → saves annotation to ethogram_index.json
"""

import json, datetime
from pathlib import Path
from typing import Optional

import requests as req_lib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

PROJECT    = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT / "data" / "ethogram_index.json"
USER, PASS = "octopus", "communication42"

app = FastAPI()


# ── data helpers ──────────────────────────────────────────────────────────────

def load_index():
    with open(INDEX_PATH) as f:
        return json.load(f)

def save_index(data):
    with open(INDEX_PATH, "w") as f:
        json.dump(data, f, indent=2)

def indexed_indices(data):
    """Return list of JSON array indices for events that are ready for review (have motion_timeline)."""
    return [i for i, e in enumerate(data)
            if e.get("status") in ("indexed", "annotated")
            and "motion_timeline" in e]


# ── video proxy ───────────────────────────────────────────────────────────────

@app.get("/proxy/{event_idx}/{camera:path}")
async def proxy_video(event_idx: int, camera: str, request: Request):
    data = load_index()
    if event_idx >= len(data):
        raise HTTPException(404, "Event not found")
    entry = data[event_idx]
    cam = next((c for c in entry.get("cameras", [])
                if c["name"] == camera and c.get("available")), None)
    if not cam:
        raise HTTPException(404, "Camera not available")

    headers = {}
    if rng := request.headers.get("range"):
        headers["Range"] = rng

    remote = req_lib.get(cam["video_url"], auth=(USER, PASS),
                         headers=headers, stream=True, timeout=30)

    resp_headers = {"Accept-Ranges": "bytes"}
    for h in ("Content-Type", "Content-Length", "Content-Range"):
        if h in remote.headers:
            resp_headers[h] = remote.headers[h]

    return StreamingResponse(
        remote.iter_content(chunk_size=65536),
        status_code=remote.status_code,
        headers=resp_headers,
        media_type=remote.headers.get("Content-Type", "video/mp4"),
    )


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/events")
def api_events():
    data = load_index()
    ids  = indexed_indices(data)
    return {"total": len(ids), "event_indices": ids}


@app.get("/api/event/{event_idx}")
def api_event(event_idx: int):
    data = load_index()
    if event_idx >= len(data):
        raise HTTPException(404)
    entry  = data[event_idx]
    ids    = indexed_indices(data)
    pos    = ids.index(event_idx) if event_idx in ids else -1
    return {
        "idx":       event_idx,
        "pos":       pos,          # 0-based position in the to-do list
        "total":     len(ids),
        "prev_idx":  ids[pos - 1] if pos > 0 else None,
        "next_idx":  ids[pos + 1] if pos < len(ids) - 1 else None,
        "entry":     entry,
    }


class Annotation(BaseModel):
    start:         str
    end:           str
    caption:       str
    cameras_used:  list[str] = []

@app.post("/api/event/{event_idx}/annotate")
def api_annotate(event_idx: int, body: Annotation):
    data = load_index()
    if event_idx >= len(data):
        raise HTTPException(404)
    data[event_idx]["annotation"] = {
        "start":         body.start,
        "end":           body.end,
        "caption":       body.caption,
        "cameras_used":  body.cameras_used,
        "annotated_at":  datetime.datetime.utcnow().isoformat(),
    }
    data[event_idx]["status"] = "annotated"
    save_index(data)
    ids    = indexed_indices(data)   # recalculate — this event is now removed
    return {"ok": True, "remaining": len(ids)}


# ── frontend ──────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Nity Annotator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ddd; font-family: system-ui, sans-serif; font-size: 14px; }

/* ── header ── */
#header {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
}
#header button {
  background: #2a2a2a; color: #ddd; border: 1px solid #444;
  padding: 5px 14px; border-radius: 4px; cursor: pointer; font-size: 13px;
}
#header button:hover { background: #3a3a3a; }
#header button:disabled { opacity: 0.3; cursor: default; }
#event-info { flex: 1; }
#event-title { font-size: 15px; font-weight: 600; color: #fff; }
#event-meta  { font-size: 12px; color: #888; margin-top: 2px; }
#event-count { font-size: 13px; color: #aaa; white-space: nowrap; }

/* ── camera selector ── */
#cam-bar {
  display: flex; align-items: center; gap: 6px;
  padding: 7px 10px; background: #161616; border-bottom: 1px solid #2a2a2a;
  flex-wrap: wrap;
}
#cam-bar span { font-size: 11px; color: #555; margin-right: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.cam-btn {
  display: flex; align-items: center; gap: 0;
  background: #242424; color: #999; border: 1px solid #383838;
  padding: 0; border-radius: 4px; cursor: pointer; font-size: 12px; overflow: hidden;
}
.cam-btn:hover { border-color: #555; }
.cam-btn.active { background: #1e3a5f; color: #93c5fd; border-color: #2563eb; }
.cam-name { padding: 4px 10px 4px 8px; }
.cam-check {
  display: flex; align-items: center; justify-content: center;
  width: 26px; padding: 4px 0;
  border-right: 1px solid #383838; font-size: 13px;
  color: #444; cursor: pointer; flex-shrink: 0;
  transition: background 0.1s;
}
.cam-check:hover { background: rgba(255,255,255,0.07); }
.cam-btn.active .cam-check { border-right-color: #2a4a7f; }
.cam-btn.marked .cam-check { color: #4ade80; }
.cam-btn:not(.marked) .cam-check { color: #3a3a3a; }

/* ── single video ── */
#video-wrap { background: #000; display: flex; justify-content: center; }
#video-wrap video { width: 100%; max-height: 62vh; display: block; }

/* ── motion timeline ── */
#timeline-wrap { padding: 8px 10px 4px; }
#timeline-label {
  display: flex; gap: 16px; font-size: 11px; color: #555; margin-bottom: 4px;
}
#timeline-label .leg { display: flex; align-items: center; gap: 5px; }
#timeline-label .dot { width: 10px; height: 10px; border-radius: 2px; }
#timeline-svg { width: 100%; cursor: crosshair; display: block; }
#time-ticks { display: flex; justify-content: space-between; font-size: 10px; color: #555; margin-top: 2px; }

/* ── annotation form ── */
#annotation {
  display: flex; align-items: flex-end; gap: 10px;
  padding: 10px 10px 14px; background: #161616; border-top: 1px solid #2a2a2a;
  flex-wrap: wrap;
}
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 11px; color: #888; }
.field input, .field textarea {
  background: #222; border: 1px solid #444; color: #eee;
  border-radius: 4px; padding: 6px 9px; font-size: 13px;
}
.field input  { width: 90px; }
.field textarea { width: 480px; height: 52px; resize: vertical; }
.time-row { display: flex; gap: 4px; align-items: center; }
.btn-capture {
  background: #1f2937; color: #9ca3af; border: 1px solid #374151;
  padding: 5px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;
}
.btn-capture:hover { background: #374151; color: #e5e7eb; }
#btn-submit {
  background: #2563eb; color: #fff; border: none;
  padding: 8px 20px; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: 600;
}
#btn-submit:hover { background: #1d4ed8; }
#btn-skip {
  background: #374151; color: #ccc; border: 1px solid #4b5563;
  padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 13px;
}
#btn-skip:hover { background: #4b5563; }
#status-msg { font-size: 12px; color: #6ee7b7; align-self: center; }
</style>
</head>
<body>

<div id="header">
  <button id="btn-prev">← Prev</button>
  <div id="event-info">
    <div id="event-title">Loading…</div>
    <div id="event-meta"></div>
  </div>
  <span id="event-count"></span>
  <button id="btn-next">Next →</button>
</div>

<div id="cam-bar">
  <span>Camera</span>
</div>

<div id="video-wrap">
  <video id="main-video" controls preload="metadata"></video>
</div>

<div id="timeline-wrap">
  <div id="timeline-label">
    <div class="leg"><div class="dot" style="background:#22c55e"></div>Motion windows</div>
    <div class="leg"><div class="dot" style="background:#f97316;width:3px;height:14px"></div>Ethogram timestamp</div>
    <div class="leg"><div class="dot" style="background:#f59e0b;width:2px;height:14px"></div>Playhead</div>
  </div>
  <svg id="timeline-svg" height="36"></svg>
  <div id="time-ticks"></div>
</div>

<div id="annotation">
  <div class="field">
    <label>Start (MM:SS)</label>
    <div class="time-row">
      <input id="inp-start" type="text" placeholder="05:10">
      <button class="btn-capture" id="btn-cap-start">⏺ Start</button>
    </div>
  </div>
  <div class="field">
    <label>End (MM:SS)</label>
    <div class="time-row">
      <input id="inp-end" type="text" placeholder="06:30">
      <button class="btn-capture" id="btn-cap-end">⏺ End</button>
    </div>
  </div>
  <div class="field">
    <label>Caption</label>
    <textarea id="inp-caption" placeholder="Describe what Nity is doing…"></textarea>
  </div>
  <button id="btn-submit">Submit &amp; Next</button>
  <button id="btn-skip">Skip</button>
  <span id="status-msg"></span>
</div>

<script>
let currentIdx      = null;
let eventList       = [];
let availCams       = [];   // [{name, event_offset}, ...]
let currentCamIdx   = 0;   // camera being viewed
let camMarked       = new Set(); // indices of cameras marked as "good view"
let currentTimeline = [];
let currentEventSec = null;

const video = document.getElementById('main-video');

// ── helpers ──────────────────────────────────────────────────────────────────

function mmss(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}
function secFromMmss(s) {
  if (!s) return null;
  const [m, sec] = s.split(':').map(Number);
  return (m || 0)*60 + (sec || 0);
}

// ── camera switching ─────────────────────────────────────────────────────────

function updateCamTabs() {
  document.querySelectorAll('.cam-btn').forEach((b, i) => {
    b.classList.toggle('active',  i === currentCamIdx);
    b.classList.toggle('marked',  camMarked.has(i));
    b.querySelector('.cam-check').textContent = camMarked.has(i) ? '✓' : '·';
  });
}

function toggleMark(idx, e) {
  e.stopPropagation();
  if (camMarked.has(idx)) camMarked.delete(idx); else camMarked.add(idx);
  updateCamTabs();
}

function switchCamera(idx) {
  const savedTime = video.readyState >= 1 ? video.currentTime : 0;
  currentCamIdx = idx;
  const cam = availCams[idx];

  updateCamTabs();

  video.src = `/proxy/${currentIdx}/${encodeURIComponent(cam.name)}`;
  video.addEventListener('loadedmetadata', () => {
    video.currentTime = savedTime;
  }, { once: true });

  currentEventSec = secFromMmss(cam.event_offset);
  renderTimeline(currentTimeline, currentEventSec, 1800);
}

// ── timeline ─────────────────────────────────────────────────────────────────

function renderTimeline(motionWindows, eventSec, totalSec) {
  const svg = document.getElementById('timeline-svg');
  const W   = svg.parentElement.clientWidth - 20;
  svg.setAttribute('width', W);
  svg.innerHTML = '';

  const TRACK = 24, Y = 6;

  // background
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bg.setAttribute('x', 0); bg.setAttribute('y', Y);
  bg.setAttribute('width', W); bg.setAttribute('height', TRACK);
  bg.setAttribute('fill', '#222'); bg.setAttribute('rx', 3);
  svg.appendChild(bg);

  // motion windows (green)
  const maxVotes = Math.max(...motionWindows.map(w => w.vote_count), 1);
  motionWindows.forEach(w => {
    const s1 = secFromMmss(w.start), s2 = secFromMmss(w.end);
    const x  = (s1 / totalSec) * W;
    const ww = Math.max(2, ((s2 - s1) / totalSec) * W);
    const alpha = 0.35 + 0.65 * (w.vote_count / maxVotes);
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x); rect.setAttribute('y', Y);
    rect.setAttribute('width', ww); rect.setAttribute('height', TRACK);
    rect.setAttribute('fill', `rgba(34,197,94,${alpha})`);
    rect.setAttribute('rx', 2);
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${w.start}–${w.end}  votes=${w.vote_count}  peak=${w.peak}`;
    rect.appendChild(title);
    svg.appendChild(rect);
  });

  // ethogram timestamp marker (orange) — drawn before playhead so it sits under it
  if (eventSec !== null) {
    const ex = (eventSec / totalSec) * W;
    const eline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    eline.setAttribute('x1', ex); eline.setAttribute('x2', ex);
    eline.setAttribute('y1', Y - 4); eline.setAttribute('y2', Y + TRACK + 4);
    eline.setAttribute('stroke', '#f97316'); eline.setAttribute('stroke-width', '3');
    svg.appendChild(eline);
    const etitle = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    etitle.textContent = `Ethogram timestamp: ${mmss(eventSec)}`;
    eline.appendChild(etitle);
  }

  // playhead (amber)
  const playhead = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  playhead.setAttribute('id', 'playhead');
  playhead.setAttribute('x1', 0); playhead.setAttribute('x2', 0);
  playhead.setAttribute('y1', Y - 2); playhead.setAttribute('y2', Y + TRACK + 2);
  playhead.setAttribute('stroke', '#f59e0b'); playhead.setAttribute('stroke-width', '2');
  svg.appendChild(playhead);

  function updateHead() {
    const x = (video.currentTime / totalSec) * W;
    document.getElementById('playhead')?.setAttribute('x1', x);
    document.getElementById('playhead')?.setAttribute('x2', x);
    requestAnimationFrame(updateHead);
  }
  requestAnimationFrame(updateHead);

  // click to seek
  svg.addEventListener('click', e => {
    const rect2 = svg.getBoundingClientRect();
    const frac  = (e.clientX - rect2.left) / W;
    video.currentTime = frac * totalSec;
  });

  // ticks
  const ticks = document.getElementById('time-ticks');
  ticks.innerHTML = '';
  for (let m = 0; m <= 30; m += 5) {
    const span = document.createElement('span');
    span.textContent = mmss(m * 60);
    ticks.appendChild(span);
  }
}

// ── load event ───────────────────────────────────────────────────────────────

async function loadEvent(idx) {
  const res  = await fetch(`/api/event/${idx}`);
  const data = await res.json();
  currentIdx = idx;

  const entry = data.entry;
  availCams   = (entry.cameras || []).filter(c => c.available);
  currentTimeline = entry.motion_timeline || [];

  // header
  document.getElementById('event-title').textContent =
    `${entry.date}  ${entry.time}  —  ${entry.event || '(no event name)'}`;
  document.getElementById('event-meta').textContent =
    entry.details ? entry.details.slice(0, 120) + (entry.details.length > 120 ? '…' : '') : '';
  document.getElementById('event-count').textContent =
    `${data.pos + 1} / ${data.total}`;

  document.getElementById('btn-prev').disabled = data.prev_idx === null;
  document.getElementById('btn-next').disabled = data.next_idx === null;
  document.getElementById('btn-prev').onclick  = () => loadEvent(data.prev_idx);
  document.getElementById('btn-next').onclick  = () => loadEvent(data.next_idx);

  const saved = entry.annotation;

  // build camera tab bar
  currentCamIdx = 0;
  camMarked     = new Set(
    saved?.cameras_used
      ?.map(name => availCams.findIndex(c => c.name === name))
      .filter(i => i >= 0) ?? []
  );
  const bar = document.getElementById('cam-bar');
  bar.innerHTML = '<span>Camera</span>';
  availCams.forEach((cam, i) => {
    const btn = document.createElement('button');
    btn.className = 'cam-btn' + (i === 0 ? ' active' : '') + (camMarked.has(i) ? ' marked' : '');
    const check = document.createElement('span');
    check.className = 'cam-check';
    check.textContent = camMarked.has(i) ? '✓' : '·';
    check.title = 'Toggle: good view for this event';
    check.onclick = (e) => toggleMark(i, e);
    const label = document.createElement('span');
    label.className = 'cam-name';
    label.textContent = cam.name;
    btn.appendChild(check);
    btn.appendChild(label);
    btn.onclick = () => switchCamera(i);
    bar.appendChild(btn);
  });

  // load camera — use saved annotation's camera if present, else first available
  if (availCams.length && !saved?.camera) {
    currentEventSec = secFromMmss(availCams[0].event_offset);
    video.src = `/proxy/${idx}/${encodeURIComponent(availCams[0].name)}`;
  } else if (!availCams.length) {
    video.src = '';
    currentEventSec = null;
  }

  // pre-fill: use existing annotation if present, else best motion window
  if (saved) {
    document.getElementById('inp-start').value   = saved.start   || '';
    document.getElementById('inp-end').value     = saved.end     || '';
    document.getElementById('inp-caption').value = saved.caption || '';
    document.getElementById('status-msg').textContent = '(previously annotated — edit and resubmit to update)';
    // also switch to the camera that was used, if still available
    if (saved.camera) {
      const savedIdx = availCams.findIndex(c => c.name === saved.camera);
      if (savedIdx >= 0) {
        currentCamIdx  = savedIdx;
        selectedCamIdx = savedIdx;
        video.src = `/proxy/${idx}/${encodeURIComponent(availCams[savedIdx].name)}`;
        currentEventSec = secFromMmss(availCams[savedIdx].event_offset);
        updateCamTabs();
      }
    }
  } else {
    if (currentTimeline.length) {
      const best = currentTimeline[0];
      document.getElementById('inp-start').value = best.start;
      document.getElementById('inp-end').value   = best.end;
    } else {
      document.getElementById('inp-start').value = '';
      document.getElementById('inp-end').value   = '';
    }
    document.getElementById('inp-caption').value = entry.event || '';
    document.getElementById('status-msg').textContent = '';
  }

  renderTimeline(currentTimeline, currentEventSec, 1800);
  window.scrollTo(0, 0);
}

// ── submit / skip ─────────────────────────────────────────────────────────────

document.getElementById('btn-submit').onclick = async () => {
  const start   = document.getElementById('inp-start').value.trim();
  const end     = document.getElementById('inp-end').value.trim();
  const caption = document.getElementById('inp-caption').value.trim();
  if (!start || !end || !caption) {
    document.getElementById('status-msg').textContent = '⚠ Fill in start, end, and caption.';
    return;
  }
  const res  = await fetch(`/api/event/${currentIdx}/annotate`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      start, end, caption,
      cameras_used: [...camMarked].map(i => availCams[i]?.name).filter(Boolean),
    }),
  });
  const data = await res.json();
  document.getElementById('status-msg').textContent = `✓ Saved. ${data.remaining} events remaining.`;

  const next = document.getElementById('btn-next');
  if (!next.disabled) next.onclick();
};

document.getElementById('btn-cap-start').onclick = () => {
  document.getElementById('inp-start').value = mmss(video.currentTime);
};
document.getElementById('btn-cap-end').onclick = () => {
  document.getElementById('inp-end').value = mmss(video.currentTime);
};

document.getElementById('btn-skip').onclick = () => {
  const next = document.getElementById('btn-next');
  if (!next.disabled) next.onclick();
};

// ── init ─────────────────────────────────────────────────────────────────────

async function init() {
  const res  = await fetch('/api/events');
  const data = await res.json();
  eventList  = data.event_indices;
  if (eventList.length === 0) {
    document.getElementById('event-title').textContent = 'No events ready yet — run exp16 first.';
    return;
  }
  loadEvent(eventList[0]);
}
init();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
