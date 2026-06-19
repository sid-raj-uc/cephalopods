"""
Frame labeler — mark time patches where octopus is visible or hidden.
Patches saved to data/octopus_patches.json for frame extraction.

Usage:  python3 ui/labeler.py  →  http://localhost:8001
"""
import json, threading
from datetime import datetime
from pathlib import Path

import requests, uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

PROJECT      = Path(__file__).resolve().parent.parent
INDEX_PATH   = PROJECT / "data" / "ethogram_index.json"
PATCHES_PATH = PROJECT / "data" / "octopus_patches.json"
USER, PASS   = "octopus", "communication42"

app   = FastAPI()
_lock = threading.Lock()


def _load_index():
    with open(INDEX_PATH) as f:
        return json.load(f)

def _load_patches():
    if PATCHES_PATH.exists():
        with open(PATCHES_PATH) as f:
            return json.load(f)
    return {"patches": []}

def _save_patches(data):
    with open(PATCHES_PATH, "w") as f:
        json.dump(data, f, indent=2)

def _sec(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)

def _indexed(data):
    return [(i, e) for i, e in enumerate(data)
            if e.get("status") in ("indexed", "annotated")
            and "motion_timeline" in e]


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/entries")
def api_entries():
    data    = _load_index()
    patches = _load_patches()["patches"]
    counts  = {}
    for p in patches:
        counts[p["entry_idx"]] = counts.get(p["entry_idx"], 0) + 1
    return [
        {
            "idx":             i,
            "date":            e.get("date", ""),
            "time":            e.get("time", ""),
            "event":           e.get("event", ""),
            "patch_count":     counts.get(i, 0),
            "cameras":         [c for c in e.get("cameras", []) if c.get("available")],
            "motion_timeline": e.get("motion_timeline", []),
        }
        for i, e in _indexed(data)
    ]


@app.get("/api/patches/{entry_idx}")
def api_get_patches(entry_idx: int):
    return [p for p in _load_patches()["patches"] if p["entry_idx"] == entry_idx]


class PatchIn(BaseModel):
    entry_idx: int
    camera:    str
    video_url: str
    start:     str
    end:       str
    label:     str


@app.post("/api/patches")
def api_add_patch(p: PatchIn):
    entry = _load_index()[p.entry_idx]
    with _lock:
        store = _load_patches()
        store["patches"].append({
            "entry_idx": p.entry_idx,
            "date":      entry.get("date", ""),
            "time":      entry.get("time", ""),
            "event":     entry.get("event", ""),
            "camera":    p.camera,
            "video_url": p.video_url,
            "start":     p.start,
            "end":       p.end,
            "start_sec": _sec(p.start),
            "end_sec":   _sec(p.end),
            "label":     p.label,
            "added_at":  datetime.now().isoformat(timespec="seconds"),
        })
        _save_patches(store)
    return {"ok": True}


@app.delete("/api/patches/{entry_idx}/{local_idx}")
def api_del_patch(entry_idx: int, local_idx: int):
    with _lock:
        store = _load_patches()
        entry_patches = [(gi, p) for gi, p in enumerate(store["patches"])
                         if p["entry_idx"] == entry_idx]
        if local_idx < 0 or local_idx >= len(entry_patches):
            raise HTTPException(404, "patch not found")
        store["patches"].pop(entry_patches[local_idx][0])
        _save_patches(store)
    return {"ok": True}


@app.get("/proxy")
def proxy(url: str, request: Request):
    auth = url.replace("://", f"://{USER}:{PASS}@", 1)
    hdrs = {}
    if "range" in request.headers:
        hdrs["Range"] = request.headers["range"]
    r = requests.get(auth, headers=hdrs, stream=True, timeout=30)
    out = {"Content-Type": r.headers.get("Content-Type", "video/mp4"), "Accept-Ranges": "bytes"}
    for h in ("Content-Range", "Content-Length"):
        if h in r.headers:
            out[h] = r.headers[h]
    return StreamingResponse(r.iter_content(65536), status_code=r.status_code, headers=out)


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Frame Labeler</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;font-size:14px;background:#111;color:#eee;display:flex;height:100vh;overflow:hidden}

/* Sidebar */
#sidebar{width:230px;flex-shrink:0;background:#1a1a1a;border-right:1px solid #333;display:flex;flex-direction:column;overflow:hidden}
#sidebar h2{padding:10px 12px;font-size:12px;color:#666;border-bottom:1px solid #2a2a2a;letter-spacing:.05em;text-transform:uppercase}
#entry-list{flex:1;overflow-y:auto}
.ei{padding:9px 12px;cursor:pointer;border-bottom:1px solid #1e1e1e}
.ei:hover{background:#222}
.ei.active{background:#1c3454;border-left:3px solid #4a9eff}
.ei-date{font-size:11px;color:#666}
.ei-ev{font-size:12px;margin-top:2px;line-height:1.35}
.ei-pc{font-size:11px;color:#4a9eff;margin-top:2px}

/* Main */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#hdr{padding:8px 14px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;gap:10px}
#hdr h1{font-size:14px;color:#aaa;white-space:nowrap}
#ev-label{font-size:13px;color:#666;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* Camera tabs */
#cam-tabs{display:flex;gap:4px;padding:7px 12px;background:#161616;border-bottom:1px solid #252525;flex-wrap:wrap;min-height:36px}
.ct{padding:3px 10px;border-radius:4px;cursor:pointer;font-size:12px;background:#252525;color:#777;border:1px solid #333}
.ct:hover{background:#2e2e2e}
.ct.active{background:#1c3454;color:#4a9eff;border-color:#3a7acc}

/* Video */
#vwrap{flex:1;background:#000;display:flex;align-items:center;justify-content:center;min-height:0}
#video{max-width:100%;max-height:100%}

/* Timeline */
#tl-wrap{padding:8px 14px 6px;background:#161616;border-top:1px solid #252525}
#timeline{position:relative;height:36px;background:#1e1e1e;border-radius:5px;cursor:crosshair;user-select:none;overflow:hidden}
#tl-label{font-size:11px;color:#555;margin-top:4px;display:flex;justify-content:space-between}

/* Controls */
#controls{padding:9px 14px;background:#1a1a1a;border-top:1px solid #252525;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.cf{display:flex;align-items:center;gap:5px}
.cf label{font-size:12px;color:#777}
.cf input{background:#252525;border:1px solid #3a3a3a;color:#eee;padding:4px 8px;border-radius:4px;font-size:12px;width:68px;font-family:monospace}
.cbtn{background:#252525;border:1px solid #444;color:#ccc;padding:4px 9px;border-radius:4px;cursor:pointer;font-size:12px}
.cbtn:hover{background:#2e2e2e}
#lbl-sel{background:#252525;border:1px solid #3a3a3a;color:#eee;padding:4px 8px;border-radius:4px;font-size:12px}
#add-btn{background:#163316;border:1px solid #2a622a;color:#5cbf5c;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:13px;font-weight:600;margin-left:4px}
#add-btn:hover{background:#1c4a1c}
#add-btn:disabled{opacity:.4;cursor:default}
#status{font-size:12px;color:#4a9eff;margin-left:6px}

/* Patch list */
#plist-wrap{padding:8px 14px;background:#161616;border-top:1px solid #252525;max-height:130px;overflow-y:auto}
#plist-wrap h3{font-size:11px;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}
.pi{display:flex;align-items:center;gap:7px;padding:4px 8px;background:#1e1e1e;border-radius:4px;margin-bottom:3px;font-size:12px}
.pi-time{font-family:monospace;color:#ddd}
.pi-lbl{padding:1px 7px;border-radius:3px;font-size:11px}
.pi-lbl.visible{background:#163316;color:#5cbf5c}
.pi-lbl.hidden{background:#331616;color:#e05555}
.pi-cam{color:#666;flex:1;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pi-del{background:none;border:none;color:#555;cursor:pointer;font-size:14px;padding:0 3px;line-height:1}
.pi-del:hover{color:#e05555}
</style>
</head>
<body>

<div id="sidebar">
  <h2>Entries &nbsp;<span id="ec" style="color:#aaa;font-weight:normal"></span></h2>
  <div id="entry-list"></div>
</div>

<div id="main">
  <div id="hdr">
    <h1>Frame Labeler</h1>
    <span id="ev-label">← Select an entry</span>
  </div>

  <div id="cam-tabs"></div>

  <div id="vwrap">
    <video id="video" controls preload="metadata"></video>
  </div>

  <div id="tl-wrap">
    <div id="timeline"></div>
    <div id="tl-label"><span id="tl-cur">0:00</span><span id="tl-dur">30:00</span></div>
  </div>

  <div id="controls">
    <div class="cf">
      <label>Start</label>
      <input id="s-in" type="text" placeholder="MM:SS">
      <button class="cbtn" onclick="capStart()">⏺</button>
    </div>
    <div class="cf">
      <label>End</label>
      <input id="e-in" type="text" placeholder="MM:SS">
      <button class="cbtn" onclick="capEnd()">⏺</button>
    </div>
    <select id="lbl-sel">
      <option value="visible">Octopus visible</option>
      <option value="hidden">Octopus hidden / negative</option>
    </select>
    <button id="add-btn" onclick="addPatch()" disabled>+ Add Patch</button>
    <span id="status"></span>
  </div>

  <div id="plist-wrap">
    <h3>Patches — <span id="pcount">0</span></h3>
    <div id="plist"></div>
  </div>
</div>

<script>
const video    = document.getElementById('video');
const timeline = document.getElementById('timeline');

let entries    = [];
let curLi      = null;   // list index in `entries`
let curEntry   = null;
let curCamIdx  = 0;
let patches    = [];     // patches for current entry
let dragging   = false;
let dragFrac0  = 0;


// ── Load entries ──────────────────────────────────────────────────────────────

async function loadEntries() {
  const res = await fetch('/api/entries');
  entries   = await res.json();
  document.getElementById('ec').textContent = entries.length;
  const list = document.getElementById('entry-list');
  list.innerHTML = entries.map((e, li) => `
    <div class="ei" id="ei-${li}" onclick="selectEntry(${li})">
      <div class="ei-date">${e.date} ${e.time}</div>
      <div class="ei-ev">${e.event}</div>
      ${e.patch_count ? `<div class="ei-pc">${e.patch_count} patch${e.patch_count>1?'es':''}</div>` : ''}
    </div>`).join('');
}


// ── Select entry ──────────────────────────────────────────────────────────────

async function selectEntry(li) {
  if (curLi !== null) document.getElementById(`ei-${curLi}`)?.classList.remove('active');
  curLi    = li;
  curEntry = entries[li];
  document.getElementById(`ei-${li}`).classList.add('active');
  document.getElementById('ev-label').textContent =
    `${curEntry.date} ${curEntry.time} — ${curEntry.event}`;
  document.getElementById('add-btn').disabled = false;

  curCamIdx = 0;
  buildCamTabs();
  await refreshPatches();
  switchCamera(0);
}

function buildCamTabs() {
  document.getElementById('cam-tabs').innerHTML =
    curEntry.cameras.map((c, i) =>
      `<div class="ct${i===0?' active':''}" id="ct-${i}" onclick="switchCamera(${i})">${c.name}</div>`
    ).join('');
}

function switchCamera(i) {
  document.querySelectorAll('.ct').forEach(t => t.classList.remove('active'));
  document.getElementById(`ct-${i}`)?.classList.add('active');
  curCamIdx = i;
  const cam  = curEntry.cameras[i];
  const prev = video.currentTime || 0;
  video.src  = `/proxy?url=${encodeURIComponent(cam.video_url)}`;
  video.load();
  video.addEventListener('loadedmetadata', () => {
    video.currentTime = prev;
    renderTimeline();
  }, { once: true });
}


// ── Timeline ──────────────────────────────────────────────────────────────────

function renderTimeline() {
  const tl  = timeline;
  const dur = video.duration || 1800;
  tl.innerHTML = '';

  // Motion windows (green)
  for (const w of (curEntry?.motion_timeline || [])) {
    const s = parseSec(w.start), e = parseSec(w.end);
    const el = div(`position:absolute;top:6px;height:14px;` +
      `left:${s/dur*100}%;width:${Math.max((e-s)/dur*100,0.2)}%;` +
      `background:rgba(76,175,80,${Math.min(0.15+w.vote_count*0.12,0.7)});border-radius:2px`);
    tl.appendChild(el);
  }

  // Existing patches (colored bars, bottom strip)
  for (const p of patches) {
    const s = p.start_sec, e = p.end_sec;
    const col = p.label === 'visible' ? '#5cbf5c' : '#e05555';
    const el  = div(`position:absolute;bottom:4px;height:8px;` +
      `left:${s/dur*100}%;width:${Math.max((e-s)/dur*100,0.2)}%;` +
      `background:${col};border-radius:2px;opacity:0.85`);
    tl.appendChild(el);
  }

  // Drag selection overlay
  const dragEl = div('position:absolute;top:0;height:100%;background:rgba(74,158,255,0.2);border:1px solid #4a9eff;border-radius:2px;display:none;pointer-events:none');
  dragEl.id = 'drag-sel';
  tl.appendChild(dragEl);

  // Playhead
  const ph = div('position:absolute;top:0;width:2px;height:100%;background:#f90;opacity:.9;pointer-events:none');
  ph.id = 'playhead';
  tl.appendChild(ph);
}

function div(css) {
  const el = document.createElement('div');
  el.style.cssText = css;
  return el;
}

video.addEventListener('timeupdate', () => {
  const dur = video.duration || 1800;
  const ph  = document.getElementById('playhead');
  if (ph) ph.style.left = `${video.currentTime/dur*100}%`;
  document.getElementById('tl-cur').textContent = fmtSec(video.currentTime);
  document.getElementById('tl-dur').textContent = fmtSec(dur);
});


// ── Timeline drag + click ─────────────────────────────────────────────────────

timeline.addEventListener('mousedown', e => {
  const rect = timeline.getBoundingClientRect();
  dragFrac0  = (e.clientX - rect.left) / rect.width;
  dragging   = true;
  const ds = document.getElementById('drag-sel');
  if (ds) { ds.style.left = `${dragFrac0*100}%`; ds.style.width = '0'; ds.style.display = 'block'; }
  e.preventDefault();
});

document.addEventListener('mousemove', e => {
  if (!dragging) return;
  const rect = timeline.getBoundingClientRect();
  const cur  = clamp((e.clientX - rect.left) / rect.width);
  const s = Math.min(dragFrac0, cur), w = Math.abs(cur - dragFrac0);
  const ds = document.getElementById('drag-sel');
  if (ds) { ds.style.left = `${s*100}%`; ds.style.width = `${w*100}%`; }
});

document.addEventListener('mouseup', e => {
  if (!dragging) return;
  dragging = false;
  const ds = document.getElementById('drag-sel');
  if (ds) ds.style.display = 'none';

  const rect  = timeline.getBoundingClientRect();
  const cur   = clamp((e.clientX - rect.left) / rect.width);
  const delta = Math.abs(cur - dragFrac0);
  const dur   = video.duration || 1800;

  if (delta * rect.width < 5) {
    // short click → seek
    video.currentTime = dragFrac0 * dur;
  } else {
    // drag → fill start/end
    const s = Math.min(dragFrac0, cur) * dur;
    const en = Math.max(dragFrac0, cur) * dur;
    document.getElementById('s-in').value = fmtSec(s);
    document.getElementById('e-in').value = fmtSec(en);
  }
});


// ── Capture buttons ───────────────────────────────────────────────────────────

function capStart() { document.getElementById('s-in').value = fmtSec(video.currentTime); }
function capEnd()   { document.getElementById('e-in').value = fmtSec(video.currentTime); }


// ── Add / delete patches ──────────────────────────────────────────────────────

async function addPatch() {
  if (!curEntry) return;
  const start = document.getElementById('s-in').value.trim();
  const end   = document.getElementById('e-in').value.trim();
  const label = document.getElementById('lbl-sel').value;
  if (!start || !end) { setStatus('Set start and end first', '#e05555'); return; }
  if (parseSec(end) <= parseSec(start)) { setStatus('End must be after start', '#e05555'); return; }

  const cam = curEntry.cameras[curCamIdx];
  const btn = document.getElementById('add-btn');
  btn.disabled = true;
  await fetch('/api/patches', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      entry_idx: curEntry.idx, camera: cam.name, video_url: cam.video_url,
      start, end, label,
    }),
  });
  document.getElementById('s-in').value = '';
  document.getElementById('e-in').value = '';
  btn.disabled = false;
  setStatus('Patch saved!', '#5cbf5c');
  await refreshPatches();
  renderTimeline();
  updateSidebarCount();
}

async function deletePatch(li) {
  await fetch(`/api/patches/${curEntry.idx}/${li}`, { method: 'DELETE' });
  await refreshPatches();
  renderTimeline();
  updateSidebarCount();
}

async function refreshPatches() {
  if (!curEntry) return;
  const res = await fetch(`/api/patches/${curEntry.idx}`);
  patches   = await res.json();
  renderPatches();
}

function renderPatches() {
  document.getElementById('pcount').textContent = patches.length;
  const list = document.getElementById('plist');
  if (!patches.length) {
    list.innerHTML = '<div style="color:#444;font-size:12px;padding:2px 0">No patches yet — drag on the timeline or use ⏺ buttons</div>';
    return;
  }
  list.innerHTML = patches.map((p, i) => `
    <div class="pi">
      <span class="pi-time">${p.start}–${p.end}</span>
      <span class="pi-lbl ${p.label}">${p.label}</span>
      <span class="pi-cam">${p.camera}</span>
      <button class="pi-del" onclick="deletePatch(${i})" title="Delete">✕</button>
    </div>`).join('');
}

function updateSidebarCount() {
  if (curLi === null) return;
  entries[curLi].patch_count = patches.length;
  const el = document.getElementById(`ei-${curLi}`);
  if (!el) return;
  let pc = el.querySelector('.ei-pc');
  if (patches.length) {
    if (!pc) { pc = document.createElement('div'); pc.className = 'ei-pc'; el.appendChild(pc); }
    pc.textContent = `${patches.length} patch${patches.length>1?'es':''}`;
  } else if (pc) { pc.remove(); }
}


// ── Helpers ───────────────────────────────────────────────────────────────────

function parseSec(t) {
  const [m, s] = t.split(':').map(Number); return m*60 + s;
}
function fmtSec(sec) {
  const m = Math.floor(sec/60), s = Math.floor(sec%60);
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}
function clamp(v) { return Math.max(0, Math.min(1, v)); }

let _stTimer;
function setStatus(msg, col='#4a9eff') {
  const el = document.getElementById('status');
  el.style.color = col; el.textContent = msg;
  clearTimeout(_stTimer);
  _stTimer = setTimeout(() => { el.textContent = ''; }, 2500);
}


// ── Init ──────────────────────────────────────────────────────────────────────

loadEntries();
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
