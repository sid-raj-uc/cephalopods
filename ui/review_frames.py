"""
Frame verifier — eyeball the pseudo-labeled frames from exp24 and confirm the
true label (visible / hidden). Built for active learning: review the UNCERTAIN
band first (most informative), spot-check present/absent for systematic errors.

Reads : data/review_frames/manifest.csv   (written by phase2/exp24_scan_frames.py)
Writes: data/review_frames/verified.csv   (your decisions — SEPARATE from clean data)

It NEVER touches data/frames/ (the clean training set). Folding verified frames
into training is a later, explicit step (phase2/exp25_merge_retrain — TBD).

Usage: venv/bin/python3 ui/review_frames.py  →  http://localhost:8003
Keys:  V = visible · H = hidden · S / Space / → = skip · U / ← / Backspace = undo
"""
import csv, threading
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT   = Path(__file__).resolve().parent.parent
REVIEW    = PROJECT / "data" / "review_frames"
MANIFEST  = REVIEW / "manifest.csv"
VERIFIED  = REVIEW / "verified.csv"
VFIELDS   = ["frame_path", "band", "p_visible", "verified_label", "verified_at"]

app   = FastAPI()
_lock = threading.Lock()


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))

def load_verified() -> dict[str, dict]:
    if not VERIFIED.exists():
        return {}
    with open(VERIFIED) as f:
        return {r["frame_path"]: r for r in csv.DictReader(f)}

def save_verified(by_path: dict[str, dict]):
    with open(VERIFIED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VFIELDS)
        w.writeheader()
        w.writerows(by_path.values())

# Sort so the most informative frames come first: uncertain, then by distance
# from 0.5 (closest = most ambiguous).
_BAND_ORDER = {"uncertain": 0, "present": 1, "absent": 1}
def _key(r):
    return (_BAND_ORDER.get(r["band"], 2), abs(float(r["p_visible"]) - 0.5))

_frames   = sorted(load_manifest(), key=_key)
_by_path  = {r["frame_path"]: r for r in _frames}
_verified = load_verified()
_skipped: set[str] = set()          # session-only
_history: list[tuple[str, str]] = []  # (frame_path, action) for undo


def pending(band="all"):
    q = [r for r in _frames
         if r["frame_path"] not in _verified and r["frame_path"] not in _skipped]
    if band != "all":
        q = [r for r in q if r["band"] == band]
    return q


@app.get("/frame-image")
def serve_image(path: str):
    p = (PROJECT / path).resolve()
    if not p.exists() or REVIEW.resolve() not in p.parents:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="image/jpeg")

@app.get("/status")
def status():
    counts = {b: sum(1 for r in _frames if r["band"] == b)
              for b in ("uncertain", "present", "absent")}
    vis = sum(1 for v in _verified.values() if v["verified_label"] == "visible")
    hid = sum(1 for v in _verified.values() if v["verified_label"] == "hidden")
    return {"total": len(_frames), "verified": len(_verified), "skipped": len(_skipped),
            "visible": vis, "hidden": hid, "bands": counts, "can_undo": bool(_history)}

@app.get("/next")
def next_frame(band: str = "all"):
    q = pending(band)
    if not q:
        return JSONResponse({"done": True})
    return {**q[0], "queue_len": len(q), "done": False}

class Decision(BaseModel):
    frame_path: str
    label: str   # "visible" | "hidden" | "skip"

@app.post("/decide")
def decide(body: Decision):
    with _lock:
        if body.label == "skip":
            _skipped.add(body.frame_path)
        else:
            row = _by_path.get(body.frame_path, {})
            _verified[body.frame_path] = {
                "frame_path": body.frame_path,
                "band": row.get("band", ""),
                "p_visible": row.get("p_visible", ""),
                "verified_label": body.label,
                "verified_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_verified(_verified)
        _history.append((body.frame_path, body.label))
        return {"ok": True}

@app.post("/undo")
def undo():
    with _lock:
        if not _history:
            return {"ok": False, "msg": "nothing to undo"}
        fp, action = _history.pop()
        if action == "skip":
            _skipped.discard(fp)
        else:
            _verified.pop(fp, None)
            save_verified(_verified)
        return {"ok": True, "restored": fp, "was": action}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Frame Verifier</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background:#111; color:#eee; font-family:system-ui,sans-serif; height:100vh; display:flex; flex-direction:column; }
  #header { padding:12px 20px; background:#1a1a1a; border-bottom:1px solid #333; display:flex; align-items:center; gap:20px; }
  #header h1 { font-size:16px; }
  #progress-bar { flex:1; height:6px; background:#333; border-radius:3px; overflow:hidden; }
  #progress-fill { height:100%; background:#4ade80; transition:width .3s; }
  #counter { font-size:13px; color:#888; white-space:nowrap; }
  #filter-bar { padding:8px 20px; background:#161616; border-bottom:1px solid #2a2a2a; display:flex; gap:8px; }
  .filter-btn { padding:4px 14px; border-radius:20px; border:1px solid #444; background:transparent; color:#aaa; cursor:pointer; font-size:13px; }
  .filter-btn.active { background:#2563eb; border-color:#2563eb; color:#fff; }
  #main { flex:1; display:flex; overflow:hidden; }
  #img-panel { flex:1; display:flex; align-items:center; justify-content:center; padding:20px; min-width:0; }
  #img-panel img { max-width:100%; max-height:100%; object-fit:contain; border-radius:4px; }
  #side { width:300px; flex-shrink:0; border-left:1px solid #222; display:flex; flex-direction:column; padding:20px; gap:14px; }
  .badge { display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
  .badge-uncertain { background:#713f12; color:#fcd34d; }
  .badge-present { background:#14532d; color:#4ade80; }
  .badge-absent { background:#292929; color:#aaa; }
  .meta { font-size:12px; color:#666; line-height:1.9; }
  .meta span { color:#aaa; }
  .pbar { height:10px; background:#333; border-radius:5px; overflow:hidden; }
  .pfill { height:100%; background:linear-gradient(90deg,#555,#4ade80); }
  .divider { border-top:1px solid #222; }
  #btn-vis { width:100%; padding:14px; border-radius:8px; border:none; background:#16a34a; color:#fff; font-size:15px; font-weight:600; cursor:pointer; }
  #btn-hid { width:100%; padding:14px; border-radius:8px; border:none; background:#525252; color:#fff; font-size:15px; font-weight:600; cursor:pointer; }
  #btn-skip { width:100%; padding:10px; border-radius:8px; border:1px solid #333; background:transparent; color:#888; font-size:14px; cursor:pointer; }
  #btn-undo { width:100%; padding:8px; border-radius:8px; border:1px solid #333; background:transparent; color:#777; font-size:13px; cursor:pointer; }
  .keyhint { font-size:11px; color:#555; line-height:1.7; }
  .keyhint b { color:#999; }
  #toast { position:fixed; bottom:24px; right:24px; padding:10px 18px; border-radius:8px; font-size:13px; opacity:0; transition:opacity .2s; }
  #toast.show { opacity:1; }
  #toast.ok { background:#14532d; color:#4ade80; }
  #toast.undo { background:#1e3a5f; color:#93c5fd; }
  #done-screen { display:none; flex:1; align-items:center; justify-content:center; font-size:20px; color:#4ade80; }
</style></head><body>
<div id="header"><h1>Frame Verifier</h1>
  <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
  <div id="counter">–</div></div>
<div id="filter-bar">
  <button class="filter-btn active" onclick="setFilter('uncertain',event)">Uncertain (review first)</button>
  <button class="filter-btn" onclick="setFilter('present',event)">Present</button>
  <button class="filter-btn" onclick="setFilter('absent',event)">Absent</button>
  <button class="filter-btn" onclick="setFilter('all',event)">All</button>
</div>
<div id="main">
  <div id="img-panel"><img id="frame-img" src="" alt="frame"></div>
  <div id="side">
    <div><span id="badge" class="badge">–</span></div>
    <div class="meta">
      <div>Model P(visible): <span id="meta-p">–</span></div>
      <div class="pbar"><div id="pfill" class="pfill" style="width:0%"></div></div>
      <div>Model says: <span id="meta-pred">–</span></div>
      <div>Source: <span id="meta-src">–</span></div>
    </div>
    <div class="divider"></div>
    <button id="btn-vis" onclick="decide('visible')">Visible &nbsp;(V)</button>
    <button id="btn-hid" onclick="decide('hidden')">Hidden &nbsp;(H)</button>
    <button id="btn-skip" onclick="decide('skip')">Skip &nbsp;(S / →)</button>
    <button id="btn-undo" onclick="undo()">↶ Undo &nbsp;(U / ←)</button>
    <div class="keyhint"><b>V</b> visible · <b>H</b> hidden<br><b>S</b>/Space/→ skip · <b>U</b>/←/Backspace undo</div>
  </div>
  <div id="done-screen">All frames in this band reviewed ✓</div>
</div>
<div id="toast"></div>
<script>
let filter='uncertain', current=null;
function setFilter(f,e){ filter=f; document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active')); e.target.classList.add('active'); document.getElementById('main').style.display='flex'; document.getElementById('done-screen').style.display='none'; loadNext(); }
async function loadNext(){
  const d = await (await fetch(`/next?band=${filter}`)).json();
  if(d.done){ document.getElementById('main').style.display='none'; const ds=document.getElementById('done-screen'); ds.style.display='flex'; current=null; updateProgress(); return; }
  current=d;
  document.getElementById('frame-img').src=`/frame-image?path=${encodeURIComponent(d.frame_path)}`;
  const p=parseFloat(d.p_visible);
  document.getElementById('badge').textContent=d.band.toUpperCase();
  document.getElementById('badge').className=`badge badge-${d.band}`;
  document.getElementById('meta-p').textContent=(p*100).toFixed(1)+'%';
  document.getElementById('pfill').style.width=(p*100)+'%';
  document.getElementById('meta-pred').textContent=d.pred;
  document.getElementById('meta-src').textContent=`${d.date} ${d.segment} ${d.camera} @${d.t_sec}s`;
  updateProgress();
}
function updateProgress(){ fetch('/status').then(r=>r.json()).then(s=>{
  const done=s.verified+s.skipped;
  const pct=s.total>0?(done/s.total*100):0;
  document.getElementById('progress-fill').style.width=pct+'%';
  document.getElementById('counter').textContent=`${done} / ${s.total} · ${s.visible} vis / ${s.hidden} hid · ${s.skipped} skip`;
}); }
async function decide(label){
  if(!current) return;
  await fetch('/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame_path:current.frame_path,label})});
  if(label!=='skip') showToast(label,'ok');
  loadNext();
}
async function undo(){
  const r = await (await fetch('/undo',{method:'POST'})).json();
  if(r.ok){ showToast('undo: '+r.was,'undo'); loadNext(); }
  else showToast('nothing to undo','undo');
}
function showToast(m,t){ const e=document.getElementById('toast'); e.textContent=m; e.className=`show ${t}`; setTimeout(()=>e.className='',1100); }
document.addEventListener('keydown',e=>{
  if(e.key==='v'||e.key==='V') decide('visible');
  else if(e.key==='h'||e.key==='H') decide('hidden');
  else if(e.key==='s'||e.key==='S'||e.key===' '||e.key==='ArrowRight'){ e.preventDefault(); decide('skip'); }
  else if(e.key==='u'||e.key==='U'||e.key==='ArrowLeft'||e.key==='Backspace'){ e.preventDefault(); undo(); }
});
loadNext();
</script></body></html>
"""

if __name__ == "__main__":
    print(f"Loaded {len(_frames)} pseudo-labeled frames ({len(_verified)} already verified)")
    bands = {b: sum(1 for r in _frames if r['band'] == b) for b in ('uncertain','present','absent')}
    print(f"  bands: {bands}")
    print("\nOpen: http://localhost:8003")
    print("Keys: V=visible · H=hidden · S/Space/→=skip · U/←/Backspace=undo\n")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")
