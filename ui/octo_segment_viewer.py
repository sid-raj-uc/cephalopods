"""octo_segment_viewer.py — live octopus segmentation viewer (FastAPI, port 8014).

Runs the trained tiny segmenter (default weights/octo_seg_points_lraspp.pt — the points-on model)
frame-by-frame on a local video, overlays the octopus mask (green) + live area%, and plays the
result back in the browser with an area-over-time strip. Pick a suggested clip or paste any local
video path.

Run:  venv/bin/python3 ui/octo_segment_viewer.py   ->  http://localhost:8014
Model override:  SEG_CKPT=weights/seg/octo_seg_v3_lraspp_BEST.pt venv/bin/python3 ui/octo_segment_viewer.py
"""
import sys, os, json, hashlib, subprocess, tempfile, threading, glob, random
from pathlib import Path
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from segment_octopus import OctoSegmenter, _largest_blob

CKPT = os.environ.get("SEG_CKPT", str(REPO / "weights" / "octo_seg_points_lraspp.pt"))
CLIPS_ROOT = REPO / "src" / "octopus_clips_verified"
OUT = REPO / "data" / "seg_viewer_new_cache"; OUT.mkdir(parents=True, exist_ok=True)
MASK_RGB = np.array([0, 235, 90], np.float32)
ALPHA = 0.5

app = FastAPI()
_SEG = None
_LOCK = threading.Lock()


def seg():
    global _SEG
    if _SEG is None:
        _SEG = OctoSegmenter(CKPT)
    return _SEG


def camera_of(p):
    for c in ("Right_Front", "Right_Back", "Right_Right", "Right_Left", "Right_Top"):
        if c in Path(p).name:
            return c
    return "?"


def suggest_clips(n=48):
    """A spread of local clips across cameras/dates so there's variety to try."""
    allc = glob.glob(str(CLIPS_ROOT / "**" / "*.mp4"), recursive=True)
    by = {}
    for p in allc:
        by.setdefault(camera_of(p), []).append(p)
    out = []
    rng = random.Random(0)
    for cam, lst in by.items():
        lst = sorted(lst); rng.shuffle(lst)
        for p in lst[: max(1, n // max(1, len(by)))]:
            out.append({"path": p, "camera": cam, "name": "/".join(Path(p).parts[-3:])})
    return sorted(out, key=lambda d: (d["camera"], d["name"]))


EMA_ALPHA = 0.45          # temporal smoothing: lower = smoother (more lag), higher = snappier
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def render_overlay(clip_path, smooth=True):
    cid = hashlib.md5((CKPT + "|" + clip_path + ("|s" if smooth else "|r")).encode()).hexdigest()[:12]
    out_mp4 = OUT / f"{cid}.mp4"
    stats_f = OUT / f"{cid}.json"
    if out_mp4.exists() and out_mp4.stat().st_size > 10000 and stats_f.exists():
        return out_mp4, json.loads(stats_f.read_text())
    S = seg()
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    areas = []
    ema = None
    with tempfile.TemporaryDirectory() as td:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            prob = S.prob(frame)                       # (in_size, in_size) float
            used = prob
            if smooth:
                ema = prob if ema is None else (EMA_ALPHA * prob + (1 - EMA_ALPHA) * ema)
                used = ema
            m = used > 0.5
            if m.any():
                m = _largest_blob(m)
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL).astype(bool)
            area = float(m.mean())
            areas.append(round(area, 4))
            # downscale the display frame first (source is often 4K), then paint the mask on it
            disp = frame
            if disp.shape[1] > 1280:
                nh = (round(disp.shape[0] * 1280 / disp.shape[1]) // 2) * 2
                disp = cv2.resize(disp, (1280, nh))
            mask_disp = cv2.resize(m.astype(np.uint8), (disp.shape[1], disp.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            arr = disp.astype(np.float32)
            if mask_disp.any():
                arr[mask_disp] = (1 - ALPHA) * arr[mask_disp] + ALPHA * MASK_RGB[::-1]
            im = arr.astype(np.uint8)
            cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
            present = area >= 0.01
            txt = f"octopus {area*100:4.1f}%" if present else "no octopus"
            col = (120, 255, 0) if present else (120, 120, 120)
            tag = "  [smoothed]" if smooth else "  [raw]"
            cv2.putText(im, txt + tag, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imwrite(f"{td}/f{i:05d}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            i += 1
        cap.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}",
                        "-i", f"{td}/f%05d.jpg", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(out_mp4)], check=False)
    stats = {"frames": len(areas), "fps": round(fps, 2), "smooth": smooth,
             "mean_area": round(float(np.mean(areas)) * 100, 2) if areas else 0,
             "max_area": round(float(np.max(areas)) * 100, 2) if areas else 0,
             "present_frac": round(float(np.mean([a >= 0.01 for a in areas])) * 100, 1) if areas else 0,
             "areas": areas, "model": Path(CKPT).name}
    stats_f.write_text(json.dumps(stats))
    return out_mp4, stats


@app.get("/")
def index():
    return HTMLResponse(PAGE)


@app.get("/api/clips")
def api_clips():
    return JSONResponse(suggest_clips())


@app.post("/api/render")
async def api_render(body: dict):
    clip = (body.get("clip") or "").strip()
    if not clip or not os.path.exists(clip):
        return JSONResponse({"error": f"video not found: {clip}"}, status_code=404)
    smooth = bool(body.get("smooth", True))
    with _LOCK:
        out_mp4, stats = render_overlay(clip, smooth=smooth)
    return JSONResponse({"id": out_mp4.stem, "stats": stats})


@app.get("/video/{cid}")
def video(cid: str):
    p = OUT / f"{cid}.mp4"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Octopus Segmentation Viewer</title>
<style>
:root{--bg:#0b0f12;--panel:#141a1f;--line:#232c33;--ink:#e8f0f0;--dim:#93a3a6;--green:#3fe08a;--cy:#4fd6c4}
*{box-sizing:border-box}body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink);display:flex;height:100vh}
#side{width:330px;flex-shrink:0;overflow:auto;border-right:1px solid var(--line);padding:12px}
#main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:16px}
h3{margin:4px 0 10px;font-size:14px}
.model{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--cy);margin-bottom:12px;word-break:break-all}
input{width:100%;padding:8px;border-radius:7px;border:1px solid var(--line);background:#0c1216;color:var(--ink);font-size:12px}
button{margin-top:8px;width:100%;padding:9px;border:0;border-radius:7px;background:var(--green);color:#062;font-weight:700;cursor:pointer}
.clip{padding:6px 8px;margin:2px 0;background:var(--panel);border-radius:6px;cursor:pointer;font-size:11.5px;word-break:break-all;border:1px solid transparent}
.clip:hover{border-color:var(--line)}.clip.active{background:#123;border-color:var(--green)}
.cam{color:var(--cy);font-weight:700;font-family:ui-monospace,monospace;font-size:10px}
video{max-width:96%;max-height:66vh;background:#000;border-radius:10px;border:1px solid var(--line)}
#status{color:#fc6;font-size:13px;min-height:18px}
#stats{font-size:13px;color:var(--dim)}#stats b{color:var(--green)}
#strip{width:96%;height:38px;background:var(--panel);border-radius:6px;border:1px solid var(--line)}
.hint{font-size:11px;color:var(--dim);margin-top:4px}
</style></head><body>
<div id=side>
  <h3>Octopus Segmentation</h3>
  <div class=model id=model>loading model…</div>
  <input id=path placeholder="paste any local video path…">
  <label style="display:flex;align-items:center;gap:7px;margin-top:9px;font-size:12.5px;color:var(--dim)">
    <input type=checkbox id=smooth checked style="width:auto"> temporal smoothing (no jitter)</label>
  <button onclick="runPath()">▶ Run segmentation</button>
  <div class=hint>or pick a clip below</div>
  <div id=list style="margin-top:10px">loading clips…</div>
</div>
<div id=main>
  <div id=status>Pick a clip or paste a path, then Run.</div>
  <video id=vid controls autoplay loop muted playsinline></video>
  <canvas id=strip></canvas>
  <div id=stats></div>
</div>
<script>
let cur=null;
async function load(){
  const cs=await (await fetch('/api/clips')).json();
  const L=document.getElementById('list');L.innerHTML='';
  cs.forEach(c=>{const d=document.createElement('div');d.className='clip';
    d.innerHTML='<span class=cam>'+c.camera+'</span> '+c.name;
    d.onclick=()=>run(c.path,d);L.appendChild(d);});
}
function runPath(){const p=document.getElementById('path').value.trim();if(p)run(p,null);}
async function run(path,el){
  document.querySelectorAll('.clip').forEach(x=>x.classList.remove('active'));if(el)el.classList.add('active');
  document.getElementById('status').textContent='Segmenting (first run renders frame-by-frame)…';
  document.getElementById('stats').textContent='';
  const smooth=document.getElementById('smooth').checked;
  const r=await (await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip:path,smooth:smooth})})).json();
  if(r.error){document.getElementById('status').textContent='Error: '+r.error;return;}
  const v=document.getElementById('vid');v.src='/video/'+r.id+'?t='+Date.now();
  const s=r.stats;document.getElementById('model').textContent='model: '+s.model;
  document.getElementById('status').textContent='Done — '+path.split('/').slice(-1);
  document.getElementById('stats').innerHTML='frames '+s.frames+' @ '+s.fps+'fps &nbsp;|&nbsp; mean mask <b>'+s.mean_area+'%</b> &nbsp;|&nbsp; peak '+s.max_area+'% &nbsp;|&nbsp; present <b>'+s.present_frac+'%</b> of frames';
  drawStrip(s.areas);
}
function drawStrip(areas){
  const c=document.getElementById('strip');const w=c.clientWidth,h=c.clientHeight;c.width=w;c.height=h;
  const ctx=c.getContext('2d');ctx.clearRect(0,0,w,h);
  const mx=Math.max(0.05,...areas);
  ctx.fillStyle='#3fe08a';
  areas.forEach((a,i)=>{const x=i/areas.length*w,bw=Math.max(1,w/areas.length),bh=a/mx*(h-4);
    ctx.fillStyle=a>=0.01?'#3fe08a':'#2a3540';ctx.fillRect(x,h-bh,bw,bh);});
  ctx.strokeStyle='#4fd6c4';ctx.setLineDash([3,3]);const y=h-(0.01/mx)*(h-4);
  ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();
}
load();
</script></body></html>"""


if __name__ == "__main__":
    print(f"Octopus segmentation viewer  (model: {Path(CKPT).name})  -> http://localhost:8014")
    uvicorn.run(app, host="127.0.0.1", port=8014)
