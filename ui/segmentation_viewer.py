"""Segmentation viewer (FastAPI, port 8012) — watch the tiny octopus segmenter run on a video.

Pick a clip; it runs the deployed segmenter (weights/seg/octo_seg_v3_lraspp_BEST.pt) frame-by-frame,
overlays the mask + live area% on each frame, and plays the result back in the browser.

Run:  venv/bin/python3 ui/segmentation_viewer.py   ->  http://localhost:8012
"""
import sys, json, hashlib, subprocess, tempfile, threading, glob, os
from pathlib import Path

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from segment_octopus import OctoSegmenter

DEFAULT_CKPT = REPO / "weights" / "seg" / "octo_seg_v3_lraspp_BEST.pt"
OUT = REPO / "data" / "seg_viewer_cache"; OUT.mkdir(parents=True, exist_ok=True)
MASK_RGB = np.array([0, 235, 120], np.float32)   # green overlay
ALPHA = 0.45

app = FastAPI()
_SEG = None
_LOCK = threading.Lock()          # one render at a time (single model)


def seg():
    global _SEG
    if _SEG is None:
        _SEG = OctoSegmenter(str(DEFAULT_CKPT))
    return _SEG


# Hand-picked clips to show in the viewer (add more paths here as you choose them).
HANDPICKED = [
    "/Users/siddharthraj/Documents/my-projects/sentiment-analysis/src/octopus_clips_verified/2026-02-24/160002/Right_Front_1412-1432.mp4",
]


def list_clips():
    """Only the hand-picked clips."""
    out = []
    for f in HANDPICKED:
        if os.path.exists(f):
            out.append({"tag": "picked", "path": f, "name": Path(f).name})
    return out


def render_overlay(clip_path):
    """Run the segmenter per frame, overlay mask+area, return (out_mp4_path, stats)."""
    cid = hashlib.md5(clip_path.encode()).hexdigest()[:12]
    out_mp4 = OUT / f"{cid}.mp4"
    if out_mp4.exists() and out_mp4.stat().st_size > 10000:
        return out_mp4, None
    S = seg()
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    areas = []
    with tempfile.TemporaryDirectory() as td:
        i = 0
        while True:
            ret, frame = cap.read()          # BGR
            if not ret:
                break
            mask, area = S.segment(frame)     # mask HxW bool at native res
            areas.append(area)
            arr = frame.astype(np.float32)
            if mask is not None and mask.any():
                arr[mask] = (1 - ALPHA) * arr[mask] + ALPHA * MASK_RGB[::-1]  # RGB->BGR blend
            im = arr.astype(np.uint8)
            cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(im, f"octopus mask: {area*100:4.1f}% of frame", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 0), 2)
            cv2.imwrite(f"{td}/f{i:05d}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            i += 1
        cap.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}",
                        "-i", f"{td}/f%05d.jpg", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(out_mp4)], check=False)
    stats = {"frames": len(areas), "mean_area": round(float(np.mean(areas)) * 100, 2) if areas else 0,
             "max_area": round(float(np.max(areas)) * 100, 2) if areas else 0}
    return out_mp4, stats


@app.get("/")
def index():
    return HTMLResponse(PAGE)


@app.get("/api/clips")
def api_clips():
    return JSONResponse(list_clips())


@app.post("/api/render_clip")
async def api_render_clip(body: dict):
    clip = body.get("clip")
    if not clip or not os.path.exists(clip):
        return JSONResponse({"error": "clip not found"}, status_code=404)
    with _LOCK:
        out_mp4, stats = render_overlay(clip)
    cid = out_mp4.stem
    return JSONResponse({"id": cid, "stats": stats,
                         "model": DEFAULT_CKPT.name})


@app.get("/overlay/{cid}")
def overlay(cid: str):
    p = OUT / f"{cid}.mp4"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Octopus Segmentation Viewer</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#111;color:#eee;display:flex;height:100vh}
#side{width:340px;overflow:auto;border-right:1px solid #333;padding:10px}
#main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px}
h3{margin:6px 0}
.clip{padding:6px 8px;margin:2px 0;background:#1c1c1c;border-radius:6px;cursor:pointer;font-size:12px;word-break:break-all}
.clip:hover{background:#2a2a2a}
.clip.active{background:#0a5}
video{max-width:90%;max-height:75vh;background:#000;border-radius:8px}
#stats{font-size:14px;color:#9f9}
#status{color:#fc0;font-size:13px}
.tag{color:#6cf;font-weight:600}
</style></head><body>
<div id=side><h3>Clips</h3><div id=list>loading...</div></div>
<div id=main>
  <div id=status>Pick a clip to run the segmenter (weights/seg/octo_seg_v3_lraspp_BEST.pt)</div>
  <video id=vid controls autoplay loop></video>
  <div id=stats></div>
</div>
<script>
let cur=null;
async function load(){
  const cs=await (await fetch('/api/clips')).json();
  const L=document.getElementById('list'); L.innerHTML='';
  cs.forEach(c=>{const d=document.createElement('div');d.className='clip';
    d.innerHTML='<span class=tag>['+c.tag+']</span> '+c.name.replace(c.tag+': ','');
    d.onclick=()=>run(c,d);L.appendChild(d);});
}
async function run(c,el){
  document.querySelectorAll('.clip').forEach(x=>x.classList.remove('active'));el.classList.add('active');
  document.getElementById('status').textContent='Segmenting '+c.name+' ... (first run renders frame-by-frame)';
  document.getElementById('stats').textContent='';
  const r=await (await fetch('/api/render_clip',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({clip:c.path})})).json();
  if(r.error){document.getElementById('status').textContent='Error: '+r.error;return;}
  document.getElementById('vid').src='/overlay/'+r.id+'?t='+Date.now();
  document.getElementById('status').textContent='Model: '+r.model;
  if(r.stats)document.getElementById('stats').textContent=
     'frames '+r.stats.frames+'  |  mean mask area '+r.stats.mean_area+'%  |  peak '+r.stats.max_area+'%';
}
load();
</script></body></html>"""


if __name__ == "__main__":
    print("Segmentation viewer -> http://localhost:8012")
    uvicorn.run(app, host="127.0.0.1", port=8012)
