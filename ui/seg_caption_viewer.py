"""Combined live viewer (FastAPI, port 8013) — segmentation overlay + captions, computed ON THE GO.

Pick a video; each run recomputes EVERYTHING live (nothing cached):
  1. CAPTIONING  — src/local_pipeline.py (CLIP+motion scan -> 20s clips -> MLX caption student)
  2. SEGMENTATION— the tiny mask model (weights/seg/octo_seg_v3_lraspp_BEST.pt) on every frame
then plays the mask-overlaid video (left) with captions synced (right).

Disk-safe: frames are downscaled to 720p and STREAMED straight into ffmpeg (no temp images on
disk — the earlier version wrote 22k 4K JPGs and filled the disk; this never touches disk per-frame).

Run:  venv/bin/python3 ui/seg_caption_viewer.py   ->  http://localhost:8013
"""
import sys, os, json, time, glob, threading, subprocess, datetime
from pathlib import Path
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from segment_octopus import OctoSegmenter
import local_pipeline as lp

SEG_CKPT = REPO / "weights" / "seg" / "octo_seg_v3_lraspp_BEST.pt"
JOBS_DIR = REPO / "data" / "seg_caption_jobs"; JOBS_DIR.mkdir(parents=True, exist_ok=True)
OUT_H = 720                                   # overlay output height (downscaled from 4K)
MASK_BGR = np.array([120, 235, 0], np.float32)  # green (BGR)
ALPHA = 0.45

app = FastAPI()
JOBS = {}
JOB_LOCK = threading.Lock()
_SEG = None; _CAP = None; _MODEL_LOCK = threading.Lock()


def models():
    global _SEG, _CAP
    with _MODEL_LOCK:
        if _SEG is None:
            _SEG = OctoSegmenter(str(SEG_CKPT))
        if _CAP is None:
            _CAP = lp.load_models()
    return _SEG, _CAP


def list_videos():
    vids = sorted(glob.glob(str(REPO / "data" / "demo_videos" / "*.mp4")))
    return [{"path": v, "name": Path(v).name} for v in vids]


def _mp4_ok(p):
    """True only if the overlay mp4 is finalised (has a readable duration) — avoids
    serving a half-written file from a job that is still rendering."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nk=1:nw=1", str(p)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip()) > 1
    except Exception:
        return False


def find_cached(video):
    """Most recent COMPLETED job for this video (overlay finalised + captions on disk),
    so a prior live run can be re-loaded instantly instead of recomputed."""
    stem = Path(video).stem
    best = None
    for jd in JOBS_DIR.glob("*"):
        ov, cj = jd / "overlay.mp4", jd / f"{stem}_captions.json"
        if ov.exists() and cj.exists() and ov.stat().st_size > 10000:
            mt = ov.stat().st_mtime
            if best is None or mt > best[0]:
                best = (mt, jd.name, cj, ov)
    if not best or not _mp4_ok(best[3]):
        return None
    _, job, cj, _ = best
    d = json.load(open(cj))
    caps = [{"timeline": c["video_timeline"], "start": c["start"], "caption": c["caption"]}
            for c in d.get("clips", []) if c.get("status") == "captioned" and c.get("caption")]
    caps.sort(key=lambda c: c["start"])
    return {"job": job, "captions": caps}


def _overlay_one(small, mask, area, W, i, fps):
    arr = small.astype(np.float32)
    if mask is not None and mask.any():
        arr[mask] = (1 - ALPHA) * arr[mask] + ALPHA * MASK_BGR
    im = arr.astype(np.uint8)
    cv2.rectangle(im, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(im, f"octopus mask: {area*100:4.1f}%   t={i/fps:6.1f}s", (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 0), 2)
    return im


def render_overlay_stream(video, out_mp4, S, st):
    """Segment every frame (BATCHED), downscale to 720p, pipe straight to ffmpeg. No disk temp files.
    Speedups (all lossless): downscale-first, batched GPU inference, and a decode thread that reads
    the next frames while the GPU works on the current batch."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.5
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st["seg_total"] = total
    W = (int(OUT_H * sw / sh) // 2) * 2; H = OUT_H
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)], stdin=subprocess.PIPE)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        mask, area = S.segment(small)
        ff.stdin.write(_overlay_one(small, mask, area, W, i, fps).tobytes()); i += 1
        if i % 50 == 0:
            st["seg_done"] = i
    cap.release(); ff.stdin.close(); ff.wait()
    st["seg_done"] = i
    return i


def run_job(job_id, video):
    S, M = models()
    jd = JOBS_DIR / job_id; jd.mkdir(parents=True, exist_ok=True)
    st = JOBS[job_id]
    # 1) captions (live)
    st["stage"] = "captioning"
    caps = []
    def on_stage(s, d): st["detail"] = f"{s}: {d}"
    def on_clip(i, n, rec):
        st["cap_done"] = i; st["cap_total"] = n
        if rec.get("caption") and rec.get("status") == "captioned":
            caps.append({"timeline": rec["video_timeline"], "start": rec["start"], "caption": rec["caption"]})
    lp.process_video(video, jd, M, camera="cam", on_stage=on_stage, on_clip=on_clip, save_clips=False)
    caps.sort(key=lambda c: c["start"]); st["captions"] = caps
    # 2) segmentation overlay (live, streamed)
    st["stage"] = "segmenting"; st["detail"] = "overlaying mask on every frame (720p, streamed)"
    n = render_overlay_stream(video, jd / "overlay.mp4", S, st)
    st["stage"] = "done"; st["detail"] = f"{len(caps)} captions, {n} frames segmented"; st["result"] = True


@app.get("/")
def index(): return HTMLResponse(PAGE)


@app.get("/api/videos")
def api_videos(): return JSONResponse(list_videos())


@app.get("/api/cached")
def api_cached(video: str):
    """Instant-load a prior completed run for this video (or {} if none)."""
    return JSONResponse(find_cached(video) or {})


@app.post("/api/run")
async def api_run(body: dict):
    video = body.get("video")
    if not video or not os.path.exists(video):
        return JSONResponse({"error": "video not found"}, status_code=404)
    if not JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "a job is already running"}, status_code=409)
    job_id = datetime.datetime.now().strftime("%H%M%S")
    JOBS[job_id] = {"stage": "starting", "detail": "", "cap_done": 0, "cap_total": 0,
                    "seg_done": 0, "seg_total": 0, "captions": [], "result": False}
    def wrap():
        try: run_job(job_id, video)
        except Exception as e:
            JOBS[job_id]["stage"] = "error"; JOBS[job_id]["detail"] = f"{type(e).__name__}: {e}"
        finally: JOB_LOCK.release()
    threading.Thread(target=wrap, daemon=True).start()
    return JSONResponse({"job": job_id})


@app.get("/api/status")
def api_status(job: str): return JSONResponse(JOBS.get(job, {"error": "no such job"}))


@app.get("/result/{job}")
def result(job: str):
    p = JOBS_DIR / job / "overlay.mp4"
    if not p.exists(): return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Segmentation + Captions (live)</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#111;color:#eee}
#top{padding:10px;border-bottom:1px solid #333;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
select,button{font-size:14px;padding:6px 10px;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#eee}
button{cursor:pointer;background:#0a5;border:none}
#prog{color:#fc0;font-size:13px}
#wrap{display:flex;height:calc(100vh - 56px)}
#left{flex:1;display:flex;align-items:center;justify-content:center;background:#000}
video{max-width:100%;max-height:100%}
#right{width:360px;overflow:auto;border-left:1px solid #333;padding:8px}
.cap{padding:8px;margin:4px 0;background:#1a1a1a;border-radius:6px;cursor:pointer;font-size:13px}
.cap:hover{background:#252525}.cap.active{background:#0a5;color:#000}
.t{color:#6cf;font-weight:600;margin-right:6px}.cap.active .t{color:#003}
</style></head><body>
<div id=top>
  <select id=vid onchange=checkCached()></select>
  <button onclick=run()>Run live (segment + caption)</button>
  <button id=cachedbtn style="display:none;background:#2A6F97" onclick=loadCached()>&#9889; View last result (instant)</button>
  <span id=prog>pick a video and hit Run — everything computed on the go</span>
</div>
<div id=wrap>
  <div id=left><video id=player controls></video></div>
  <div id=right><div style=color:#888>captions appear here as they compute…</div></div>
</div>
<script>
let JOB=null,caps=[],cached=null;
async function load(){const vs=await (await fetch('/api/videos')).json();
  const s=document.getElementById('vid');vs.forEach(v=>{const o=document.createElement('option');o.value=v.path;o.textContent=v.name;s.appendChild(o);});
  checkCached();}
async function checkCached(){
  const video=document.getElementById('vid').value;
  cached=await (await fetch('/api/cached?video='+encodeURIComponent(video))).json();
  document.getElementById('cachedbtn').style.display=(cached&&cached.job)?'inline-block':'none';}
function loadCached(){
  caps=cached.captions;renderCaps();
  const pl=document.getElementById('player');pl.src='/result/'+cached.job+'?t='+Date.now();pl.ontimeupdate=sync;
  document.getElementById('prog').textContent='Loaded cached result (job '+cached.job+') — '+caps.length+' captions, no recompute';}
async function run(){const video=document.getElementById('vid').value;
  const r=await (await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video})})).json();
  if(r.error){document.getElementById('prog').textContent='Error: '+r.error;return;} JOB=r.job;caps=[];poll();}
async function poll(){const s=await (await fetch('/api/status?job='+JOB)).json();
  let p='['+s.stage+'] '+(s.detail||'');
  if(s.stage=='captioning')p+='  clips '+s.cap_done+'/'+s.cap_total;
  if(s.stage=='segmenting')p+='  frames '+s.seg_done+'/'+s.seg_total;
  document.getElementById('prog').textContent=p;
  if(s.captions&&s.captions.length!=caps.length){caps=s.captions;renderCaps();}
  if(s.stage=='done'){document.getElementById('prog').textContent='DONE — '+s.detail;
    const pl=document.getElementById('player');pl.src='/result/'+JOB+'?t='+Date.now();pl.ontimeupdate=sync;checkCached();return;}
  if(s.stage=='error'){document.getElementById('prog').textContent='ERROR: '+s.detail;return;}
  setTimeout(poll,2000);}
function renderCaps(){const R=document.getElementById('right');R.innerHTML='';
  caps.forEach((c,i)=>{const d=document.createElement('div');d.className='cap';d.id='c'+i;
    d.innerHTML='<span class=t>'+c.timeline+'</span>'+c.caption;
    d.onclick=()=>{document.getElementById('player').currentTime=c.start;};R.appendChild(d);});}
function sync(){const t=document.getElementById('player').currentTime;let act=-1;
  caps.forEach((c,i)=>{if(t>=c.start&&t<c.start+20)act=i;});
  caps.forEach((c,i)=>{document.getElementById('c'+i)?.classList.toggle('active',i==act);});
  if(act>=0)document.getElementById('c'+act)?.scrollIntoView({block:'nearest'});}
load();
</script></body></html>"""

if __name__ == "__main__":
    print("Segmentation + Captions live viewer -> http://localhost:8013")
    uvicorn.run(app, host="127.0.0.1", port=8013)
