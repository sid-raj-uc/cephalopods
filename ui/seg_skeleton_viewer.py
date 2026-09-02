"""Seg -> Smoothed -> Skeleton viewer (FastAPI, port 8017).

Pick/enter a local video; it runs src/seg_skeleton_pipeline.process_video_3way and plays the THREE
synchronized overlay videos side by side:
  1) RAW segmentation   2) SMOOTHED segmentation   3) SKELETON (tracked anatomical graph)

Run: venv/bin/python3 ui/seg_skeleton_viewer.py  -> http://localhost:8017
"""
import sys, threading, time, uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from seg_skeleton_pipeline import process_video_3way, DEFAULT_CKPT
from segment_octopus import OctoSegmenter

CACHE = REPO / "data" / "seg_skeleton_cache"; CACHE.mkdir(parents=True, exist_ok=True)
CLIPS_ROOT = REPO / "src" / "octopus_clips_verified"
app = FastAPI()
JOBS = {}
_LOCK = threading.Lock()
_S = None


def seg():
    global _S
    if _S is None:
        _S = OctoSegmenter(str(DEFAULT_CKPT))
    return _S


def run_job(job_id, clip, fps, refine=False):
    st = JOBS[job_id]
    try:
        def on_stage(s):
            st["stage"] = s
        r = process_video_3way(clip, CACHE / job_id, S=seg(), fps=fps, on_stage=on_stage,
                               refine=refine)
        st.update(r); st["stage"] = "done"; st["done"] = True
    except Exception as exc:
        st["stage"] = f"error: {exc}"; st["done"] = True; st["error"] = str(exc)


@app.get("/api/suggestions")
def suggestions():
    # a handful of colour-camera clips (the model is colour-trained)
    out = []
    for p in sorted(CLIPS_ROOT.glob("**/*.mp4")):
        if any(c in p.name for c in ("Right_Front", "Right_Back", "Right_Right")):
            out.append(str(p))
        if len(out) >= 40:
            break
    return {"clips": out}


@app.post("/api/run")
def run(body: dict):
    clip = (body.get("clip") or "").strip()
    if not clip or not Path(clip).exists():
        return JSONResponse({"error": f"not found: {clip}"}, status_code=404)
    with _LOCK:
        if any(not j["done"] for j in JOBS.values()):
            return JSONResponse({"error": "a job is already running"}, status_code=429)
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"stage": "queued", "done": False, "clip": clip}
        threading.Thread(target=run_job, args=(job_id, clip, float(body.get("fps", 5)),
                                               bool(body.get("refine", False))), daemon=True).start()
    return {"job": job_id}


@app.get("/api/status/{job}")
def status(job: str):
    return JOBS.get(job, {"error": "unknown job"})


@app.get("/video/{which}/{job}")
def video(which: str, job: str):
    p = CACHE / job / f"{which}.mp4"
    if not p.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Seg -> Skeleton</title><style>
 body{margin:0;background:#0f1013;color:#ddd;font:14px system-ui}
 #bar{padding:10px 14px;background:#1a1c22;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 select,input,button{background:#25272e;color:#eee;border:1px solid #3a3d46;border-radius:6px;padding:7px 10px}
 button{cursor:pointer} button:hover{background:#30333c}
 #msg{color:#6cf}
 #vids{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;padding:12px}
 .pane{background:#000;border:1px solid #2a2d35;border-radius:8px;overflow:hidden}
 .pane h3{margin:0;padding:7px 10px;font-size:12.5px;background:#1a1c22;color:#8f8}
 video{width:100%;display:block;background:#000}
 .hint{color:#777;font-size:12px}
</style></head><body>
<div id=bar>
 <b>Segmentation → Smoothed → Skeleton</b>
 <select id=sug><option value="">— pick a clip —</option></select>
 <input id=path placeholder="or paste a local video path" size=42>
 <label class=hint>fps <input id=fps type=number value=5 min=1 max=10 style="width:56px"></label>
 <label class=hint><input type=checkbox id=refine> SAM2 refine (best quality, ~2 min extra)</label>
 <button onclick="go()">▶ Run pipeline</button>
 <span id=msg></span>
</div>
<div id=vids>
 <div class=pane><h3>1) RAW segmentation</h3><video id=v1 controls loop muted></video></div>
 <div class=pane><h3 style="color:#9f9">2) SMOOTHED segmentation</h3><video id=v2 controls loop muted></video></div>
 <div class=pane><h3 style="color:#fd6">3) SKELETON (tracked)</h3><video id=v3 controls loop muted></video></div>
</div>
<script>
async function load(){const d=await (await fetch('/api/suggestions')).json();const s=document.getElementById('sug');
 (d.clips||[]).forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c.split('/').slice(-3).join('/');s.appendChild(o);});}
function msg(t){document.getElementById('msg').textContent=t;}
async function go(){
 const clip=document.getElementById('path').value.trim()||document.getElementById('sug').value;
 if(!clip){msg('pick or paste a clip');return;}
 const fps=document.getElementById('fps').value;
 const refine=document.getElementById('refine').checked;
 msg('starting…');
 const r=await (await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,fps,refine})})).json();
 if(r.error){msg(r.error);return;}
 const job=r.job; poll(job);
}
async function poll(job){
 const s=await (await fetch('/api/status/'+job)).json();
 msg('stage: '+(s.stage||'…'));
 if(s.done){
   if(s.error){msg('error: '+s.error);return;}
   const bust='?t='+Date.now();
   v1.src='/video/raw/'+job+bust; v2.src='/video/smooth/'+job+bust; v3.src='/video/skeleton/'+job+bust;
   [v1,v2,v3].forEach(v=>{v.load();v.play().catch(()=>{});});
   msg(`done — ${s.n_present} present frames, ${s.n_tracked} skeleton-tracked @ ${s.fps} fps`);
   // keep the three roughly in sync
   v1.onplay=()=>{v2.currentTime=v1.currentTime;v3.currentTime=v1.currentTime;v2.play();v3.play();};
   return;
 }
 setTimeout(()=>poll(job),1200);
}
load();
</script></body></html>"""

if __name__ == "__main__":
    print("Seg -> Skeleton viewer -> http://localhost:8017")
    uvicorn.run(app, host="127.0.0.1", port=8017)
