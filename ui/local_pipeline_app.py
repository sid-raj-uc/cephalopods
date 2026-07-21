"""
Local pipeline UI — video in → octopus clips → captions, with timeline (FastAPI, port 8010).

Pick a suggested server video (ones known to contain a visible, active octopus) or point it at
your own local video file; it runs the optimized local pipeline (`src/local_pipeline.py`: concurrent
scan + score-reuse captioning with the MLX 4-bit student) and shows each extracted clip inline next
to its caption and video-timeline, streaming results as they are produced.

Run:  venv/bin/python3 ui/local_pipeline_app.py   ->  http://localhost:8010
"""
import sys, json, uuid, threading, subprocess
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"; sys.path.insert(0, str(SRC))
from server_creds import USER, PASS
import local_pipeline as lp

INDEX_JSON = SRC / "octopus_clips_verified.json"
JOBS_DIR = REPO / "local_pipeline_out" / "ui_jobs"; JOBS_DIR.mkdir(parents=True, exist_ok=True)
DL_DIR = REPO / "local_pipeline_out" / "downloads"; DL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
JOBS = {}                 # job_id -> state dict
JOB_LOCK = threading.Lock()   # one pipeline job at a time (single shared GPU/model)
MODELS = None
MODELS_LOCK = threading.Lock()


def models():
    global MODELS
    with MODELS_LOCK:
        if MODELS is None:
            MODELS = lp.load_models()
    return MODELS


def suggestions(n=8):
    """Top source videos from the clip index, ranked by # confirmed-present-octopus clips."""
    if not INDEX_JSON.exists():
        return []
    idx = json.load(open(INDEX_JSON))
    by = defaultdict(lambda: {"present": 0, "total": 0})
    for c in idx.get("clips", []):
        key = (c.get("date"), c.get("segment"), c.get("camera"))
        g = by[key]; g["total"] += 1; g["url"] = c.get("video_url")
        g["date"], g["segment"], g["camera"] = c.get("date"), c.get("segment"), c.get("camera")
        cap = (c.get("caption_235b") or c.get("caption") or "").lower()
        if cap and "not present" not in cap:
            g["present"] += 1
    ranked = sorted((g for g in by.values() if g.get("url")), key=lambda g: g["present"], reverse=True)
    return [{"date": g["date"], "segment": g["segment"], "camera": g["camera"], "video_url": g["url"],
             "present": g["present"], "total": g["total"]} for g in ranked[:n]]


def _download(video_url, dest: Path, job):
    if dest.exists() and dest.stat().st_size > 100000:
        return
    job["stage"] = "downloading"; job["detail"] = f"{dest.name} …"
    au = video_url.replace("https://", f"https://{USER}:{PASS}@")
    subprocess.run(["curl", "-s", "-o", str(dest), au], check=True)


def run_job(job_id, source, video_url, local_path, camera):
    job = JOBS[job_id]
    try:
        with JOB_LOCK:
            if source == "server":
                stem = f"{video_url.rstrip('/').split('/')[-1].split('--')[0]}_{camera}"
                dest = DL_DIR / f"{stem}.mp4"
                _download(video_url, dest, job)
                video_path = dest
            else:
                video_path = Path(local_path)
                if not video_path.exists():
                    raise FileNotFoundError(f"no such file: {local_path}")
            out_dir = JOBS_DIR / job_id
            def on_stage(s, d): job["stage"] = s; job["detail"] = d
            def on_clip(i, n, rec):
                job["total"] = n
                job["clips"].append({
                    "i": len(job["clips"]), "timeline": rec["video_timeline"], "status": rec["status"],
                    "caption": rec.get("caption"), "max_p": rec.get("max_p_visible"),
                    "clip_name": rec.get("clip_name")})
            res = lp.process_video(video_path, out_dir, models(), camera=camera,
                                   on_stage=on_stage, on_clip=on_clip)
            job["elapsed"] = res["elapsed_sec"]; job["n_clips"] = res["n_clips"]
        job["stage"] = "done"; job["done"] = True
    except Exception as e:
        job["error"] = str(e); job["stage"] = "error"; job["done"] = True


@app.get("/api/suggestions")
def api_suggestions():
    return JSONResponse(suggestions())


@app.post("/api/run")
async def api_run(req: Request):
    b = await req.json()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"stage": "queued", "detail": "", "clips": [], "total": None,
                    "done": False, "error": None, "camera": b.get("camera", "cam")}
    threading.Thread(target=run_job, args=(job_id, b.get("source"), b.get("video_url"),
                     b.get("local_path"), b.get("camera", "cam")), daemon=True).start()
    return JSONResponse({"job": job_id})


@app.get("/api/status")
def api_status(job: str):
    j = JOBS.get(job)
    if not j:
        return JSONResponse({"error": "no such job"}, status_code=404)
    busy = JOB_LOCK.locked()
    return JSONResponse({**{k: v for k, v in j.items() if k != "clips"},
                         "clips": j["clips"], "queued_behind": busy and j["stage"] == "queued"})


@app.get("/clip")
def clip(job: str, i: int):
    j = JOBS.get(job)
    if not j or i >= len(j["clips"]):
        return JSONResponse({"error": "bad clip"}, status_code=404)
    name = j["clips"][i].get("clip_name")
    p = JOBS_DIR / job / "clips" / name
    if not p.exists():
        return JSONResponse({"error": "missing file"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Octopus pipeline</title>
<style>
 :root{--bg:#0f1216;--card:#1a1f27;--fg:#e6e9ee;--mut:#8b95a5;--acc:#4aa8ff;--ok:#39d98a;--warn:#f0a24b}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:18px 24px;border-bottom:1px solid #262c36}
 h1{margin:0;font-size:19px} .sub{color:var(--mut);font-size:13px;margin-top:3px}
 main{max-width:1100px;margin:0 auto;padding:22px}
 h2{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:26px 0 12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
 .card{background:var(--card);border:1px solid #262c36;border-radius:10px;padding:14px}
 .card b{font-size:14px} .card .meta{color:var(--mut);font-size:12px;margin:6px 0 10px}
 .pill{display:inline-block;background:#12351f;color:var(--ok);border-radius:20px;padding:1px 9px;font-size:11px}
 button{background:var(--acc);color:#04121f;border:0;border-radius:7px;padding:8px 14px;
   font-weight:600;cursor:pointer;font-size:13px} button:disabled{opacity:.5;cursor:default}
 button.ghost{background:#232a34;color:var(--fg)}
 input[type=text]{background:#0c0f13;border:1px solid #2a313c;color:var(--fg);border-radius:7px;
   padding:9px 11px;width:100%;font-size:13px}
 .row{display:flex;gap:10px;align-items:center}
 #status{background:var(--card);border:1px solid #262c36;border-radius:10px;padding:14px 16px;margin:12px 0}
 .bar{height:7px;background:#0c0f13;border-radius:6px;overflow:hidden;margin-top:9px}
 .bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .3s}
 .clip{background:var(--card);border:1px solid #262c36;border-radius:10px;padding:12px;margin:10px 0;
   display:grid;grid-template-columns:300px 1fr;gap:14px;align-items:start}
 .clip video{width:300px;border-radius:7px;background:#000} .tl{color:var(--acc);font-weight:600;font-size:13px}
 .cap{margin-top:6px} .abs{color:var(--warn)} .muted{color:var(--mut);font-size:12px}
 @media(max-width:640px){.clip{grid-template-columns:1fr}.clip video{width:100%}}
</style></head><body>
<header><h1>🐙 Local octopus pipeline — video → clips → captions</h1>
 <div class=sub>Runs on your Mac: CLIP+motion extraction · MLX 4-bit caption student · concurrent scan + score-reuse</div></header>
<main>
 <h2>Suggested videos <span class=muted>(known to contain an active, visible octopus)</span></h2>
 <div id=suggs class=grid><div class=muted>loading…</div></div>
 <h2>Or run your own local video</h2>
 <div class=row><input id=localpath type=text placeholder="/full/path/to/your/30min_video.mp4">
   <input id=localcam type=text style="max-width:160px" placeholder="camera label" value="cam">
   <button id=runlocal class=ghost>Run</button></div>
 <div id=panel style=display:none>
  <h2>Run</h2>
  <div id=status></div>
  <div id=clips></div>
 </div>
</main>
<script>
let JOB=null, timer=null, shown=0;
const $=s=>document.querySelector(s);
async function loadSuggs(){
  const r=await fetch('/api/suggestions'); const s=await r.json();
  $('#suggs').innerHTML = s.length? s.map(v=>`<div class=card>
    <b>${v.camera}</b> · <span class=muted>${v.date}</span>
    <div class=meta>segment ${v.segment} · <span class=pill>${v.present} active clips</span></div>
    <button onclick='run({source:"server",video_url:${JSON.stringify(v.video_url)},camera:${JSON.stringify(v.camera)}})'>Run this video</button>
  </div>`).join('') : '<div class=muted>no index found</div>';
}
async function run(body){
  if(timer)clearInterval(timer);
  $('#panel').style.display='block'; $('#clips').innerHTML=''; shown=0;
  $('#status').innerHTML='starting…'; window.scrollTo(0,document.body.scrollHeight);
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  JOB=(await r.json()).job; timer=setInterval(poll,1500); poll();
}
$('#runlocal').onclick=()=>run({source:'local',local_path:$('#localpath').value.trim(),camera:$('#localcam').value.trim()||'cam'});
function fmtStage(j){
  const map={queued:'queued…',downloading:'⬇ downloading video',scanning:'🔍 scanning (octopus + motion)',
    scanned:'✓ scanned',windows:'✂ finding clips',done:'✅ done',error:'⚠ error'};
  return (map[j.stage]||j.stage)+(j.detail?` — <span class=muted>${j.detail}</span>`:'');
}
async function poll(){
  const r=await fetch('/api/status?job='+JOB); const j=await r.json();
  let pct = j.total? Math.round(100*j.clips.length/j.total):(['scanned','windows'].includes(j.stage)?8:3);
  if(j.stage=='done')pct=100;
  $('#status').innerHTML=`<div class=row style=justify-content:space-between>
     <div>${fmtStage(j)}</div><div class=muted>${j.clips.length}${j.total?'/'+j.total:''} clips${j.elapsed?' · '+j.elapsed+'s':''}</div></div>
     <div class=bar><i style=width:${pct}%></i></div>${j.error?`<div class=abs style=margin-top:8px>${j.error}</div>`:''}`;
  for(let k=shown;k<j.clips.length;k++){
    const c=j.clips[k];
    const el=document.createElement('div'); el.className='clip';
    const body = c.status=='captioned'
      ? `<div class=tl>▶ ${c.timeline}</div><div class=cap>${c.caption}</div>
         <div class=muted>p_visible ${c.max_p}</div>`
      : `<div class=tl>${c.timeline}</div><div class="cap abs">${c.caption||c.status}</div>`;
    el.innerHTML = (c.status=='captioned'
        ? `<video controls preload=none src="/clip?job=${JOB}&i=${c.i}"></video>`
        : `<div class=muted style="width:300px;text-align:center;padding:40px 0">no octopus</div>`)
      + `<div>${body}</div>`;
    $('#clips').appendChild(el); shown=k+1;
  }
  if(j.done){clearInterval(timer);}
}
loadSuggs();
</script></body></html>"""

if __name__ == "__main__":
    print("loading models (once)…", flush=True); models()
    print("→ http://localhost:8010", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8010, log_level="warning")
