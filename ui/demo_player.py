"""
Demo player UI — full video on the left, synced octopus captions on the right (FastAPI, port 8011).

Reads pre-processed local demo videos from `data/demo_videos/` (each `*.mp4` + its
`*_captions.json` produced by `src/local_pipeline.py` with save_clips=False). Pick a video; it plays
on the left while its captions list on the right, each tagged with the `mm:ss-mm:ss` timeline. Click
a caption to jump the video there; the active caption highlights and auto-scrolls as it plays.

Also has an UPLOAD button: drop in any local video, it runs the full pipeline (CLIP+motion extraction
→ MLX 4-bit caption student), writes a `*_captions.json`, and the video appears as a new tab.

Run:  venv/bin/python3 ui/demo_player.py   ->  http://localhost:8011
"""
import json, re, sys, uuid, threading, shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"; sys.path.insert(0, str(SRC))
DEMO_DIR = REPO / "data" / "demo_videos"; DEMO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
JOBS = {}                         # job_id -> {stage, detail, total, done, error, demo_id}
JOB_LOCK = threading.Lock()       # one pipeline job at a time (single shared GPU/model)
_MODELS = None
_MODELS_LOCK = threading.Lock()


def models():
    global _MODELS
    with _MODELS_LOCK:
        if _MODELS is None:
            import local_pipeline as lp
            _MODELS = lp.load_models()
    return _MODELS


def demo_list():
    out = []
    for mp4 in sorted(DEMO_DIR.glob("*.mp4")):
        cap = mp4.with_name(mp4.stem + "_captions.json")
        info = {"id": mp4.stem, "name": mp4.stem, "processed": cap.exists(), "n_present": 0}
        m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d+)_(Right_\w+)", mp4.stem)
        info["name"] = f"{m.group(3)} · {m.group(1)}" if m else mp4.stem
        if cap.exists():
            d = json.load(open(cap))
            info["n_present"] = sum(1 for c in d.get("clips", []) if c.get("status") == "captioned")
        out.append(info)
    return out


def _safe_stem(name):
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem)[:60] or "upload"
    return f"upload_{stem}"


def run_upload_job(job_id, video_path):
    import local_pipeline as lp
    job = JOBS[job_id]
    try:
        with JOB_LOCK:
            job["stage"] = "loading"; job["detail"] = "loading models…"
            M = models()
            def on_stage(s, d): job["stage"] = s; job["detail"] = d
            def on_clip(i, n, rec): job["total"] = n; job["progress"] = i
            job["stage"] = "scanning"
            camera = "upload"
            lp.process_video(video_path, DEMO_DIR, M, camera=camera,
                             on_stage=on_stage, on_clip=on_clip, save_clips=False)
        job["stage"] = "done"; job["done"] = True; job["demo_id"] = Path(video_path).stem
    except Exception as e:
        job["error"] = str(e); job["stage"] = "error"; job["done"] = True


@app.get("/api/demos")
def api_demos():
    return JSONResponse(demo_list())


@app.get("/api/captions/{demo_id}")
def api_captions(demo_id: str):
    cap = DEMO_DIR / f"{demo_id}_captions.json"
    if not cap.exists():
        return JSONResponse({"error": "not processed"}, status_code=404)
    d = json.load(open(cap))
    clips = [{"start": c["start"], "end": c["end"], "timeline": c["video_timeline"],
              "caption": c.get("caption"), "status": c.get("status"), "max_p": c.get("max_p_visible")}
             for c in d.get("clips", [])]
    return JSONResponse({"n_clips": len(clips), "clips": clips})


@app.get("/video/{demo_id}")
def video(demo_id: str):
    mp4 = DEMO_DIR / f"{demo_id}.mp4"          # FileResponse supports Range -> seekable
    if not mp4.exists():
        return JSONResponse({"error": "no video"}, status_code=404)
    return FileResponse(str(mp4), media_type="video/mp4")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    stem = _safe_stem(file.filename)
    # avoid clobbering an existing name
    dest = DEMO_DIR / f"{stem}.mp4"
    k = 1
    while dest.exists():
        dest = DEMO_DIR / f"{stem}_{k}.mp4"; k += 1
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if dest.stat().st_size < 10000:
        dest.unlink(missing_ok=True)
        return JSONResponse({"error": "upload too small / not a video"}, status_code=400)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"stage": "queued", "detail": file.filename, "total": None, "progress": 0,
                    "done": False, "error": None, "demo_id": dest.stem}
    threading.Thread(target=run_upload_job, args=(job_id, str(dest)), daemon=True).start()
    return JSONResponse({"job": job_id, "demo_id": dest.stem})


@app.get("/api/upload_status")
def api_upload_status(job: str):
    j = JOBS.get(job)
    if not j:
        return JSONResponse({"error": "no such job"}, status_code=404)
    return JSONResponse({**j, "busy": JOB_LOCK.locked()})


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Octopus demo player</title>
<style>
 :root{--bg:#0f1216;--card:#1a1f27;--fg:#e6e9ee;--mut:#8b95a5;--acc:#4aa8ff;--ok:#39d98a;--warn:#f0a24b;--hl:#1e3a52}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;height:100vh;display:flex;flex-direction:column}
 header{padding:14px 22px;border-bottom:1px solid #262c36;flex:0 0 auto}
 h1{margin:0;font-size:18px} .sub{color:var(--mut);font-size:12px;margin-top:2px}
 .toprow{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
 #tabs{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
 .tab{background:var(--card);border:1px solid #262c36;border-radius:8px;padding:7px 12px;cursor:pointer;font-size:13px}
 .tab.on{border-color:var(--acc);color:var(--acc)} .tab small{color:var(--mut);display:block;font-size:11px}
 .tab.pending{opacity:.5}
 #uploadbox{flex:0 0 auto;text-align:right}
 button{background:var(--acc);color:#04121f;border:0;border-radius:8px;padding:9px 15px;font-weight:600;cursor:pointer;font-size:13px}
 button:disabled{opacity:.5;cursor:default}
 #uprog{font-size:12px;color:var(--mut);margin-top:6px;max-width:260px}
 .bar{height:6px;background:#0c0f13;border-radius:6px;overflow:hidden;margin-top:5px}
 .bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .3s}
 main{flex:1 1 auto;display:grid;grid-template-columns:1.15fr .85fr;gap:0;min-height:0}
 #left{padding:18px 22px;border-right:1px solid #262c36;display:flex;flex-direction:column;min-height:0}
 #left video{width:100%;background:#000;border-radius:10px}
 #now{margin-top:14px;background:var(--card);border:1px solid #262c36;border-radius:10px;padding:14px 16px}
 #now .tl{color:var(--acc);font-weight:600} #now .cap{margin-top:6px;font-size:16px}
 #right{overflow-y:auto;padding:14px 18px}
 #right h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:2px 0 12px}
 .row{display:flex;gap:11px;padding:10px 11px;border-radius:9px;cursor:pointer;border:1px solid transparent}
 .row:hover{background:#161b22} .row.on{background:var(--hl);border-color:var(--acc)}
 .row .tl{color:var(--acc);font-weight:600;font-size:12px;white-space:nowrap;padding-top:1px}
 .row.absent{opacity:.5} .row.absent .tl{color:var(--mut)}
 .row .c{font-size:13.5px} .empty{color:var(--mut);padding:30px}
 #drop{display:none;position:fixed;inset:0;z-index:99;background:rgba(15,18,22,.85);
   align-items:center;justify-content:center;border:3px dashed var(--acc);
   font-size:22px;color:var(--acc);pointer-events:none}
</style></head><body>
<div id=drop>⬇ Drop a video to caption it</div>
<header>
 <div class=toprow>
  <div>
   <h1>🐙 Octopus caption demo — video + timeline</h1>
   <div class=sub>Local pipeline: CLIP+motion extraction · MLX 4-bit caption student. Click a caption to jump the video.</div>
  </div>
  <div id=uploadbox>
   <input id=file type=file accept="video/*" style=display:none>
   <button id=upbtn onclick="document.getElementById('file').click()">⬆ Upload a video</button>
   <div id=uprog></div>
  </div>
 </div>
 <div id=tabs></div>
</header>
<main>
 <div id=left>
   <video id=vid controls preload=metadata></video>
   <div id=now><div class=tl id=nowtl>—</div><div class=cap id=nowcap>Select a caption or press play.</div></div>
 </div>
 <div id=right><h2 id=rhead>Captions</h2><div id=list class=empty>Pick a video above, or upload one.</div></div>
</main>
<script>
const $=s=>document.querySelector(s), vid=$('#vid');
let CLIPS=[], curId=null, active=-1;
async function loadTabs(selId){
  const demos=await (await fetch('/api/demos')).json();
  $('#tabs').innerHTML = demos.map(d=>`<div class="tab ${d.processed?'':'pending'}" data-id="${d.id}"
     ${d.processed?`onclick="pick('${d.id}',this)"`:''}>${d.name}<small>${d.processed? d.n_present+' captions':'processing…'}</small></div>`).join('');
  const want = selId || curId || (demos.find(d=>d.processed)||{}).id;
  if(want){ const el=[...document.querySelectorAll('.tab')].find(t=>t.dataset.id===want); if(el && el.onclick) pick(want, el); }
}
async function pick(id, tabEl){
  curId=id; active=-1;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on')); if(tabEl)tabEl.classList.add('on');
  vid.src='/video/'+id; vid.load();
  const d=await (await fetch('/api/captions/'+id)).json(); CLIPS=d.clips||[];
  $('#rhead').textContent=`Captions — ${CLIPS.filter(c=>c.status=='captioned').length} of ${CLIPS.length} windows`;
  $('#list').className=''; render();
}
function render(){
  $('#list').innerHTML = CLIPS.map((c,i)=>`<div class="row ${c.status=='captioned'?'':'absent'}" id="r${i}" onclick="seek(${i})">
     <div class=tl>${c.timeline}</div><div class=c>${c.caption||c.status}</div></div>`).join('') || '<div class=empty>No clips passed the gates.</div>';
}
function seek(i){ vid.currentTime=CLIPS[i].start; vid.play(); setActive(i); }
function setActive(i){
  if(i===active) return;
  const p=$('#r'+active), n=$('#r'+i);
  if(p)p.classList.remove('on'); if(n){n.classList.add('on'); n.scrollIntoView({block:'nearest',behavior:'smooth'});}
  active=i;
  if(i>=0){ $('#nowtl').textContent=CLIPS[i].timeline; $('#nowcap').textContent=CLIPS[i].caption||CLIPS[i].status; }
}
vid.addEventListener('timeupdate',()=>{
  const t=vid.currentTime, i=CLIPS.findIndex(c=>t>=c.start && t<c.end);
  if(i>=0 && i!==active) setActive(i);
});
// ---- upload ----
const STAGE={queued:'queued…',loading:'loading models…',scanning:'🔍 scanning video…',scanned:'✓ scanned',
             windows:'✂ finding clips…',done:'✅ done',error:'⚠ error'};
async function uploadFile(f){
  if(!f) return;
  if(!f.type.startsWith('video/') && !/[.](mp4|mov|mkv|avi|webm|m4v)$/i.test(f.name)){
    $('#uprog').textContent='not a video file'; return; }
  $('#upbtn').disabled=true;
  $('#uprog').innerHTML=`uploading <b>${f.name}</b>…<div class=bar><i style=width:5%></i></div>`;
  const fd=new FormData(); fd.append('file', f);
  let r; try{ r=await (await fetch('/api/upload',{method:'POST',body:fd})).json(); }
  catch(err){ $('#uprog').textContent='upload failed'; $('#upbtn').disabled=false; return; }
  if(r.error){ $('#uprog').textContent=r.error; $('#upbtn').disabled=false; return; }
  pollUpload(r.job);
}
$('#file').onchange=e=>uploadFile(e.target.files[0]);
// drag & drop anywhere on the page
const drop=$('#drop');
['dragenter','dragover'].forEach(ev=>document.addEventListener(ev,e=>{e.preventDefault(); drop.style.display='flex';}));
['dragleave','drop'].forEach(ev=>document.addEventListener(ev,e=>{
  e.preventDefault(); if(ev==='drop'||e.relatedTarget===null) drop.style.display='none';}));
document.addEventListener('drop',e=>{ const f=e.dataTransfer.files[0]; if(f) uploadFile(f); });
async function pollUpload(job){
  const j=await (await fetch('/api/upload_status?job='+job)).json();
  const pct = j.stage=='done'?100 : j.total? Math.round(10+85*(j.progress/j.total)) : (j.stage=='scanning'?8:4);
  $('#uprog').innerHTML=`${STAGE[j.stage]||j.stage} ${j.total?`— ${j.progress}/${j.total} clips`:''}
     <div class=bar><i style=width:${pct}%></i></div>${j.error?`<div style=color:var(--warn)>${j.error}</div>`:''}`;
  if(j.done){
    $('#upbtn').disabled=false;
    if(!j.error){ $('#uprog').innerHTML=`✅ done — <b>${j.demo_id}</b>`; await loadTabs(j.demo_id); }
    return;
  }
  setTimeout(()=>pollUpload(job), 1500);
}
loadTabs();
</script></body></html>"""

if __name__ == "__main__":
    print("→ http://localhost:8011", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8011, log_level="warning")
