"""
Caption comparison UI — for each clip show the video (or the frames the model saw)
alongside three captions:
  • Teacher      — Qwen3-VL-235B      (ref)
  • Student base — Qwen3-VL-2B        (base)
  • Student+LoRA — Qwen3-VL-2B + LoRA (lora)

Reads data/caption_eval/caption_comparison_v1.json (3,458 rows). Videos resolve from
src/ or data/octopus_clips_verified/; the 6 CLAHE frames (the actual model input) come
from src/dataset/v1/frames/. Val-split rows (held-out) are flagged so you can filter to
the honest eval set.

Run:  venv/bin/python3 ui/compare_captions_lora.py   ->  http://localhost:8010
"""
import json, hashlib
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

PROJECT = Path(__file__).resolve().parent.parent
COMP    = PROJECT / "data" / "caption_eval" / "caption_comparison_v1.json"
FRAMES  = PROJECT / "src" / "dataset" / "v1" / "frames"
VAL_FRAC = 0.10

app = FastAPI()
rows = json.load(open(COMP))


def clip_rel(cp): return cp.split("octopus_clips_verified/", 1)[-1]

def video_path(cp):
    rel = clip_rel(cp)
    for base in ("src/octopus_clips_verified", "data/octopus_clips_verified"):
        p = PROJECT / base / rel
        if p.exists():
            return p
    return None

def is_val(cp):
    rel = clip_rel(cp); parts = rel.split("/")
    if len(parts) < 3: return False
    date, seg, fn = parts[0], parts[1], parts[2]
    cam = fn.rsplit("_", 1)[0]
    h = int(hashlib.md5(f"{date}/{seg}/{cam}".encode()).hexdigest(), 16)
    return (h % 1000) < int(VAL_FRAC * 1000)

# precompute lightweight metadata
META = []
for i, r in enumerate(rows):
    rel = clip_rel(r["clip"])
    META.append({"i": i, "clip": rel, "val": is_val(r["clip"]),
                 "hasvid": video_path(r["clip"]) is not None,
                 "nframes": len(r.get("frames", [])),
                 "ref": r.get("ref", ""), "base": r.get("base", ""), "lora": r.get("lora", "")})
N_VAL = sum(1 for m in META if m["val"])


@app.get("/vid/{i}")
def vid(i: int):
    p = video_path(rows[i]["clip"])
    if not p: return JSONResponse({"error": "no video"}, status_code=404)
    return FileResponse(p, media_type="video/mp4")

@app.get("/frame/{i}/{j}")
def frame(i: int, j: int):
    fr = rows[i].get("frames", [])
    if j >= len(fr): return JSONResponse({"error": "no frame"}, status_code=404)
    p = FRAMES / Path(fr[j]).name
    if not p.exists(): return JSONResponse({"error": "missing"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg")

@app.get("/meta")
def meta(): return JSONResponse(META)


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html><head><meta charset=utf-8><title>Caption comparison — teacher vs base vs LoRA</title><style>
 :root{--bg:#0d1413;--panel:#151f1e;--line:#26332f;--ink:#d3ddd9;--dim:#7f918c;--teal:#3fb6a8;--gold:#d9b25e;--slate:#7f95c9}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
 header{background:var(--panel);border-bottom:1px solid var(--line);padding:10px 18px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600} h1 b{color:var(--teal)}
 .meta{font:12px ui-monospace,monospace;color:var(--dim)} .val{color:var(--gold);font-weight:600}
 label{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:5px;cursor:pointer}
 .keys{margin-left:auto;font-size:12px;color:var(--dim)} .keys b{color:var(--ink)}
 main{flex:1;display:grid;grid-template-columns:minmax(360px,44%) 1fr;gap:16px;padding:16px 18px;min-height:0}
 .media{display:flex;flex-direction:column;gap:10px;min-height:0}
 video{width:100%;background:#000;border:1px solid var(--line);border-radius:8px;max-height:52vh}
 .strip{display:flex;gap:5px;overflow-x:auto} .strip img{height:74px;border:1px solid var(--line);border-radius:4px}
 .caps{display:flex;flex-direction:column;gap:12px;overflow-y:auto;min-height:0}
 .cap{background:var(--panel);border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:12px 14px}
 .cap .tag{font:600 11px ui-monospace,monospace;letter-spacing:.04em;text-transform:uppercase;margin-bottom:6px}
 .cap .txt{font-size:14.5px;line-height:1.5}
 .ref{border-left-color:var(--gold)} .ref .tag{color:var(--gold)}
 .base{border-left-color:var(--slate)} .base .tag{color:var(--slate)}
 .lora{border-left-color:var(--teal)} .lora .tag{color:var(--teal)}
 footer{background:var(--panel);border-top:1px solid var(--line);padding:8px;display:flex;gap:10px;justify-content:center;align-items:center}
 footer button{border:0;padding:8px 15px;border-radius:6px;background:#243230;color:#fff;cursor:pointer;font-size:14px}
 footer button:hover{filter:brightness(1.3)} #jump{width:70px;background:#0d1413;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px;font:12px ui-monospace,monospace}
</style></head><body>
<header>
 <h1>Caption comparison &mdash; <b>Nity</b> clips</h1>
 <span class="meta" id="pos"></span>
 <label><input type="checkbox" id="vidonly" checked> video only</label>
 <label><input type="checkbox" id="valonly"> val-split only (<span id=nval></span>)</label>
 <span class="keys"><b>&larr;/&rarr;</b> nav &middot; <b>Space</b> replay</span>
</header>
<main>
 <div class="media">
   <video id="vid" autoplay loop muted playsinline></video>
   <div id="novid" style="display:none;padding:14px;border:1px dashed var(--line);border-radius:8px;color:var(--dim);font-size:13px">No local video for this clip &mdash; showing the model-input frames below.</div>
   <div class="strip" id="strip"></div>
 </div>
 <div class="caps">
   <div class="cap ref"><div class="tag">Teacher &mdash; Qwen3-VL-235B</div><div class="txt" id="ref"></div></div>
   <div class="cap base"><div class="tag">Student base &mdash; Qwen3-VL-2B</div><div class="txt" id="base"></div></div>
   <div class="cap lora"><div class="tag">Student + LoRA &mdash; Qwen3-VL-2B</div><div class="txt" id="lora"></div></div>
 </div>
</main>
<footer>
 <button onclick="go(-1)">&larr; Prev</button>
 <input id="jump" type="number" min="1" onchange="jump(this.value)">
 <button onclick="go(1)">Next &rarr;</button>
</footer>
<script>
let META=[], view=[], k=0;
const $=id=>document.getElementById(id);
async function load(){
 META=await (await fetch('/meta')).json();
 $('nval').textContent=META.filter(m=>m.val).length;
 applyFilter();
}
function applyFilter(){
 const vo=$('valonly').checked, vd=$('vidonly').checked;
 view=META.filter(m=>(!vo||m.val)&&(!vd||m.hasvid)); k=0; render();
}
function render(){
 if(!view.length)return;
 const m=view[k];
 $('pos').innerHTML=`clip ${k+1}/${view.length} &middot; ${m.clip} ${m.val?'<span class=val>[VAL]</span>':''}`;
 $('jump').value=k+1;
 $('ref').textContent=m.ref||'—'; $('base').textContent=m.base||'—'; $('lora').textContent=m.lora||'—';
 const v=$('vid');
 if(m.hasvid){ v.style.display=''; $('novid').style.display='none'; v.src='/vid/'+m.i; v.load(); } else { v.style.display='none'; $('novid').style.display=''; }
 const s=$('strip'); s.innerHTML='';
 for(let j=0;j<m.nframes;j++){ const im=new Image(); im.src='/frame/'+m.i+'/'+j; s.appendChild(im); }
}
function go(d){ k=Math.max(0,Math.min(view.length-1,k+d)); render(); }
function jump(v){ const n=parseInt(v)-1; if(n>=0&&n<view.length){k=n;render();} }
$('valonly').addEventListener('change',applyFilter);
$('vidonly').addEventListener('change',applyFilter);
document.addEventListener('keydown',e=>{
 if(e.key=='ArrowRight'){go(1);} else if(e.key=='ArrowLeft'){go(-1);}
 else if(e.key==' '){e.preventDefault();const v=$('vid'); if(v.style.display!=='none'){v.currentTime=0;v.play();}}
});
load();
</script></body></html>"""


if __name__ == "__main__":
    print(f"Caption comparison ({len(rows)} clips, {N_VAL} val) -> http://localhost:8010")
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="warning")
