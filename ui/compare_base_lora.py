"""
Base vs LoRA caption comparison UI (for demo_clips_captions.json).

Each clip is streamed on demand from the server (ffmpeg byte-range over the entry's
video_url + start/end), shown next to its `caption_base` (plain Qwen2.5-VL-3B) and
`caption_lora` (base + fine-tuned adapter). Optional vote which is better ->
data/demo_caption_votes.json.

Creds come from server_creds.py (repo-root .env) — no password in this file.

Usage:  venv/bin/python3 ui/compare_base_lora.py   ->  http://localhost:8009
"""
import sys, json, subprocess, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from server_creds import USER, PASS

JSON  = PROJECT / "data" / "demo_clips_captions.json"
VOTES = PROJECT / "data" / "demo_caption_votes.json"

app = FastAPI()


def auth(url: str) -> str:
    return url.replace("https://", f"https://{USER}:{PASS}@") if USER else url


def load_clips():
    d = json.load(open(JSON))
    return d["clips"] if isinstance(d, dict) else d


def load_votes():
    return json.load(open(VOTES)) if VOTES.exists() else {}


@app.get("/clip")
def clip(i: int):
    clips = load_clips()
    if i < 0 or i >= len(clips):
        return JSONResponse({"error": "bad index"}, status_code=404)
    c = clips[i]
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(c["start_sec"]), "-to", str(c["end_sec"]),
           "-i", auth(c["video_url"]), "-c", "copy",
           "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def gen():
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try: proc.stdout.close()
            except Exception: pass
            proc.kill()
    return StreamingResponse(gen(), media_type="video/mp4")


@app.post("/vote")
async def vote(req: Request):
    body = await req.json()
    votes = load_votes()
    key = str(body["i"])
    if body.get("vote") == "clear":
        votes.pop(key, None)
    else:
        votes[key] = body["vote"]
    json.dump(votes, open(VOTES, "w"), indent=2)
    from collections import Counter
    c = Counter(votes.values())
    return JSONResponse({"ok": True, **{k: c.get(k, 0) for k in ["base", "lora", "tie", "bad"]}, "total": len(votes)})


@app.get("/", response_class=HTMLResponse)
def index():
    clips = load_clips()
    votes = load_votes()
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Base vs LoRA captions</title><style>
 *{{box-sizing:border-box}} body{{font-family:system-ui;margin:0;background:#0d0d0d;color:#eee;
   height:100vh;display:flex;flex-direction:column;overflow:hidden}}
 header{{background:#1c1c1c;padding:8px 16px;border-bottom:1px solid #333;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
 h1{{font-size:15px;margin:0}} .counts span{{margin-right:10px;font-size:13px}}
 .keys{{font-size:12px;color:#aaa;margin-left:auto}} .keys b{{color:#ddd}}
 .stage{{flex:1;display:flex;min-height:0;gap:14px;padding:14px}}
 .vid{{flex:1;display:flex;align-items:center;justify-content:center;min-width:0}}
 video{{max-width:100%;max-height:100%;border:3px solid #333;border-radius:8px;background:#000}}
 .cols{{flex:1.2;display:flex;flex-direction:column;gap:10px;overflow:auto}}
 .meta span{{display:inline-block;background:#262626;border-radius:5px;padding:3px 8px;margin:2px;font-size:12px}}
 .ab{{display:flex;gap:10px}}
 .card{{flex:1;background:#161616;border:1px solid #333;border-radius:8px;padding:12px}}
 .card h3{{margin:0 0 8px;font-size:12px;letter-spacing:.5px}} .card.base h3{{color:#89a}} .card.lora h3{{color:#7ec8a0}}
 .cap{{font-size:15px;line-height:1.5}}
 footer{{background:#1c1c1c;border-top:1px solid #333;padding:8px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap}}
 footer button{{border:0;padding:9px 16px;border-radius:6px;color:#fff;cursor:pointer;font-size:14px}}
 .b{{background:#2c3e50}} .l{{background:#27632a}} .tie{{background:#555}} .bad{{background:#7b241c}} .nav{{background:#222}}
 .voted{{outline:3px solid #fff}} footer button:hover{{filter:brightness(1.3)}}
</style></head><body>
<header><h1>⚖️ Base vs LoRA captions</h1>
<div class="counts"><span>clip <b id=pos>1</b>/<b id=tot>0</b></span>
<span style="color:#89a">base <b id=cb>0</b></span><span style="color:#7ec8a0">lora <b id=cl>0</b></span>
<span>tie <b id=ct>0</b></span><span style="color:#e88">bad <b id=cx>0</b></span></div>
<div class="keys"><b>1</b> base · <b>2</b> lora · <b>3</b> tie · <b>0</b> both bad · <b>←/→</b> nav · <b>U</b> clear</div>
</header>
<div class="stage">
  <div class="vid"><video id="vid" controls autoplay loop muted></video></div>
  <div class="cols">
    <div class="meta" id="meta"></div>
    <div class="ab">
      <div class="card base"><h3>BASE — Qwen2.5-VL-3B</h3><div class="cap" id="cbase"></div></div>
      <div class="card lora"><h3>LORA — fine-tuned student</h3><div class="cap" id="clora"></div></div>
    </div>
  </div>
</div>
<footer>
 <button class="nav" onclick="go(-1)">←</button>
 <button class="b" id="vb" onclick="vote('base')">base better (1)</button>
 <button class="tie" id="vt" onclick="vote('tie')">tie (3)</button>
 <button class="l" id="vl" onclick="vote('lora')">lora better (2)</button>
 <button class="bad" id="vx" onclick="vote('bad')">both bad (0)</button>
 <button class="nav" onclick="go(1)">→</button>
</footer>
<script>
const clips={json.dumps(clips)};
let votes={json.dumps(votes)};
let i=0;
const $=id=>document.getElementById(id);
function counts(){{
 const v=Object.values(votes);
 $('cb').textContent=v.filter(x=>x=='base').length; $('cl').textContent=v.filter(x=>x=='lora').length;
 $('ct').textContent=v.filter(x=>x=='tie').length; $('cx').textContent=v.filter(x=>x=='bad').length;
 $('tot').textContent=clips.length;
}}
function fmt(v){{return v===undefined||v===null?'–':v;}}
function render(){{
 const c=clips[i];
 $('vid').src='/clip?i='+i;
 $('meta').innerHTML=`<span>⏱ ${{fmt(c.video_timeline)}}</span><span>vis ${{fmt(c.visible_frac)}}</span><span>motion ${{fmt(c.mean_motion)}}</span>`;
 $('cbase').textContent=c.caption_base||'(none)';
 $('clora').textContent=c.caption_lora||'(none)';
 $('pos').textContent=i+1;
 const v=votes[String(i)];
 for(const [id,val] of [['vb','base'],['vl','lora'],['vt','tie'],['vx','bad']]) $(id).classList.toggle('voted', v==val);
 counts();
}}
function go(d){{ i=Math.max(0,Math.min(clips.length-1,i+d)); render(); }}
async function vote(v){{
 const cur=votes[String(i)];
 if(cur==v){{ delete votes[String(i)]; await fetch('/vote',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{i:i,vote:'clear'}})}}); render(); return; }}
 votes[String(i)]=v;
 await fetch('/vote',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{i:i,vote:v}})}});
 render(); setTimeout(()=>go(1),150);
}}
function clearVote(){{ delete votes[String(i)]; fetch('/vote',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{i:i,vote:'clear'}})}}); render(); }}
document.addEventListener('keydown',e=>{{
 if(e.key=='1')vote('base'); else if(e.key=='2')vote('lora'); else if(e.key=='3')vote('tie');
 else if(e.key=='0')vote('bad'); else if(e.key=='ArrowRight'||e.key==' '){{e.preventDefault();go(1);}}
 else if(e.key=='ArrowLeft')go(-1); else if(e.key=='u'||e.key=='U')clearVote();
}});
render();
</script></body></html>"""


if __name__ == "__main__":
    print("Base vs LoRA captions → http://localhost:8009")
    uvicorn.run(app, host="0.0.0.0", port=8009, log_level="warning")
