"""
Caption A/B — compare v1 vs v2 captions/labels side by side, per clip.

Left = v1 captions (octopus_clips_verified.json), right = v2 (octopus_clips_verified-2.json),
joined by clip_path. Plays the local clip video next to both. Flags label
disagreements, shows the v2 max_p_visible, and lets you vote which is better
(v1 / v2 / tie / both bad). Votes -> data/caption_ab_votes.json.

Usage:  venv/bin/python3 ui/compare_captions.py   ->  http://localhost:8007
"""
import json, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

PROJECT   = Path(__file__).resolve().parent.parent
V1_JSON   = PROJECT / "data" / "octopus_clips_verified.json"
V2_JSON   = PROJECT / "data" / "octopus_clips_verified-2.json"
CLIPS_DIR = PROJECT / "data" / "octopus_clips_verified"
VOTES     = PROJECT / "data" / "caption_ab_votes.json"

app = FastAPI()


def load_pairs():
    v1 = {c["clip_path"]: c for c in json.load(open(V1_JSON))["clips"]}
    v2 = {c["clip_path"]: c for c in json.load(open(V2_JSON))["clips"]}
    pairs = []
    for cp in v1:                      # base on v1 (clips that exist locally)
        if not (PROJECT / cp).exists():
            continue
        a, b = v1[cp], v2.get(cp, {})
        pairs.append({
            "clip_path": cp, "camera": a.get("camera"), "date": a.get("date"),
            "segment": a.get("segment"), "video_timeline": a.get("video_timeline"),
            "max_p_visible": b.get("max_p_visible"),
            "v1_caption": a.get("caption", ""), "v1_label": a.get("ethogram_label", ""),
            "v2_caption": b.get("caption", "(no v2)"), "v2_label": b.get("ethogram_label", "(no v2)"),
            "differs": a.get("ethogram_label") != b.get("ethogram_label"),
        })
    return pairs


def load_votes():
    return json.load(open(VOTES)) if VOTES.exists() else {}


@app.get("/video")
def video(path: str):
    p = (PROJECT / path).resolve()
    if not str(p).startswith(str(CLIPS_DIR.resolve())) or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="video/mp4")


@app.post("/vote")
async def vote(req: Request):
    body = await req.json()
    votes = load_votes()
    if body.get("vote") == "clear":
        votes.pop(body["clip_path"], None)
    else:
        votes[body["clip_path"]] = body["vote"]
    json.dump(votes, open(VOTES, "w"), indent=2)
    from collections import Counter
    c = Counter(votes.values())
    return JSONResponse({"ok": True, **{k: c.get(k, 0) for k in ["v1", "v2", "tie", "bad"]}, "total": len(votes)})


@app.get("/", response_class=HTMLResponse)
def index():
    pairs = load_pairs()
    votes = load_votes()
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Caption A/B — v1 vs v2</title><style>
 *{{box-sizing:border-box}} body{{font-family:system-ui;margin:0;background:#0d0d0d;color:#eee;
   height:100vh;display:flex;flex-direction:column;overflow:hidden}}
 header{{background:#1c1c1c;padding:8px 16px;border-bottom:1px solid #333;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
 h1{{font-size:15px;margin:0}} .counts span{{margin-right:10px;font-size:13px}}
 label.tog{{font-size:12px;color:#9ab;margin-left:auto}} .keys{{font-size:12px;color:#aaa}} .keys b{{color:#ddd}}
 .stage{{flex:1;display:flex;min-height:0;gap:14px;padding:14px}}
 .vid{{flex:1;display:flex;align-items:center;justify-content:center;min-width:0}}
 video{{max-width:100%;max-height:100%;border:3px solid #333;border-radius:8px;background:#000}}
 .cols{{flex:1.2;display:flex;flex-direction:column;gap:10px;overflow:auto}}
 .meta span{{display:inline-block;background:#262626;border-radius:5px;padding:3px 8px;margin:2px;font-size:12px}}
 .ab{{display:flex;gap:10px}}
 .card{{flex:1;background:#161616;border:1px solid #333;border-radius:8px;padding:10px}}
 .card h3{{margin:0 0 6px;font-size:13px;color:#89a}} .card.v2 h3{{color:#7ec8a0}}
 .lab{{display:inline-block;background:#2c3e50;border-radius:5px;padding:2px 8px;font-size:12px;margin-bottom:6px}}
 .lab.diff{{background:#7d5a1e}} .cap{{font-size:14px;line-height:1.45}}
 .path{{font-size:11px;color:#888;word-break:break-all}}
 .diffbadge{{background:#7d5a1e;color:#fff;font-size:11px;padding:2px 7px;border-radius:5px}}
 footer{{background:#1c1c1c;border-top:1px solid #333;padding:8px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap}}
 footer button{{border:0;padding:9px 15px;border-radius:6px;color:#fff;cursor:pointer;font-size:14px}}
 .b1{{background:#2c3e50}} .b2{{background:#27632a}} .tie{{background:#555}} .bad{{background:#7b241c}} .nav{{background:#222}}
 .voted{{outline:3px solid #fff}} footer button:hover{{filter:brightness(1.3)}}
</style></head><body>
<header><h1>⚖️ Caption A/B — v1 vs v2</h1>
<div class="counts"><span>clip <b id=pos>1</b>/<b id=tot>0</b></span>
<span style="color:#89a">v1 <b id=cv1>0</b></span><span style="color:#7ec8a0">v2 <b id=cv2>0</b></span>
<span>tie <b id=cvt>0</b></span><span style="color:#e88">bad <b id=cvb>0</b></span>
<span id=diffn style="color:#d9a441"></span></div>
<label class="tog"><input type=checkbox id=only onchange="applyFilter()"> only where labels differ</label>
<div class="keys"><b>1</b> v1 · <b>2</b> v2 · <b>3</b> tie · <b>0</b> both bad · <b>←/→</b> nav · <b>U</b> clear</div>
</header>
<div class="stage">
  <div class="vid"><video id="vid" controls autoplay loop muted></video></div>
  <div class="cols">
    <div class="meta" id="meta"></div>
    <div class="ab">
      <div class="card v1"><h3>v1 <span id="d1"></span></h3><span class="lab" id="l1"></span><div class="cap" id="c1"></div></div>
      <div class="card v2"><h3>v2 (enhanced)</h3><span class="lab" id="l2"></span><div class="cap" id="c2"></div></div>
    </div>
    <div class="path" id="path"></div>
  </div>
</div>
<footer>
 <button class="nav" onclick="go(-1)">←</button>
 <button class="b1" id="vb1" onclick="vote('v1')">v1 better (1)</button>
 <button class="tie" id="vbt" onclick="vote('tie')">tie (3)</button>
 <button class="b2" id="vb2" onclick="vote('v2')">v2 better (2)</button>
 <button class="bad" id="vbb" onclick="vote('bad')">both bad (0)</button>
 <button class="nav" onclick="go(1)">→</button>
</footer>
<script>
const ALL={json.dumps(pairs)};
let votes={json.dumps(votes)};
let view=ALL, i=0;
const $=id=>document.getElementById(id);
function counts(){{
 const v=Object.values(votes);
 $('cv1').textContent=v.filter(x=>x=='v1').length; $('cv2').textContent=v.filter(x=>x=='v2').length;
 $('cvt').textContent=v.filter(x=>x=='tie').length; $('cvb').textContent=v.filter(x=>x=='bad').length;
 $('diffn').textContent=ALL.filter(p=>p.differs).length+' differ';
 $('tot').textContent=view.length;
}}
function fmt(v){{return v===undefined||v===null?'–':v;}}
function render(){{
 if(!view.length){{return;}}
 const p=view[i];
 $('vid').src='/video?path='+encodeURIComponent(p.clip_path);
 $('meta').innerHTML=`<span>${{p.camera}}</span><span>${{p.date}} ${{p.segment}}</span>`+
   `<span>⏱ ${{fmt(p.video_timeline)}}</span><span>v2 max_p ${{fmt(p.max_p_visible)}}</span>`+
   (p.differs?`<span class="diffbadge">labels differ</span>`:``);
 $('l1').className='lab'+(p.differs?' diff':''); $('l1').textContent=p.v1_label;
 $('l2').className='lab'+(p.differs?' diff':''); $('l2').textContent=p.v2_label;
 $('c1').textContent=p.v1_caption; $('c2').textContent=p.v2_caption;
 $('path').textContent=p.clip_path; $('pos').textContent=i+1;
 const v=votes[p.clip_path];
 for(const [id,val] of [['vb1','v1'],['vb2','v2'],['vbt','tie'],['vbb','bad']])
   $(id).classList.toggle('voted', v==val);
 counts();
}}
function go(d){{ i=Math.max(0,Math.min(view.length-1,i+d)); render(); }}
async function vote(v){{
 const p=view[i]; const cur=votes[p.clip_path];
 if(cur==v){{ delete votes[p.clip_path]; await fetch('/vote',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{clip_path:p.clip_path,vote:'clear'}})}}); render(); return; }}
 votes[p.clip_path]=v;
 await fetch('/vote',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{clip_path:p.clip_path,vote:v}})}});
 render(); setTimeout(()=>go(1),150);
}}
function clearVote(){{ const p=view[i]; delete votes[p.clip_path]; fetch('/vote',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{clip_path:p.clip_path,vote:'clear'}})}}); render(); }}
function applyFilter(){{ view=$('only').checked?ALL.filter(p=>p.differs):ALL; i=0; render(); }}
document.addEventListener('keydown',e=>{{
 if(e.key=='1')vote('v1'); else if(e.key=='2')vote('v2'); else if(e.key=='3')vote('tie');
 else if(e.key=='0')vote('bad');
 else if(e.key=='ArrowRight'||e.key==' '){{e.preventDefault();go(1);}}
 else if(e.key=='ArrowLeft')go(-1); else if(e.key=='u'||e.key=='U')clearVote();
}});
render();
</script></body></html>"""


if __name__ == "__main__":
    print("Caption A/B (v1 vs v2) → http://localhost:8007")
    uvicorn.run(app, host="0.0.0.0", port=8007, log_level="warning")
