"""
Hard-negative frame review UI — go through the extracted Right_Top frames (VLM said
"octopus not present") and confirm each: is the octopus REALLY absent, or did the VLM
miss it?

  O / 1  -> octopus IS present (VLM was wrong)      [border green]
  N / 0  -> no octopus (genuine hard negative)      [border red]
  Space / -> skip forward   |   <- back   |   U clear   |   F full-res

Reads every *.jpg in HN_DIR (default data/hardneg_right_top). Decisions saved to
HN_DIR/review_decisions.csv (resumable; starts at first unreviewed).

Usage:  venv/bin/python3 ui/review_hardneg_frames.py   ->  http://localhost:8006
        HN_DIR=data/other_frames venv/bin/python3 ui/review_hardneg_frames.py
"""
import os, csv, json, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

PROJECT   = Path(__file__).resolve().parent.parent
HN_DIR    = (PROJECT / os.environ.get("HN_DIR", "data/hardneg_right_top")).resolve()
DECISIONS = HN_DIR / "review_decisions.csv"
PORT      = int(os.environ.get("PORT", "8006"))

app = FastAPI()


def load_items():
    return [{"frame": p.name} for p in sorted(HN_DIR.glob("*.jpg"))]


def load_decisions():
    d = {}
    if DECISIONS.exists():
        for row in csv.DictReader(open(DECISIONS)):
            d[row["frame"]] = row["label"]
    return d


def save_decisions(d):
    with open(DECISIONS, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "label", "ts"])
        for k, v in d.items():
            w.writerow([k, v, datetime.datetime.now().isoformat(timespec="seconds")])


@app.get("/img/{name}")
def img(name: str):
    p = HN_DIR / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.post("/mark")
async def mark(req: Request):
    body = await req.json()
    d = load_decisions()
    if body.get("label") == "clear":
        d.pop(body["frame"], None)
    else:
        d[body["frame"]] = body["label"]
    save_decisions(d)
    vals = list(d.values())
    return JSONResponse({"ok": True, "octopus": vals.count("octopus"),
                         "hardneg": vals.count("hardneg")})


@app.get("/", response_class=HTMLResponse)
def index():
    items = load_items()
    dec = load_decisions()
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Right_Top hard-neg review</title><style>
 *{{box-sizing:border-box}} body{{font-family:system-ui;margin:0;background:#0d1413;color:#cdd8d4;
   height:100vh;display:flex;flex-direction:column;overflow:hidden}}
 header{{background:#141d1c;padding:8px 16px;border-bottom:1px solid #243230;display:flex;
   align-items:center;gap:18px;flex-wrap:wrap}}
 h1{{font-size:15px;margin:0}} .counts span{{margin-right:12px;font-size:13px}}
 .keys{{font-size:12px;color:#7f918c;margin-left:auto}} .keys b{{color:#cdd8d4}}
 .stage{{flex:1;position:relative;display:flex;align-items:center;justify-content:center;min-height:0;padding:8px}}
 #img{{max-width:100%;max-height:100%;object-fit:contain;border:4px solid #243230;border-radius:6px}}
 #img.octopus{{border-color:#2ecc71}} #img.hardneg{{border-color:#e74c3c}}
 .hud{{position:absolute;top:14px;left:18px;background:#000a;padding:6px 10px;border-radius:6px;font-size:12px;color:#bbb}}
 .verdict{{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);font-size:22px;font-weight:700}}
 .v-octopus{{color:#2ecc71}} .v-hardneg{{color:#e74c3c}}
 footer{{background:#141d1c;border-top:1px solid #243230;padding:8px;display:flex;gap:10px;justify-content:center}}
 footer button{{border:0;padding:9px 16px;border-radius:6px;color:#fff;cursor:pointer;font-size:14px}}
 .oct{{background:#27632a}} .neg{{background:#7b241c}} .skip{{background:#34495e}} .nav{{background:#222}}
 footer button:hover{{filter:brightness(1.3)}}
</style></head><body>
<header><h1>Right_Top hard-neg review</h1>
<div class="counts"><span>pos <b id=pos>1</b>/<b id=tot>0</b></span>
<span style="color:#2ecc71">🐙 present <b id=oct>0</b></span>
<span style="color:#e74c3c">∅ empty <b id=neg>0</b></span>
<span>left <b id=un>0</b></span></div>
<div class="keys"><b>O</b>/1 octopus present · <b>N</b>/0 no octopus · <b>Space</b>/→ skip · <b>←</b> back · <b>U</b> clear · <b>F</b> full-res</div>
</header>
<div class="stage">
  <img id="img">
  <div class="hud"><span id="fn"></span></div>
  <div id="verdict" class="verdict"></div>
</div>
<footer>
 <button class="nav" onclick="go(-1)">← Back</button>
 <button class="neg" onclick="mark('hardneg')">∅ No octopus (N)</button>
 <button class="oct" onclick="mark('octopus')">🐙 Octopus present (O)</button>
 <button class="skip" onclick="go(1)">Skip →</button>
</footer>
<script>
const items={json.dumps(items)};
let dec={json.dumps(dec)};
let i=0;
const $=id=>document.getElementById(id);
i=items.findIndex(it=>!dec[it.frame]); if(i<0)i=0;
function counts(){{
 const v=Object.values(dec);
 $('oct').textContent=v.filter(x=>x=='octopus').length;
 $('neg').textContent=v.filter(x=>x=='hardneg').length;
 $('un').textContent=items.length-v.length;
 $('tot').textContent=items.length;
}}
function render(){{
 const it=items[i];
 $('img').src='/img/'+it.frame;
 $('fn').textContent=it.frame;
 $('pos').textContent=i+1;
 const lab=dec[it.frame];
 $('img').className=lab||'';
 $('verdict').className='verdict'+(lab?' v-'+lab:'');
 $('verdict').textContent=lab=='octopus'?'🐙 OCTOPUS':lab=='hardneg'?'∅ NO OCTOPUS':'';
 if(items[i+1])new Image().src='/img/'+items[i+1].frame;
 counts();
}}
function go(d){{ i=Math.max(0,Math.min(items.length-1,i+d)); render(); }}
async function send(frame,label){{
 await fetch('/mark',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify({{frame:frame,label:label}})}});
}}
function mark(label){{
 const f=items[i].frame;
 if(dec[f]==label){{ delete dec[f]; send(f,'clear'); render(); return; }}
 dec[f]=label; send(f,label); render(); setTimeout(()=>go(1),100);
}}
function clearOne(){{ const f=items[i].frame; delete dec[f]; send(f,'clear'); render(); }}
document.addEventListener('keydown',e=>{{
 if(e.key=='o'||e.key=='O'||e.key=='1')mark('octopus');
 else if(e.key=='n'||e.key=='N'||e.key=='0')mark('hardneg');
 else if(e.key==' '||e.key=='ArrowRight'){{e.preventDefault();go(1);}}
 else if(e.key=='ArrowLeft')go(-1);
 else if(e.key=='u'||e.key=='U')clearOne();
 else if(e.key=='f'||e.key=='F')window.open($('img').src);
}});
render();
</script></body></html>"""


if __name__ == "__main__":
    print(f"Right_Top hard-neg review ({len(load_items())} frames) -> http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
