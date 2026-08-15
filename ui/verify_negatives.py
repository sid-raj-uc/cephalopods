#!/usr/bin/env python3
"""verify_negatives.py — human confirmation of the staged negative sets (port 8020).

WHY THIS EXISTS. EMPTY-V2 (`data/empty_negatives/`, 120 frames / 60 videos) and the reflection set
(`data/reflection_negatives/`, 42 reviewed) were labelled by an AI vision model, not a human. Every
result computed on them (PAPER_NOTES R9, R10, R13) is therefore held OUT of the paper until a human
confirms the labels. R13 in particular RAISES a headline number (presence AUC 0.794 -> 0.917), which
is exactly the kind of result that should not go in on a machine's say-so.

The repo has a scar here: 166 of 232 assumed-negative frames turned out to contain the octopus.

WHAT YOU DO. Each frame shows with the model's proposed label. Confirm it or override it:
    E / 1   empty      (no octopus anywhere — a valid negative)
    O / 2   octopus    (animal or arm visible — NOT a negative)
    A / 3   ambiguous  (cannot tell — excluded from scoring either way)
    <- / -> navigate         U  clear this frame's human label
    F       full-resolution in a new tab
Every keypress saves immediately (resumable); reload returns you to the first unconfirmed frame.

Human labels are written to `human` / `human_at` and NEVER overwrite the model's `review` field, so
model-vs-human agreement stays measurable. Only `human` counts for scoring once you have started.

Run:  venv/bin/python3 ui/verify_negatives.py        then open http://localhost:8020
"""
import argparse, datetime, json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "empty": ROOT / "data" / "empty_negatives",
    "reflection": ROOT / "data" / "reflection_negatives",
}
app = FastAPI()
STATE = {}


def load(name):
    d = SETS[name]
    idx = json.load(open(d / "index.json"))
    rows = [r for r in idx["rows"] if (d / r["image"]).exists()]
    # reflection set was only reviewed to index 41; don't ask for frames the model never judged
    rows = [r for r in rows if r.get("review") is not None]
    return d, idx, rows


def save(name):
    d, idx, _ = STATE[name]
    json.dump(idx, open(d / "index.json", "w"), indent=1)


@app.on_event("startup")
def boot():
    for n in SETS:
        if (SETS[n] / "index.json").exists():
            STATE[n] = load(n)
            print(f"  {n}: {len(STATE[n][2])} frames")


@app.get("/api/rows")
def rows(set: str = "empty"):
    if set not in STATE:
        return JSONResponse({"error": f"no staged set '{set}'"}, status_code=404)
    _, _, rs = STATE[set]
    out = [{"i": i, "key": r["key"], "video": r.get("video"), "camera": r.get("camera", ""),
            "model": r.get("review"), "human": r.get("human")} for i, r in enumerate(rs)]
    first = next((o["i"] for o in out if not o["human"]), 0)
    done = sum(1 for o in out if o["human"])
    agree = sum(1 for o in out if o["human"] and o["human"] == o["model"])
    return {"rows": out, "first_unreviewed": first, "n": len(out), "done": done, "agree": agree}


@app.get("/img")
def img(set: str, i: int):
    d, _, rs = STATE[set]
    return FileResponse(d / rs[i]["image"])


@app.post("/api/label")
def label(set: str, i: int, value: str = ""):
    d, idx, rs = STATE[set]
    r = rs[i]
    if value:
        r["human"] = value
        r["human_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    else:
        r.pop("human", None); r.pop("human_at", None)
    save(set)
    done = sum(1 for x in rs if x.get("human"))
    agree = sum(1 for x in rs if x.get("human") and x["human"] == x.get("review"))
    return {"ok": True, "done": done, "n": len(rs), "agree": agree}


PAGE = """
<style>
 body{background:#111;color:#eee;font:14px system-ui;margin:0;padding:12px}
 #top{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
 button,select{background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;font:14px system-ui;cursor:pointer}
 button.on{background:#2a78d6;border-color:#2a78d6}
 #wrap{text-align:center}
 img{max-width:100%;max-height:74vh;border:1px solid #333;border-radius:6px}
 .tag{padding:3px 9px;border-radius:99px;font-weight:600}
 .empty{background:#1d5c2f}.present{background:#a33}.ambiguous{background:#7a5c12}
 .octopus_present{background:#a33}
 #hint{color:#888;font-size:12px}
</style>
<div id=top>
 <select id=set onchange="boot()"><option value=empty>EMPTY-V2</option><option value=reflection>reflection</option></select>
 <span id=pos></span><span id=prog></span>
 <span>model: <span id=model class=tag>-</span></span>
 <span>human: <span id=human class=tag>-</span></span>
 <button onclick="lab('empty')">E · empty</button>
 <button onclick="lab('octopus_present')">O · octopus</button>
 <button onclick="lab('ambiguous')">A · ambiguous</button>
 <button onclick="lab('')">U · clear</button>
 <button onclick="full()">F · full-res</button>
</div>
<div id=wrap><img id=im></div>
<div id=hint>E/1 empty · O/2 octopus · A/3 ambiguous · U clear · ←/→ navigate · F full-res. Saves on every keypress.</div>
<script>
let R=[],i=0,S='empty';
async function boot(){S=document.getElementById('set').value;
 const d=await (await fetch('/api/rows?set='+S)).json();
 if(d.error){alert(d.error);return} R=d.rows;i=d.first_unreviewed;draw();upd(d)}
function upd(d){document.getElementById('prog').textContent=
 `— ${d.done}/${d.n} confirmed, ${d.agree} agree with model`}
function draw(){const r=R[i];if(!r)return;
 document.getElementById('im').src=`/img?set=${S}&i=${i}&_=${Date.now()}`;
 document.getElementById('pos').textContent=`#${i} ${r.camera} ${r.video}`;
 const m=document.getElementById('model');m.textContent=r.model||'-';m.className='tag '+(r.model||'');
 const h=document.getElementById('human');h.textContent=r.human||'-';h.className='tag '+(r.human||'');}
async function lab(v){const d=await (await fetch(`/api/label?set=${S}&i=${i}&value=${v}`,{method:'POST'})).json();
 R[i].human=v||null;upd(d);if(v&&i<R.length-1){i++}draw()}
function full(){window.open(`/img?set=${S}&i=${i}`,'_blank')}
document.onkeydown=e=>{const k=e.key.toLowerCase();
 if(k==='arrowright'){i=Math.min(i+1,R.length-1);draw()}
 else if(k==='arrowleft'){i=Math.max(i-1,0);draw()}
 else if(k==='e'||k==='1')lab('empty');
 else if(k==='o'||k==='2')lab('octopus_present');
 else if(k==='a'||k==='3')lab('ambiguous');
 else if(k==='u')lab('');
 else if(k==='f')full()};
boot();
</script>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    a = ap.parse_args()
    print(f"verify_negatives -> http://localhost:{a.port}")
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
