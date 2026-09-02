"""Skeleton phase-results viewer (FastAPI, port 8018).

Browses the per-frame GT-mask vs MODEL-mask skeleton comparisons written by src/skel_diagnostic.py
(and later phases). Shows the arm-count summary + steps through the side-by-side overlays, sortable
worst-first (biggest GT-model gap) to see exactly where the model mask loses arms.

Run: venv/bin/python3 ui/skel_diag_viewer.py -> http://localhost:8018
"""
import json, sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

REPO = Path(__file__).resolve().parent.parent
DIAG = REPO / "data" / "skel_diag"
CHART = REPO / "results" / "segmentation" / "skel_phase1_armcount.png"
app = FastAPI()


@app.get("/api/summary")
def summary():
    if not (DIAG / "summary.json").exists():
        return {"rows": [], "stats": {}, "meta": {}}
    d = json.load(open(DIAG / "summary.json"))
    # normalize both formats: phase 1 was a bare list with gt_arms/model_arms
    if isinstance(d, list):
        rows = [{"file": r["file"], "left_arms": r["gt_arms"], "right_arms": r["model_arms"]} for r in d]
        meta = {"title": "Phase 1 — GT vs MODEL mask", "left": "GT mask", "right": "MODEL mask"}
    else:
        rows = d["rows"]; meta = d.get("meta", {})
    n = max(1, len(rows))
    la = [r["left_arms"] for r in rows]; ra = [r["right_arms"] for r in rows]
    stats = {"n": len(rows), "left_mean": round(sum(la)/n, 2), "right_mean": round(sum(ra)/n, 2),
             "left_ge6": sum(x >= 6 for x in la), "right_ge6": sum(x >= 6 for x in ra)}
    return {"rows": rows, "stats": stats, "meta": meta}


@app.get("/img/{name}")
def img(name: str):
    p = DIAG / name
    return FileResponse(str(p)) if p.exists() else JSONResponse({"error": "no"}, status_code=404)


@app.get("/chart")
def chart():
    p = DIAG / "chart.png"                       # current-phase chart (each phase writes it here)
    if not p.exists():
        p = CHART
    return FileResponse(str(p)) if p.exists() else JSONResponse({"error": "no"}, status_code=404)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Skeleton diagnostic</title><style>
 body{margin:0;background:#0f1013;color:#ddd;font:14px system-ui}
 #bar{padding:9px 13px;background:#1a1c22;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 b{color:#8f8} .k{color:#888}
 button,select{background:#25272e;color:#eee;border:1px solid #3a3d46;border-radius:6px;padding:6px 11px;cursor:pointer}
 #wrap{display:flex;gap:12px;padding:12px;align-items:flex-start}
 #left{flex:2;min-width:0} #right{flex:1;min-width:280px}
 img{max-width:100%;border:1px solid #2a2d35;border-radius:6px;display:block}
 h3{margin:6px 0;color:#bbb;font-size:13px}
</style></head><body>
<div id=bar>
 <b id=title>Skeleton phase results</b>
 <span class=k id=llab>left</span> <b id=gtm>-</b> <span class=k>arms ·</span> <span class=k id=rlab>right</span> <b id=mm>-</b> <span class=k>arms</span>
 <span class=k>| ≥6 arms:</span> <span id=gt6>-</span> · <span id=m6>-</span>
 <span style="margin-left:12px">frame <span id=pos>-</span>/<span id=tot>-</span> <span id=lbl class=k></span></span>
 <select id=sort><option value=gap>sort: biggest gap</option><option value=order>order</option></select>
 <button onclick="nav(-1)">◀</button><button onclick="nav(1)">▶</button>
 <span class=k style="margin-left:auto" id=hint>left vs right skeleton</span>
</div>
<div id=wrap>
 <div id=left><img id=cmp></div>
 <div id=right>
  <h3>Legend</h3>
  <div style="background:#16181d;border:1px solid #2a2d35;border-radius:8px;padding:12px;font-size:13px;line-height:2">
   <div><span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:#00e678;border:1.5px solid #1e1e1e;vertical-align:-2px"></span>&nbsp; <b style="color:#7ed47e">Head</b> <span class=k>(at the neck; short green line links it to the body)</span></div>
   <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#ffd700;border:1.5px solid #1e1e1e;vertical-align:-2px"></span>&nbsp; <b style="color:#fd6">Arm tips</b></div>
   <div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#fff;border:1.5px solid #1e1e1e;vertical-align:-1px"></span>&nbsp; <b>Arm landmarks</b> <span class=k>(base / mid)</span></div>
   <div><span style="display:inline-block;width:26px;height:4px;border-radius:2px;background:linear-gradient(90deg,#e64646,#f09123,#41af4b,#3773e1,#9155d2);vertical-align:2px"></span>&nbsp; <b>Arm splines</b> <span class=k>(one colour per arm, consistent IDs)</span></div>
   <div class=k style="line-height:1.5;margin-top:6px">Body-centre marker hidden — the arm splines converge there.</div>
  </div>
 </div>
</div>
<script>
let rows=[], order=[], pos=0;
async function load(){const d=await (await fetch('/api/summary')).json();
 rows=d.rows; const s=d.stats, m=d.meta||{};
 document.getElementById('title').textContent=m.title||'Skeleton phase results';
 document.getElementById('llab').textContent=m.left||'left'; document.getElementById('rlab').textContent=m.right||'right';
 document.getElementById('hint').textContent='left = '+(m.left||'left')+' · right = '+(m.right||'right');
 gtm.textContent=s.left_mean; mm.textContent=s.right_mean; gt6.textContent=s.left_ge6+'/'+s.n; m6.textContent=s.right_ge6+'/'+s.n;
 document.getElementById('tot').textContent=rows.length; sortit(); show();}
function sortit(){const by=document.getElementById('sort').value;
 order=rows.map((r,i)=>i);
 if(by==='gap') order.sort((a,b)=>Math.abs(rows[b].left_arms-rows[b].right_arms)-Math.abs(rows[a].left_arms-rows[a].right_arms));
 pos=0;}
function show(){const r=rows[order[pos]]; if(!r)return;
 document.getElementById('cmp').src='/img/'+r.file+'?t='+Date.now();
 document.getElementById('pos').textContent=pos+1;
 document.getElementById('lbl').textContent=`(${r.left_arms} vs ${r.right_arms})`;}
function nav(d){pos=Math.max(0,Math.min(order.length-1,pos+d));show();}
document.getElementById('sort').onchange=()=>{sortit();show();};
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')nav(1);else if(e.key==='ArrowLeft')nav(-1);});
load();
</script></body></html>"""

if __name__ == "__main__":
    print("Skeleton diagnostic viewer -> http://localhost:8018")
    uvicorn.run(app, host="127.0.0.1", port=8018)
