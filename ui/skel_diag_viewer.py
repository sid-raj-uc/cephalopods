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
        return {"rows": [], "stats": {}}
    rows = json.load(open(DIAG / "summary.json"))
    ga = [r["gt_arms"] for r in rows]; ma = [r["model_arms"] for r in rows]
    n = max(1, len(rows))
    stats = {"n": len(rows),
             "gt_mean": round(sum(ga)/n, 2), "model_mean": round(sum(ma)/n, 2),
             "gt_ge6": sum(x >= 6 for x in ga), "model_ge6": sum(x >= 6 for x in ma)}
    return {"rows": rows, "stats": stats}


@app.get("/img/{name}")
def img(name: str):
    p = DIAG / name
    return FileResponse(str(p)) if p.exists() else JSONResponse({"error": "no"}, status_code=404)


@app.get("/chart")
def chart():
    return FileResponse(str(CHART)) if CHART.exists() else JSONResponse({"error": "no"}, status_code=404)


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
 <b>Phase 1 — mask vs skeletonizer</b>
 <span class=k>GT mask</span> <b id=gtm>-</b> <span class=k>arms · MODEL mask</span> <b id=mm>-</b> <span class=k>arms</span>
 <span class=k>| ≥6 arms:</span> GT <span id=gt6>-</span> · MODEL <span id=m6>-</span>
 <span style="margin-left:12px">frame <span id=pos>-</span>/<span id=tot>-</span> <span id=lbl class=k></span></span>
 <select id=sort><option value=gap>sort: biggest GT-model gap</option><option value=order>order</option></select>
 <button onclick="nav(-1)">◀</button><button onclick="nav(1)">▶</button>
 <span class=k style="margin-left:auto">left = GT mask skeleton · right = MODEL mask skeleton</span>
</div>
<div id=wrap>
 <div id=left><img id=cmp></div>
 <div id=right><h3>Arm-count summary (n=40)</h3><img id=chart src="/chart"></div>
</div>
<script>
let rows=[], order=[], pos=0;
async function load(){const d=await (await fetch('/api/summary')).json();
 rows=d.rows; const s=d.stats;
 gtm.textContent=s.gt_mean; mm.textContent=s.model_mean; gt6.textContent=s.gt_ge6+'/'+s.n; m6.textContent=s.model_ge6+'/'+s.n;
 document.getElementById('tot').textContent=rows.length; sortit(); show();}
function sortit(){const by=document.getElementById('sort').value;
 order=rows.map((r,i)=>i);
 if(by==='gap') order.sort((a,b)=>(rows[b].gt_arms-rows[b].model_arms)-(rows[a].gt_arms-rows[a].model_arms));
 pos=0;}
function show(){const r=rows[order[pos]]; if(!r)return;
 document.getElementById('cmp').src='/img/'+r.file+'?t='+Date.now();
 document.getElementById('pos').textContent=pos+1;
 document.getElementById('lbl').textContent=`(GT ${r.gt_arms} vs MODEL ${r.model_arms})`;}
function nav(d){pos=Math.max(0,Math.min(order.length-1,pos+d));show();}
document.getElementById('sort').onchange=()=>{sortit();show();};
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')nav(1);else if(e.key==='ArrowLeft')nav(-1);});
load();
</script></body></html>"""

if __name__ == "__main__":
    print("Skeleton diagnostic viewer -> http://localhost:8018")
    uvicorn.run(app, host="127.0.0.1", port=8018)
