"""Static skeletonization viewer (FastAPI, port 8019).

Pick a benchmark/dataset frame (or paste any image path) -> segments it with the trained model
(thin768), optionally SAM2-refines the mask, extracts the anatomical skeleton, and shows the
overlay (mask + arm splines + mantle/head/tips) with stats. Per-image, static — the cleanest way
to inspect skeletonization quality without the temporal layer.

Run: venv/bin/python3 ui/skel_static_viewer.py -> http://localhost:8019
"""
import base64, json, sys, threading
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "src" / "skeleton"))
from segment_octopus import OctoSegmenter, _largest_blob
from skel_head_fix import full_graph, plaus
from seg_skeleton_pipeline import _draw_skeleton, DEFAULT_CKPT

DS = REPO / "data" / "dataset_seg_human"
BENCH = REPO / "data" / "skel_bench50" / "frames.json"
HEAD_GT = REPO / "data" / "skel_bench50" / "head_gt.json"
app = FastAPI()
_LOCK = threading.Lock()
_S = None


def _head_gt():
    return json.load(open(HEAD_GT)) if HEAD_GT.exists() else {}


def seg():
    global _S
    if _S is None:
        _S = OctoSegmenter(str(DEFAULT_CKPT))
    return _S


@app.get("/api/suggestions")
def suggestions():
    out = []
    if BENCH.exists():
        for f in json.load(open(BENCH)):
            out.append(str(DS / f["image"]))
    return {"images": out}


@app.post("/api/run")
def run(body: dict):
    p = (body.get("image") or "").strip()
    if not p or not Path(p).exists():
        return JSONResponse({"error": f"not found: {p}"}, status_code=404)
    refine = bool(body.get("refine", False))
    with _LOCK:
        img = cv2.imread(p)
        if img is None:
            return JSONResponse({"error": "unreadable image"}, status_code=400)
        S = seg()
        mm, area = S.segment(img)
        if refine and mm.any():
            from mask_refine import sam2_refine
            mm = sam2_refine(img, mm, largest_blob=_largest_blob)
        m255 = (mm.astype(np.uint8)) * 255
        nodes, edges = full_graph(m255)
        vis = cv2.addWeighted(img, 0.62, np.zeros_like(img), 0.38, 0)
        sel = m255 > 0
        vis[sel] = (0.75 * vis[sel] + 0.25 * np.array([60, 150, 60])).astype(np.uint8)
        arms, head_ok = 0, False
        if nodes is not None:
            _draw_skeleton(vis, nodes, edges, 2)
            arms = len({n["branch_id"] for n in nodes if n["branch_id"] > 0})
            c = next((n for n in nodes if n["is_center"]), None)
            hd = next((n for n in nodes if n.get("is_head")), None)
            bases = [(n["x"], n["y"]) for n in nodes if "Base" in n.get("body_part", "")]
            if c and hd and len(bases) >= 2:
                head_ok = plaus((hd["x"], hd["y"]), (c["x"], c["y"]),
                                tuple(np.mean(np.asarray(bases, float), axis=0)))
        # draw existing human head-GT (magenta ring) for feedback while labelling
        gt = _head_gt().get(p)
        if gt:
            cv2.circle(vis, (int(gt[0]), int(gt[1])), 11, (255, 0, 255), 2, cv2.LINE_AA)
        h, w = vis.shape[:2]
        scale = 1.0
        if w > 1400:
            scale = 1400.0 / w
            vis = cv2.resize(vis, (1400, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return {"img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                "arms": arms, "head_ok": bool(head_ok), "scale": scale,
                "n_head_gt": len(_head_gt()),
                "mask_area_pct": round(float((m255 > 0).mean()) * 100, 2),
                "refined": refine}


@app.post("/api/head_gt")
def head_gt(body: dict):
    """Record a human head click (original-image coordinates) for a benchmark frame."""
    p = (body.get("image") or "").strip()
    if not p:
        return JSONResponse({"error": "no image"}, status_code=400)
    d = _head_gt()
    d[p] = [float(body["x"]), float(body["y"])]
    HEAD_GT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(HEAD_GT, "w"), indent=1)
    return {"n_head_gt": len(d)}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Static skeletonization</title><style>
 body{margin:0;background:#0f1013;color:#ddd;font:14px system-ui}
 #bar{padding:10px 14px;background:#1a1c22;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 select,input,button{background:#25272e;color:#eee;border:1px solid #3a3d46;border-radius:6px;padding:7px 10px}
 button{cursor:pointer} button:hover{background:#30333c}
 #msg{color:#6cf} b{color:#8f8}
 #wrap{display:flex;justify-content:center;padding:12px}
 img{max-width:98%;border:1px solid #2a2d35;border-radius:8px}
 .hint{color:#777;font-size:12px}
</style></head><body>
<div id=bar>
 <b>Static skeletonization</b>
 <select id=sug><option value="">— pick a benchmark frame —</option></select>
 <button onclick="nav(-1)">◀</button><button onclick="nav(1)">▶</button>
 <input id=path placeholder="or paste any image path" size=40>
 <label class=hint><input type=checkbox id=refine checked> SAM2 refine</label>
 <label class=hint style="color:#f6f"><input type=checkbox id=labelhead> label-head mode (click the eyes)</label>
 <span class=hint>GT: <span id=ngt>0</span>/50</span>
 <button onclick="go()">▶ Skeletonize</button>
 <span id=stats></span> <span id=msg></span>
</div>
<div id=wrap><img id=out style="cursor:crosshair"></div>
<script>
let imgs=[], idx=-1, lastScale=1, lastImage='';
async function load(){const d=await (await fetch('/api/suggestions')).json(); imgs=d.images||[];
 const s=document.getElementById('sug');
 imgs.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+': '+p.split('/').pop();s.appendChild(o);});
 s.onchange=()=>{idx=parseInt(s.value); if(!isNaN(idx)) go();};}
function msg(t){document.getElementById('msg').textContent=t;}
async function go(){
 let image=document.getElementById('path').value.trim();
 if(!image && idx>=0) image=imgs[idx];
 if(!image){msg('pick a frame or paste a path');return;}
 const refine=document.getElementById('refine').checked;
 msg(refine?'segmenting + SAM2 refining…':'segmenting…');
 const r=await (await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image,refine})})).json();
 if(r.error){msg(r.error);return;}
 lastScale=r.scale||1; lastImage=image;
 document.getElementById('out').src=r.img;
 document.getElementById('ngt').textContent=r.n_head_gt||0;
 document.getElementById('stats').innerHTML=`<b>${r.arms} arms</b> · head ${r.head_ok?'ok':'off'} · mask ${r.mask_area_pct}%${r.refined?' · SAM2-refined':''}`;
 msg('');}
document.getElementById('out').addEventListener('click', async e=>{
 if(!document.getElementById('labelhead').checked || !lastImage) return;
 const img=e.target, rect=img.getBoundingClientRect();
 const x=(e.clientX-rect.left)*(img.naturalWidth/rect.width)/lastScale;
 const y=(e.clientY-rect.top)*(img.naturalHeight/rect.height)/lastScale;
 const r=await (await fetch('/api/head_gt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:lastImage,x,y})})).json();
 document.getElementById('ngt').textContent=r.n_head_gt;
 msg('head GT saved ('+r.n_head_gt+') — advancing');
 setTimeout(()=>nav(1), 350);
});
function nav(d){ if(!imgs.length) return; idx=Math.max(0,Math.min(imgs.length-1,(idx<0?0:idx+d)));
 document.getElementById('sug').value=idx; document.getElementById('path').value=''; go(); }
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')nav(1);else if(e.key==='ArrowLeft')nav(-1);});
load();
</script></body></html>"""

if __name__ == "__main__":
    print("Static skeletonization viewer -> http://localhost:8019")
    uvicorn.run(app, host="127.0.0.1", port=8019)
