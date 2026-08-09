"""Eval review UI (FastAPI, port 8016) — step through the held-out val set one frame at a time,
comparing the model PREDICTION (green outline) vs your GROUND TRUTH (red fill), with per-frame IoU.

Reproduces the exact train/val split train_segmenter.py uses (split BY SOURCE VIDEO, seed 42,
val_frac 0.2) so you're looking at the real held-out frames. Sort by IoU (worst-first to inspect
failures) or by order. Toggle overlays. Flag a frame if the GT itself looks wrong (writes
data/dataset_seg_human/_eval_flags.txt) so we can fix labels.

Run:  venv/bin/python3 ui/seg_eval_review.py   ->  http://localhost:8016
Model via env SEG_EVAL_CKPT (default weights/seg/octo_seg_human_lraspp.pt).
"""
import base64, json, os, sys, threading
from pathlib import Path
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from segment_octopus import OctoSegmenter

DS = REPO / "data" / "dataset_seg_human"
CKPT = Path(os.environ.get("SEG_EVAL_CKPT", REPO / "weights" / "seg" / "octo_seg_human_lraspp.pt"))
FLAGS = DS / "_eval_flags.txt"
VAL_FRAC = float(os.environ.get("SEG_EVAL_VALFRAC", 0.2))
SEED = int(os.environ.get("SEG_EVAL_SEED", 42))

app = FastAPI()
_LOCK = threading.Lock()
_SEG = None
_ROWS = None      # cached [{image,mask,clip,iou,pred_area,gt_area}], computed once


def seg():
    global _SEG
    if _SEG is None:
        _SEG = OctoSegmenter(str(CKPT))
        print("model loaded:", CKPT.name, flush=True)
    return _SEG


def source_video(clip):
    p = Path(clip); return f"{p.parent.parent.name}/{p.parent.name}"


def val_rows():
    rows = [json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("image") and r.get("mask") and r.get("source") == "human"]
    rng = np.random.RandomState(SEED)
    vids = sorted({source_video(r["clip"]) for r in rows}); rng.shuffle(vids)
    val = set(vids[:max(1, int(len(vids) * VAL_FRAC))])
    return [r for r in rows if source_video(r["clip"]) in val]


def compute():
    global _ROWS
    S = seg(); out = []
    for r in val_rows():
        im = cv2.imread(str(DS / r["image"])); gt = cv2.imread(str(DS / r["mask"]), 0) > 127
        pred, area = S.segment(im)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        inter = (pred & gt).sum(); union = (pred | gt).sum(); iou = float(inter / union) if union > 0 else 1.0
        out.append({"image": r["image"], "mask": r["mask"], "clip": r["clip"],
                    "video": source_video(r["clip"]), "iou": round(iou, 3),
                    "pred_area": round(float(pred.mean()), 4), "gt_area": round(float(gt.mean()), 4)})
    _ROWS = out
    return out


def rows_cached():
    return _ROWS if _ROWS is not None else compute()


def flags():
    return set(l.strip() for l in open(FLAGS)) if FLAGS.exists() else set()


def render(i, show_gt=True, show_pred=True):
    r = rows_cached()[i]
    im = cv2.imread(str(DS / r["image"]))
    gt = cv2.imread(str(DS / r["mask"]), 0) > 127
    S = seg(); pred, _ = S.segment(im)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ov = im.copy()
    if show_gt and gt.any():
        ov[gt] = (0.5 * ov[gt] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)   # GT = red fill
    if show_pred and pred.any():
        cnts, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)                            # pred = green outline
    h, w = ov.shape[:2]
    if w > 1280:
        ov = cv2.resize(ov, (1280, int(h * 1280 / w)))
    ok, buf = cv2.imencode(".jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.get("/api/list")
def api_list(sort: str = "iou"):
    with _LOCK:
        rows = rows_cached()
        order = sorted(range(len(rows)), key=lambda i: rows[i]["iou"]) if sort == "iou" else list(range(len(rows)))
        ious = [rows[i]["iou"] for i in range(len(rows))]
        fl = flags()
        return {"n": len(rows), "order": order,
                "mean_iou": round(float(np.mean(ious)), 3) if ious else 0,
                "median_iou": round(float(np.median(ious)), 3) if ious else 0,
                "rows": [{"iou": r["iou"], "video": r["video"], "pred_area": r["pred_area"],
                          "gt_area": r["gt_area"], "flagged": r["image"] in fl} for r in rows]}


@app.post("/api/frame")
def api_frame(body: dict):
    with _LOCK:
        i = int(body.get("i", 0)); rows = rows_cached()
        i = max(0, min(i, len(rows) - 1)); r = rows[i]
        return {"i": i, "img": render(i, body.get("gt", True), body.get("pred", True)),
                "iou": r["iou"], "video": r["video"], "clip": Path(r["clip"]).name,
                "pred_area": r["pred_area"], "gt_area": r["gt_area"], "flagged": r["image"] in flags()}


@app.post("/api/flag")
def api_flag(body: dict):
    with _LOCK:
        r = rows_cached()[int(body["i"])]; img = r["image"]; fl = flags()
        if img in fl:
            fl.discard(img)
        else:
            fl.add(img)
        FLAGS.write_text("\n".join(sorted(fl)) + ("\n" if fl else ""))
        return {"flagged": img in fl, "n_flagged": len(fl)}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Seg eval review</title><style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui;display:flex;flex-direction:column;height:100vh}
 #bar{padding:8px 12px;background:#1b1b1b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 b{color:#8f8} .k{color:#888}
 #wrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden}
 #cv{max-width:98%;max-height:98%;border:1px solid #333}
 .lg{padding:2px 8px;border-radius:4px} .gt{background:#c00}.pr{color:#0f0;border:1px solid #0f0}
 .flag{background:#a60;color:#fff}
 label{display:flex;gap:4px;align-items:center}
</style></head><body>
<div id=bar>
 <b>Eval review</b> <span><span id=pos>-</span>/<span id=tot>-</span></span>
 <span class=k>mean IoU</span> <b id=miou>-</b> <span class=k>median</span> <b id=medi>-</b>
 <span>this: IoU <b id=iou>-</b></span>
 <span class="lg gt">GT (red fill)</span> <span class="lg pr">pred (green outline)</span>
 <span id=vid class=k></span> <span>pred area <span id=pa>-</span> · gt area <span id=ga>-</span></span>
 <label><input type=checkbox id=cgt checked>GT</label>
 <label><input type=checkbox id=cpr checked>pred</label>
 <select id=sort><option value=iou>sort: worst IoU first</option><option value=order>sort: capture order</option></select>
 <button onclick="flag()" id=flagbtn>⚑ flag bad GT (F)</button>
 <span id=msg class=k></span>
 <span style="margin-left:auto" class=k>←/→ nav · F flag · G/P toggle overlays</span>
</div>
<div id=wrap><img id=cv></div>
<script>
let order=[], pos=0, busy=false;
const cv=document.getElementById('cv');
async function post(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})})).json();}
async function loadList(){const s=document.getElementById('sort').value; const d=await (await fetch('/api/list?sort='+s)).json();
  order=d.order; document.getElementById('tot').textContent=d.n; document.getElementById('miou').textContent=d.mean_iou; document.getElementById('medi').textContent=d.median_iou; pos=0; show();}
async function show(){ if(busy)return; busy=true;
  const i=order[pos]; const d=await post('/api/frame',{i,gt:document.getElementById('cgt').checked,pred:document.getElementById('cpr').checked});
  busy=false; cv.src=d.img; document.getElementById('pos').textContent=pos+1;
  document.getElementById('iou').textContent=d.iou; document.getElementById('vid').textContent=d.video+' / '+d.clip;
  document.getElementById('pa').textContent=(d.pred_area*100).toFixed(1)+'%'; document.getElementById('ga').textContent=(d.gt_area*100).toFixed(1)+'%';
  document.getElementById('flagbtn').style.background=d.flagged?'#fa0':''; }
async function flag(){const i=order[pos]; const d=await post('/api/flag',{i}); document.getElementById('msg').textContent=d.flagged?'flagged bad-GT ('+d.n_flagged+')':'unflagged ('+d.n_flagged+')'; show();}
document.getElementById('sort').onchange=loadList;
document.getElementById('cgt').onchange=show; document.getElementById('cpr').onchange=show;
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){pos=Math.min(pos+1,order.length-1);show();}
  else if(e.key==='ArrowLeft'){pos=Math.max(pos-1,0);show();}
  else if(e.key==='f'||e.key==='F'){flag();}
  else if(e.key==='g'||e.key==='G'){document.getElementById('cgt').checked=!document.getElementById('cgt').checked;show();}
  else if(e.key==='p'||e.key==='P'){document.getElementById('cpr').checked=!document.getElementById('cpr').checked;show();}});
loadList();
</script></body></html>"""

if __name__ == "__main__":
    print("Seg eval review -> http://localhost:8016  (computing predictions on first load…)")
    uvicorn.run(app, host="127.0.0.1", port=8016)
