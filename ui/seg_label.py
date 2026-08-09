"""Human-in-the-loop octopus mask labeler (FastAPI, port 8015) — the reliable GT path.

Auto-labeling failed on this footage: GroundingDINO grabs cloth/pipes, CLIP says "octopus"
everywhere, motion catches TVs/people. Every automatic localizer has a distractor. So put the
human exactly where the machines fail (LOCATING the octopus) and keep the machine where it's
strong (SAM2 making + propagating the mask from a correct point).

Per clip:
  1. PRE-SEED with the motion guess (largest motion blob box) -> SAM2 -> mask shown on the seed frame.
  2. If wrong (TV/person/empty), you CLICK the octopus (left = keep / right = exclude) -> SAM2 re-masks.
  3. Accept (A) -> propagate the mask through the clip -> save N clean (image, mask) pairs.
     Reject (R) -> skip (no octopus / unusable).  ←/→ navigate.

Output: data/dataset_seg_human/{images,masks,manifest.jsonl} — trustworthy GT for train + a real val set.
Resumable (skips clips already in the manifest). SAM2 on MPS/CUDA/CPU (auto).

Run:  venv/bin/python3 ui/seg_label.py   ->  http://localhost:8015
"""
import base64, glob, io, json, os, sys, tempfile, subprocess, threading
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from auto_segment import motion_seed, largest_blob, build_prompts

CLIPS_ROOT = Path(os.environ.get("SEG_LABEL_CLIPS", REPO / "src" / "octopus_clips_verified"))
CAMERAS = ["Right_Front", "Right_Back", "Right_Right"]      # colour dens; not Right_Left/Right_Top
OUT = REPO / "data" / "dataset_seg_human"
(OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "masks").mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "manifest.jsonl"
FPS = 3; MAXSIDE = 1024; N_PER_CLIP = 4
AREA_MIN, AREA_MAX = 0.0008, 0.6

app = FastAPI()
_LOCK = threading.Lock()
_SAM = None
CUR = {}   # live per-clip state


def sam():
    global _SAM
    if _SAM is None:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _SAM = (SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-small", device=dev), dev)
        print("SAM2 loaded on", dev, flush=True)
    return _SAM


def camera_of(p):
    for c in CAMERAS:
        if c in p:
            return c
    return None


def all_clips():
    return sorted(p for p in glob.glob(f"{CLIPS_ROOT}/**/*.mp4", recursive=True) if camera_of(p))


def done_set():
    d = set()
    if MANIFEST.exists():
        for l in open(MANIFEST):
            try:
                d.add(json.loads(l)["clip"])
            except Exception:
                pass
    return d


def source_video(clip):
    p = Path(clip); return f"{p.parent.parent.name}/{p.parent.name}"


def _seed_mask(st, predictor, seed_idx, box, points, labels):
    if box is None and not points:                 # no prompt yet -> empty mask (nothing to segment)
        return np.zeros(CUR["imgs"][seed_idx].size[::-1], bool)
    predictor.reset_state(st)
    predictor.add_new_points_or_box(
        st, frame_idx=seed_idx, obj_id=1,
        box=(np.array(box, np.float32) if box is not None else None),
        points=(np.array(points, np.float32) if points else None),
        labels=(np.array(labels, np.int32) if labels else None))
    # the mask for the seed frame comes back on the first propagate step from that frame
    for oi, _, logits in predictor.propagate_in_video(st, start_frame_idx=seed_idx, max_frame_num_to_track=0):
        m = (logits[0] > 0).cpu().numpy()[0]
        return largest_blob(m) if m.any() else m
    return np.zeros(CUR["imgs"][seed_idx].size[::-1], bool)


def _composite_b64(img, mask, points, labels):
    im = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR).copy()
    if mask is not None and mask.any():
        im[mask] = (0.5 * im[mask] + 0.5 * np.array([0, 235, 120])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (0, 0, 255), 2)
    for (x, y), l in zip(points, labels):
        c = (0, 220, 0) if l == 1 else (0, 0, 235)
        cv2.circle(im, (int(x), int(y)), 7, c, -1); cv2.circle(im, (int(x), int(y)), 7, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def load_clip(index):
    clips = all_clips()
    clip = clips[index]
    # clean up previous
    if CUR.get("td"):
        try: CUR["td"].cleanup()
        except Exception: pass
    td = tempfile.TemporaryDirectory()
    fdir = f"{td.name}/f"; os.makedirs(fdir)
    subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vf",
                    f"fps={FPS},scale='min({MAXSIDE},iw)':-2", f"{fdir}/%05d.jpg"], check=False)
    files = sorted(glob.glob(f"{fdir}/*.jpg"))
    imgs = [Image.open(f).convert("RGB") for f in files]
    predictor, dev = sam()
    st = predictor.init_state(video_path=fdir)
    ms = motion_seed(imgs) if imgs else None
    seed_idx = ms[0] if ms else (len(imgs) // 2 if imgs else 0)
    box = ms[1] if ms else None
    CUR.clear()
    CUR.update(dict(clip=clip, index=index, td=td, fdir=fdir, imgs=imgs, st=st,
                    seed_idx=seed_idx, box=box, points=[], labels=[], W=imgs[0].size[0], H=imgs[0].size[1]))
    mask = _seed_mask(st, predictor, seed_idx, box, [], []) if imgs else None
    CUR["mask"] = mask
    return {"index": index, "clip": clip, "camera": camera_of(clip), "W": CUR["W"], "H": CUR["H"],
            "seed_frac": None if ms is None else round((box[2]-box[0])*(box[3]-box[1])/(CUR["W"]*CUR["H"]), 3),
            "area": round(float(mask.mean()), 4) if mask is not None else 0.0,
            "img": _composite_b64(imgs[seed_idx], mask, [], [])}


@app.get("/api/state")
def api_state():
    clips = all_clips(); done = done_set()
    return {"total": len(clips), "done": sum(1 for c in clips if c in done),
            "clips_root": str(CLIPS_ROOT)}


@app.post("/api/load")
def api_load(body: dict):
    with _LOCK:
        idx = int(body.get("index", 0))
        clips = all_clips()
        if not clips:
            return JSONResponse({"error": f"no clips under {CLIPS_ROOT}"}, status_code=404)
        idx = max(0, min(idx, len(clips) - 1))
        r = load_clip(idx)
        r["is_done"] = clips[idx] in done_set()
        return r


@app.post("/api/click")
def api_click(body: dict):
    with _LOCK:
        if not CUR:
            return JSONResponse({"error": "no clip loaded"}, status_code=400)
        CUR["points"].append([float(body["x"]), float(body["y"])])
        CUR["labels"].append(int(body.get("label", 1)))
        predictor, _ = sam()
        # clicks REFINE the pre-seed: keep the motion box (if any) and add the points. If the box was
        # wrong (TV/person) the user hits Z first to clear it, then clicks the octopus fresh.
        mask = _seed_mask(CUR["st"], predictor, CUR["seed_idx"], CUR["box"], CUR["points"], CUR["labels"])
        CUR["mask"] = mask
        return {"area": round(float(mask.mean()), 4),
                "img": _composite_b64(CUR["imgs"][CUR["seed_idx"]], mask, CUR["points"], CUR["labels"])}


@app.post("/api/reset")
def api_reset(body: dict):
    with _LOCK:
        if not CUR:
            return JSONResponse({"error": "no clip"}, status_code=400)
        # clear EVERYTHING (drop the motion box + points) -> blank, so the user clicks the octopus fresh
        CUR["points"], CUR["labels"], CUR["box"] = [], [], None
        h, w = CUR["H"], CUR["W"]; mask = np.zeros((h, w), bool); CUR["mask"] = mask
        return {"area": 0.0, "img": _composite_b64(CUR["imgs"][CUR["seed_idx"]], mask, [], [])}


@app.post("/api/accept")
def api_accept(body: dict):
    with _LOCK:
        if not CUR:
            return JSONResponse({"error": "no clip"}, status_code=400)
        predictor, _ = sam(); st = CUR["st"]; seed = CUR["seed_idx"]; imgs = CUR["imgs"]
        # re-seed with the final prompt, then propagate BOTH directions to track the octopus
        predictor.reset_state(st)
        predictor.add_new_points_or_box(
            st, frame_idx=seed, obj_id=1,
            box=(np.array(CUR["box"], np.float32) if CUR["box"] is not None else None),
            points=(np.array(CUR["points"], np.float32) if CUR["points"] else None),
            labels=(np.array(CUR["labels"], np.int32) if CUR["labels"] else None))
        masks = [None] * len(imgs)
        for oi, _, lg in predictor.propagate_in_video(st, start_frame_idx=seed):
            masks[oi] = (lg[0] > 0).cpu().numpy()[0]
        for oi, _, lg in predictor.propagate_in_video(st, start_frame_idx=seed, reverse=True):
            masks[oi] = (lg[0] > 0).cpu().numpy()[0]
        masks = [largest_blob(m) if (m is not None and m.any()) else None for m in masks]
        areas = np.array([m.mean() if m is not None else 0.0 for m in masks])
        med = np.median(areas[areas > 0]) if (areas > 0).any() else 0.0
        good = [k for k in range(len(imgs)) if masks[k] is not None and AREA_MIN <= areas[k] <= AREA_MAX
                and (med == 0 or areas[k] <= 3 * med)]
        pick = sorted(set(good[i] for i in np.linspace(0, len(good) - 1, min(N_PER_CLIP, len(good))).astype(int))) if good else []
        clip = CUR["clip"]; vid = source_video(clip).replace("/", "_"); cam = camera_of(clip)
        n = 0
        with open(MANIFEST, "a") as mf:
            for j, k in enumerate(pick):
                stem = f"{vid}_{Path(clip).stem}_{cam}_{j}"
                imgs[k].save(OUT / "images" / f"{stem}.jpg", quality=90)
                Image.fromarray((masks[k] * 255).astype(np.uint8)).save(OUT / "masks" / f"{stem}.png")
                mf.write(json.dumps({"clip": clip, "camera": cam, "image": f"images/{stem}.jpg",
                                     "mask": f"masks/{stem}.png", "area": round(float(areas[k]), 4),
                                     "source": "human"}) + "\n")
                n += 1
        return {"saved": n}


@app.post("/api/reject")
def api_reject(body: dict):
    with _LOCK:
        if CUR:
            with open(MANIFEST, "a") as mf:
                mf.write(json.dumps({"clip": CUR["clip"], "camera": camera_of(CUR["clip"]),
                                     "image": None, "mask": None, "area": 0.0, "source": "reject"}) + "\n")
        return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Octopus mask labeler</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui;display:flex;flex-direction:column;height:100vh}
 #bar{padding:8px 12px;background:#1b1b1b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #bar b{color:#8f8} .k{color:#888}
 #wrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
 #cv{max-width:98%;max-height:98%;cursor:crosshair;border:1px solid #333}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer}
 button:hover{background:#333} .hint{color:#777;font-size:12px}
 #msg{color:#6cf}
</style></head><body>
<div id=bar>
 <b>Octopus mask labeler</b>
 <span>clip <span id=idx>-</span>/<span id=tot>-</span></span>
 <span class=k>saved pairs from</span> <span id=done>-</span> <span class=k>clips</span>
 <span id=cam class=k></span> <span>area <span id=area>-</span></span>
 <span id=msg></span>
 <span style="margin-left:auto" class=hint>left-click = octopus · right-click = not-octopus · A accept · R reject · Z reset · ←/→ nav</span>
</div>
<div id=wrap><img id=cv></div>
<script>
let idx=0, tot=0, W=1, H=1, busy=false;
const cv=document.getElementById('cv');
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});return r.json();}
function setImg(d){ if(d.img) cv.src=d.img; if(d.area!==undefined) document.getElementById('area').textContent=(d.area*100).toFixed(1)+'%'; }
async function refreshState(){const s=await (await fetch('/api/state')).json(); tot=s.total; document.getElementById('tot').textContent=tot; document.getElementById('done').textContent=s.done;}
async function load(i){ busy=true; msg('loading…'); const d=await post('/api/load',{index:i}); busy=false;
  if(d.error){msg(d.error);return;} idx=d.index; W=d.W; H=d.H; document.getElementById('idx').textContent=idx+1;
  document.getElementById('cam').textContent=d.camera+(d.is_done?' ✓done':''); setImg(d); msg(d.is_done?'already labeled (re-doing overwrites)':''); refreshState();}
function msg(t){document.getElementById('msg').textContent=t;}
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',async e=>{ if(busy)return; e.preventDefault();
  const r=cv.getBoundingClientRect(); const x=(e.clientX-r.left)/r.width*W; const y=(e.clientY-r.top)/r.height*H;
  const label=e.button===2?0:1; busy=true; const d=await post('/api/click',{x,y,label}); busy=false; setImg(d);});
document.addEventListener('keydown',async e=>{ if(busy)return;
  if(e.key==='a'||e.key==='A'){busy=true;msg('propagating…');const d=await post('/api/accept',{});busy=false;msg('saved '+d.saved+' pairs');await load(Math.min(idx+1,tot-1));}
  else if(e.key==='r'||e.key==='R'){busy=true;await post('/api/reject',{});busy=false;msg('rejected');await load(Math.min(idx+1,tot-1));}
  else if(e.key==='z'||e.key==='Z'){busy=true;const d=await post('/api/reset',{});busy=false;setImg(d);msg('reset to motion pre-seed');}
  else if(e.key==='ArrowRight'){load(Math.min(idx+1,tot-1));}
  else if(e.key==='ArrowLeft'){load(Math.max(idx-1,0));}});
(async()=>{await refreshState();await load(0);})();
</script></body></html>"""

if __name__ == "__main__":
    print("Octopus mask labeler -> http://localhost:8015")
    uvicorn.run(app, host="127.0.0.1", port=8015)
