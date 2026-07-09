"""
Octopus clip extractor — FAST variant (single decode pass + parallel + GPU-decode).

Same behavior/outputs as extract_octopus_clips.py — reads/writes the SAME JSONs, the
SAME schema, the SAME resume/tracking (skips videos already in the processed ledger,
skips clips whose mp4 already exists). Just faster:

  1. SINGLE decode pass — octopus (CLIP+MLP) AND motion are computed from ONE ffmpeg
     1 fps stream (the plain extractor decodes each video twice). ~2x.
  2. PARALLEL videos — a thread pool decodes/scans N videos at once (decode is I/O-bound,
     so this scales near-linearly). `--workers N`.
  3. GPU decode — `--hwaccel cuda` uses NVDEC on an A100 box (much faster than CPU decode).
     CLIP runs on CUDA automatically if available.

Outputs (identical to the plain extractor):
  octopus_clips_verified/{date}/{segment}/{Camera}_{start}-{end}.mp4
  octopus_clips_verified.json      (clip index — appended, existing captions preserved)
  octopus_clips_processed.json     (processed-video ledger — resume; never reprocess)

Usage (on the A100 box):
  python3 extract_octopus_clips_fast.py --workers 12 --hwaccel cuda
  python3 extract_octopus_clips_fast.py --date 2026-02-22 --workers 8
"""
import argparse, datetime, json, re, subprocess, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server_creds import USER, PASS

CKPT_PATH  = HERE / "clip_mlp_hardneg_v2.pt"
CLIPS_DIR  = HERE / "octopus_clips_verified"
INDEX_JSON = HERE / "octopus_clips_verified.json"
PROCESSED  = HERE / "octopus_clips_processed.json"

BASE = "https://repo.octopus-intelligence.org/public/O-vulgaris-Nity-2026-2-20--"
CAMERAS = ["Right Back", "Right Front", "Right Left", "Right Right", "Right Top"]

# ── gates (identical to extract_octopus_clips.py) ──
SAMPLE_FPS       = 1.0
CLIP_LEN         = 20
MIN_VISIBLE_FRAC = 0.50
VIS_THRESH       = 0.60
MOTION_THRESH    = 0.008
MOTION_PIX       = 25
DW, DH, BATCH    = 640, 360, 64          # single-pass decode size (16:9), CLIP batch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_model_lock = threading.Lock()           # serialize GPU inference across worker threads
_json_lock  = threading.Lock()           # serialize registry writes
_CM = _PP = _CLF = _VIS = None           # shared model (loaded once)


def auth(url: str) -> str:
    return url.replace("https://", f"https://{USER}:{PASS}@")

def letterbox(img, size=224, fill=(128, 128, 128)):
    w, h = img.size; s = size / max(w, h); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    cv = Image.new("RGB", (size, size), fill); cv.paste(img, ((size - nw) // 2, (size - nh) // 2)); return cv


# ── server enumeration (identical) ──
def _curl(url):
    return subprocess.run(["curl", "-s", "--user", f"{USER}:{PASS}", url], capture_output=True, text=True).stdout

def list_dates():
    return sorted(set(re.findall(r'href="(\d{4}-\d{2}-\d{2})/"', _curl(f"{BASE}/Right%20Top/Local/"))))

def list_segments(cam, date):
    enc = urllib.parse.quote(cam); out = _curl(f"{BASE}/{enc}/Local/{date}/"); rows = []
    for f in re.findall(r'href="([^"]+\.mp4)"', out):
        m = re.match(r"(\d+)--", f)
        if not m: continue
        seg = m.group(1); cam_us = cam.replace(" ", "_")
        rows.append({"video": f"data/aquarium/full/{date}/{seg}/{cam_us}.mp4",
                     "date": date, "segment": seg, "camera": cam_us,
                     "url": f"{BASE}/{enc}/Local/{date}/{f}"})
    return rows

def enumerate_candidates(dates, cams):
    tasks = [(c, d) for d in dates for c in cams]; out = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(lambda a: list_segments(*a), tasks): out.extend(r)
    return out


# ── registries (identical schema) ──
def load_json(path, default):
    return json.load(open(path)) if path.exists() else default

def init_registries():
    proc = load_json(PROCESSED, {"task": "octopus_clip_extraction",
        "description": "Videos processed by the consolidated 20s-clip pipeline. Do not reprocess.",
        "updated_at": None, "count": 0, "processed": []})
    idx = load_json(INDEX_JSON, {"description": "Extracted 20s octopus clips.",
        "model": CKPT_PATH.name, "updated_at": None, "count": 0, "clips": []})
    return proc, idx

def save_registries(proc, idx):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    proc["count"] = len(proc["processed"]); proc["updated_at"] = now
    idx["count"] = len(idx["clips"]); idx["updated_at"] = now
    json.dump(proc, open(PROCESSED, "w"), indent=2)
    json.dump(idx, open(INDEX_JSON, "w"), indent=2)


# ── model ──
def build_clf(ck):
    feat = ck["feat_dim"]; arch = ck.get("arch", "linear")
    if arch == "linear": return nn.Linear(feat, 2)
    hid = [int(x) for x in arch.replace("mlp_", "").split("_")]; dims = [feat] + hid + [2]; L = []
    for i in range(len(dims) - 1):
        L.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2: L += [nn.ReLU(), nn.Dropout(0.3)]
    return nn.Sequential(*L)

def load_model():
    global _CM, _PP, _CLF, _VIS
    try:
        import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception: pass
    import clip as clip_lib
    ck = torch.load(CKPT_PATH, map_location=device)
    _CM, _PP = clip_lib.load(ck["clip_model"], device=device); _CM.eval()
    _CLF = build_clf(ck).to(device); _CLF.load_state_dict(ck["state_dict"]); _CLF.eval()
    _VIS = ck.get("label_map", {}).get("visible", 1)
    print(f"model: {CKPT_PATH.name} {ck['clip_model']}+{ck.get('arch')} acc={ck.get('test_acc',0):.1%} | device {device}")


# ── SINGLE-PASS scan: octopus p_visible + motion from one decode ──
def scan_video(url, hwaccel):
    cmd = ["ffmpeg"]
    if hwaccel: cmd += ["-hwaccel", "cuda"]
    cmd += ["-loglevel", "error", "-i", auth(url),
            "-vf", f"fps={SAMPLE_FPS},scale={DW}:{DH}", "-f", "image2pipe",
            "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsize = DW * DH * 3
    pv, motion, buf = [], [], []
    prev_gray = None

    def flush():
        if not buf: return
        with _model_lock:
            batch = torch.stack([_PP(letterbox(im)) for im in buf]).to(device)
            with torch.no_grad():
                f = _CM.encode_image(batch).float(); f = f / f.norm(dim=-1, keepdim=True)
                p = torch.softmax(_CLF(f), dim=1)[:, _VIS]
        pv.extend(p.cpu().tolist()); buf.clear()

    while True:
        raw = proc.stdout.read(fsize)
        if len(raw) < fsize: break
        arr = np.frombuffer(raw, np.uint8).reshape(DH, DW, 3)
        g = arr.astype(np.float32).mean(axis=2)                       # grayscale for motion
        if prev_gray is not None:
            d = np.abs(g - prev_gray); d[int(DH * 0.88):, int(DW * 0.60):] = 0.0   # timestamp mask
            motion.append(float((d > MOTION_PIX).mean()))
        prev_gray = g
        buf.append(Image.fromarray(arr))
        if len(buf) >= BATCH: flush()
    flush()
    proc.stdout.close(); proc.wait()
    return np.array(pv, np.float32), np.array(motion, np.float32)


def find_windows(pv, mot):
    L = int(CLIP_LEN * SAMPLE_FPS); N = len(pv)
    m = np.zeros(N, np.float32); m[1:1 + len(mot)] = mot[:max(0, N - 1)]     # align motion to pv grid
    out, s = [], 0
    while s + L <= N:
        vf = float((pv[s:s + L] >= VIS_THRESH).mean()); mm = float(m[s:s + L].mean())
        if vf > MIN_VISIBLE_FRAC and mm >= MOTION_THRESH:
            out.append({"start_sec": s, "end_sec": s + L, "visible_frac": round(vf, 3),
                        "mean_motion": round(mm, 5)}); s += L
        else: s += 1
    return out

def hhmmss(x): return f"{x//60:02d}:{x%60:02d}"

def extract_clip(url, s, e, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 10000: return True     # skip already-extracted
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(s), "-to", str(e),
                        "-i", auth(url), "-c", "copy", str(path)], capture_output=True)
    return path.exists() and path.stat().st_size > 10000


def process_video(c, hwaccel):
    """Worker: scan one video, extract its qualifying clips. Returns (c, clip_entries, n_frames)."""
    try:
        pv, mot = scan_video(c["url"], hwaccel)
    except Exception as e:
        print(f"  ! scan failed {c['date']} {c['segment']} {c['camera']}: {e}", flush=True)
        return c, [], 0
    if len(pv) == 0:
        return c, [], 0
    entries = []
    for w in find_windows(pv, mot):
        path = CLIPS_DIR / c["date"] / c["segment"] / f"{c['camera']}_{w['start_sec']:04d}-{w['end_sec']:04d}.mp4"
        if extract_clip(c["url"], w["start_sec"], w["end_sec"], path):
            entries.append({"video": c["video"], "video_url": c["url"], "date": c["date"],
                            "segment": c["segment"], "camera": c["camera"],
                            "start_sec": w["start_sec"], "end_sec": w["end_sec"],
                            "video_timeline": f"{hhmmss(w['start_sec'])}-{hhmmss(w['end_sec'])}",
                            "visible_frac": w["visible_frac"], "mean_motion": w["mean_motion"],
                            "clip_path": str(path.relative_to(HERE)),
                            "added_at": datetime.datetime.now().isoformat(timespec="seconds")})
    return c, entries, int(len(pv))


def main():
    global MOTION_THRESH, MIN_VISIBLE_FRAC, VIS_THRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--date", type=str, default=None)
    ap.add_argument("--workers", type=int, default=8, help="videos processed in parallel")
    ap.add_argument("--hwaccel", choices=["none", "cuda"], default="none", help="cuda = NVDEC GPU decode")
    ap.add_argument("--motion-thresh", type=float, default=MOTION_THRESH)
    ap.add_argument("--visible-frac", type=float, default=MIN_VISIBLE_FRAC)
    ap.add_argument("--vis-thresh", type=float, default=VIS_THRESH)
    args = ap.parse_args()
    MOTION_THRESH, MIN_VISIBLE_FRAC, VIS_THRESH = args.motion_thresh, args.visible_frac, args.vis_thresh
    hwaccel = args.hwaccel == "cuda"

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    load_model()
    proc_reg, clip_idx = init_registries()
    done = {r["video"] for r in proc_reg["processed"]}                 # SAME resume set as before

    dates = [args.date] if args.date else list_dates()
    cands = enumerate_candidates(dates, CAMERAS)
    todo = [c for c in cands if c["video"] not in done]                # skip already-processed
    if args.limit: todo = todo[:args.limit]
    print(f"{len(cands)} candidates; {len(todo)} to process | {args.workers} workers | "
          f"hwaccel={'cuda' if hwaccel else 'cpu'}\n" + "-" * 64, flush=True)
    if not todo: print("nothing to do."); return

    t0 = time.perf_counter(); n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_video, c, hwaccel): c for c in todo}
        for fut in as_completed(futs):
            c, entries, nframes = fut.result()
            with _json_lock:                                           # serialized write, same schema
                clip_idx["clips"].extend(entries)
                proc_reg["processed"].append({"video": c["video"], "date": c["date"],
                    "segment": c["segment"], "camera": c["camera"], "n_clips": len(entries),
                    "n_frames": nframes, "sources": ["extract_octopus_clips_fast"]})
                done.add(c["video"]); save_registries(proc_reg, clip_idx)
            n_done += 1
            rate = n_done / max(1e-9, time.perf_counter() - t0)
            print(f"  [{n_done}/{len(todo)}] {c['date']} {c['segment']} {c['camera']} "
                  f"-> {len(entries)} clips | {rate*60:.1f} vids/min", flush=True)

    print("-" * 64 + f"\nDONE. clips: {clip_idx['count']} | processed videos: {proc_reg['count']}")


if __name__ == "__main__":
    main()
