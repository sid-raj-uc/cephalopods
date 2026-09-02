"""
Exp 26 — Stream-scan UNUSED Right-camera footage from the remote repo, gate on
motion, and classify each surviving frame hidden/visible with the CLIP+MLP model.

No full download: ffmpeg pulls only the sampled frames over HTTP range requests.

Pipeline (per video):
  1. Enumerate Right-camera videos on the server (all dates).
  2. Skip anything already in data/processed_videos.json.
  3. MOTION pass: scan_motion_area() -> per-second absolute changed-pixel fraction
     (timestamp-masked). This is the new, non-normalized motion method.
  4. OCTOPUS pass: stream 1 frame / 2s (fps=0.5), annotate each frame with its
     motion fraction. With the gate ON (default), frames below --motion-thresh are
     marked "static" and are NOT classified or saved; moving frames are classified
     with weights/clip_mlp_best.pt and saved to data/scanned_frames/{visible,hidden}/
     (p_visible in the filename so you can re-threshold later).
  5. Write per-frame predictions + a per-video summary to data/scan_results/.
  6. Append each finished video to data/processed_videos.json (so it's never reused).

Resumable: a video already in the registry is skipped. Safe to Ctrl-C.

Usage:
  python3 phase2/exp26_remote_scan.py --limit 1            # smoke test, 1 video
  python3 phase2/exp26_remote_scan.py --limit 50           # batch of 50
  python3 phase2/exp26_remote_scan.py --date 2026-02-21    # one day, all Right cams
  python3 phase2/exp26_remote_scan.py --no-gate            # classify every frame
  python3 phase2/exp26_remote_scan.py --motion-thresh 0.01 # stricter motion gate
"""
import argparse, datetime, json, re, subprocess, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from motion_detector import scan_motion_area

PROJECT   = Path(__file__).resolve().parents[2]   # repo root (file is phase2/octo-clip-extraction/)
REGISTRY  = PROJECT / "data" / "processed_videos.json"
# Octopus model: clip_mlp_best.pt — letterbox + 66 verified hard negs, the current default.
CKPT_PATH = PROJECT / "weights" / "clip_mlp_best.pt"
OUT_DIR   = PROJECT / "data" / "scan_results"

BASE      = "https://repo.octopus-intelligence.org/public/O-vulgaris-Nity-2026-2-20--"
import sys
sys.path.insert(0, str(PROJECT))
from server_creds import USER, PASS as PWD   # creds from env / .env, not hardcoded
RIGHT_CAMS = ["Right Back", "Right Front", "Right Left", "Right Right", "Right Top"]

FPS        = 0.5    # 1 frame / 2s
SIZE       = 224
BATCH_SIZE = 64
ABSENT_MAX, PRESENT_MIN = 0.35, 0.65
VIS_THRESH = 0.5    # p_visible >= this -> "visible", else "hidden"
FRAMES_DIR = PROJECT / "data" / "scanned_frames"   # <FRAMES_DIR>/{visible,hidden}/

# Motion gate (scan_motion_area: absolute changed-pixel fraction, timestamp-masked).
# A frame is "moving" if its changed-pixel fraction >= MOTION_THRESH. When gating is
# on, static frames are NOT sent to the octopus classifier or saved — only annotated.
MOTION_PIX    = 25      # per-pixel grey-level change counted as "moved" (matches exp30)
MOTION_THRESH = 0.005   # 0.5% of frame changed (matches exp30 survive bar)


# ── server enumeration ──────────────────────────────────────────────────────────

def _curl(url: str) -> str:
    return subprocess.run(["curl", "-s", "--user", f"{USER}:{PWD}", url],
                          capture_output=True, text=True).stdout

def list_dates() -> list[str]:
    out = _curl(f"{BASE}/Right%20Top/Local/")
    return sorted(set(re.findall(r'href="(\d{4}-\d{2}-\d{2})/"', out)))

def list_segments(cam: str, date: str) -> list[dict]:
    cam_enc = urllib.parse.quote(cam)
    out = _curl(f"{BASE}/{cam_enc}/Local/{date}/")
    rows = []
    for f in re.findall(r'href="([^"]+\.mp4)"', out):
        m = re.match(r"(\d+)--", f)
        if not m:
            continue
        seg = m.group(1)
        cam_us = cam.replace(" ", "_")
        rows.append({
            "video":   f"data/aquarium/full/{date}/{seg}/{cam_us}.mp4",
            "date":    date, "segment": seg, "camera": cam_us,
            "url":     f"{BASE}/{cam_enc}/Local/{date}/{f}",
        })
    return rows

def enumerate_candidates(dates: list[str]) -> list[dict]:
    tasks = [(cam, d) for d in dates for cam in RIGHT_CAMS]
    out = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(lambda a: list_segments(*a), tasks):
            out.extend(r)
    return out


# ── registry ──────────────────────────────────────────────────────────────────

def load_registry() -> dict:
    return json.load(open(REGISTRY))

def registry_video_set(reg: dict) -> set[str]:
    return {r["video"] for r in reg["processed"]}

def append_to_registry(reg: dict, rec: dict):
    reg["processed"].append(rec)
    reg["count"] = len(reg["processed"])
    reg["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    reg["processed"].sort(key=lambda r: r["video"])
    json.dump(reg, open(REGISTRY, "w"), indent=2)


# ── model (mirrors exp24) ───────────────────────────────────────────────────────

def build_classifier(ckpt: dict) -> nn.Module:
    feat_dim = ckpt["feat_dim"]
    arch = ckpt.get("arch", "linear")
    if arch == "linear":
        return nn.Linear(feat_dim, 2)
    hidden = [int(x) for x in arch.replace("mlp_", "").split("_")]
    dims = [feat_dim] + hidden + [2]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(0.3)]
    return nn.Sequential(*layers)

def load_model(device):
    try:
        import pkg_resources
        import packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception:
        pass
    import clip as clip_lib
    ckpt = torch.load(CKPT_PATH, map_location=device)
    clip_model, preprocess = clip_lib.load(ckpt["clip_model"], device=device)
    clip_model.eval()
    classifier = build_classifier(ckpt).to(device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()
    vis_idx = ckpt.get("label_map", {}).get("visible", 1)
    print(f"Model: CLIP {ckpt['clip_model']} + {ckpt.get('arch')}  "
          f"(test_acc {ckpt.get('test_acc', 0):.1%}, visible=idx{vis_idx})")
    return clip_model, preprocess, classifier, vis_idx


# ── streaming scan ──────────────────────────────────────────────────────────────

def auth(url: str) -> str:
    return url.replace("https://", f"https://{USER}:{PWD}@")


def motion_at(ts_m: np.ndarray, scores: np.ndarray, t: float):
    """Nearest motion fraction to time t (None if no motion data)."""
    if len(ts_m) == 0:
        return None
    i = int(np.searchsorted(ts_m, t))
    if i <= 0:
        return float(scores[0])
    if i >= len(ts_m):
        return float(scores[-1])
    return float(scores[i - 1]) if abs(ts_m[i - 1] - t) <= abs(ts_m[i] - t) else float(scores[i])


def stream_frames(url: str):
    """Yield (t_sec, PIL.Image) sampled at FPS via ffmpeg HTTP range requests."""
    auth_url = auth(url)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", auth_url,
           "-vf", (f"fps={FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,"
                   f"pad={SIZE}:{SIZE}:-1:-1:color=gray"),  # letterbox (no crop) — matches training
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fsize = SIZE * SIZE * 3
    t, interval = 0.0, 1.0 / FPS
    while True:
        raw = proc.stdout.read(fsize)
        if len(raw) < fsize:
            break
        yield t, Image.fromarray(np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3))
        t += interval
    proc.stdout.close()
    proc.wait()

def classify_video(url, clip_model, preprocess, classifier, vis_idx, device,
                   save_dir: Path = None, name_prefix: str = "",
                   motion_ts=None, motion_scores=None, gate: bool = True):
    """Stream + classify, with the scan_motion_area motion gate.

    Every sampled frame is annotated with its motion fraction. When `gate` is on
    (and motion data is available), frames below MOTION_THRESH are treated as
    static: they are NOT run through the octopus classifier and NOT saved — just
    recorded as pred="static". Moving frames are classified and saved by label.

    Returns (records, n_saved, n_static) where records is a list of per-frame dicts.
    """
    records, buf_t, buf_im = [], [], []
    n_saved = 0
    n_static = 0
    have_motion = gate and motion_ts is not None and len(motion_ts) > 0

    def flush():
        nonlocal n_saved
        if not buf_im:
            return
        batch = torch.stack([preprocess(im) for im in buf_im]).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            p = torch.softmax(classifier(feats), dim=1)[:, vis_idx]
        p = p.cpu().tolist()
        for tt, im, pv in zip(buf_t, buf_im, p):
            mv = motion_at(motion_ts, motion_scores, tt) if motion_ts is not None else None
            label = "visible" if pv >= VIS_THRESH else "hidden"
            if save_dir is not None:
                im.save(save_dir / label / f"p{pv:.2f}_{name_prefix}_t{int(tt):04d}.jpg",
                        quality=90)
                n_saved += 1
            records.append({"t_sec": int(tt), "p_visible": round(float(pv), 4),
                            "pred": label, "band": band(float(pv)),
                            "motion": None if mv is None else round(mv, 5)})
        buf_t.clear(); buf_im.clear()

    for t, im in stream_frames(url):
        mv = motion_at(motion_ts, motion_scores, t) if motion_ts is not None else None
        if have_motion and mv is not None and mv < MOTION_THRESH:
            # static frame — gate it out before the (expensive) classifier
            n_static += 1
            records.append({"t_sec": int(t), "p_visible": None, "pred": "static",
                            "band": "static", "motion": round(mv, 5)})
            continue
        buf_t.append(t); buf_im.append(im)
        if len(buf_im) >= BATCH_SIZE:
            flush()
    flush()
    records.sort(key=lambda r: r["t_sec"])
    return records, n_saved, n_static

def band(p: float) -> str:
    if p < ABSENT_MAX:  return "absent"
    if p > PRESENT_MIN: return "present"
    return "uncertain"


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    global FPS, MOTION_THRESH, MOTION_PIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max number of videos to scan")
    ap.add_argument("--date", type=str, default=None, help="restrict to a single date YYYY-MM-DD")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--motion-thresh", type=float, default=MOTION_THRESH,
                    help="changed-pixel fraction >= this counts as motion (default %(default)s)")
    ap.add_argument("--motion-pix", type=int, default=MOTION_PIX,
                    help="per-pixel grey-level change counted as 'moved' (default %(default)s)")
    ap.add_argument("--no-gate", action="store_true",
                    help="disable the motion gate (classify every frame, still annotate motion)")
    args = ap.parse_args()
    FPS = args.fps
    MOTION_THRESH = args.motion_thresh
    MOTION_PIX = args.motion_pix
    gate = not args.no_gate
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (FRAMES_DIR / "visible").mkdir(parents=True, exist_ok=True)
    (FRAMES_DIR / "hidden").mkdir(parents=True, exist_ok=True)

    reg = load_registry()
    done = registry_video_set(reg)

    dates = [args.date] if args.date else list_dates()
    cands = enumerate_candidates(dates)
    todo = [c for c in cands if c["video"] not in done]
    print(f"{len(cands)} Right-camera videos on server for {len(dates)} date(s); "
          f"{len(todo)} unprocessed")
    if args.limit:
        todo = todo[:args.limit]
        print(f"  limited to first {len(todo)}")
    if not todo:
        print("Nothing to scan."); return

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess, classifier, vis_idx = load_model(device)
    print(f"Device: {device}  |  sampling {FPS} fps (1 frame/{int(1/FPS)}s)")
    print(f"Motion gate: {'ON' if gate else 'OFF'}  "
          f"(thresh={MOTION_THRESH}, pix={MOTION_PIX})\n" + "-" * 64)

    HEADER = ("video,date,segment,camera,n_frames,n_classified,n_static,frac_static,"
              "n_visible,n_hidden,frac_visible,n_present,n_uncertain,n_absent,"
              "motion_mean,n_saved,scanned_at\n")
    summary_path = OUT_DIR / "scan_summary.csv"
    if summary_path.exists() and summary_path.open().readline() != HEADER:
        legacy = OUT_DIR / "scan_summary_pre_motion.csv"
        summary_path.rename(legacy)
        print(f"  rotated old summary -> {legacy.name} (schema changed)")
    new_summary = not summary_path.exists()
    summ = open(summary_path, "a")
    if new_summary:
        summ.write(HEADER)

    import time
    for i, c in enumerate(todo, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(todo)}] {c['date']} {c['segment']} {c['camera']}", flush=True)
        name_prefix = f"{c['date']}_{c['segment']}_{c['camera']}"

        # 1) motion pass — absolute changed-pixel fraction (scan_motion_area).
        motion_ts = motion_scores = None
        if gate:
            try:
                motion_ts, motion_scores = scan_motion_area(
                    auth(c["url"]), fps=FPS, pix_thresh=MOTION_PIX)
            except Exception as e:
                print(f"   ! motion scan failed ({e}); classifying all frames")
                motion_ts = motion_scores = None

        # 2) octopus pass — gated + annotated by motion.
        try:
            records, n_saved, n_static = classify_video(
                c["url"], clip_model, preprocess, classifier, vis_idx, device,
                save_dir=FRAMES_DIR, name_prefix=name_prefix,
                motion_ts=motion_ts, motion_scores=motion_scores, gate=gate)
        except Exception as e:
            print(f"   ! scan failed: {e}"); continue
        if not records:
            print("   ! no frames; skipping"); continue

        n_frames = len(records)
        classified = [r for r in records if r["pred"] != "static"]
        n_clf = len(classified)
        n_vis = sum(1 for r in classified if r["pred"] == "visible")
        n_pres = sum(1 for r in classified if r["band"] == "present")
        n_unc  = sum(1 for r in classified if r["band"] == "uncertain")
        n_abs  = sum(1 for r in classified if r["band"] == "absent")
        frac_vis = (n_vis / n_clf) if n_clf else 0.0
        motion_vals = [r["motion"] for r in records if r["motion"] is not None]
        motion_mean = float(np.mean(motion_vals)) if motion_vals else 0.0

        rec_out = {
            "video": c["video"], "date": c["date"], "segment": c["segment"],
            "camera": c["camera"], "url": c["url"], "fps": FPS,
            "model": CKPT_PATH.name, "motion_gate": gate,
            "motion_thresh": MOTION_THRESH, "motion_pix": MOTION_PIX,
            "n_frames": n_frames, "n_classified": n_clf, "n_static": n_static,
            "frames": records,
        }
        out_json = OUT_DIR / f"{c['date']}_{c['segment']}_{c['camera']}.json"
        json.dump(rec_out, open(out_json, "w"), indent=1)

        summ.write(f"{c['video']},{c['date']},{c['segment']},{c['camera']},"
                   f"{n_frames},{n_clf},{n_static},{n_static/n_frames:.4f},"
                   f"{n_vis},{n_clf-n_vis},{frac_vis:.4f},"
                   f"{n_pres},{n_unc},{n_abs},"
                   f"{motion_mean:.5f},{n_saved},"
                   f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
        summ.flush()

        append_to_registry(reg, {
            "video": c["video"], "date": c["date"], "segment": c["segment"],
            "camera": c["camera"], "sources": ["exp26_remote_scan"],
        })
        done.add(c["video"])
        dt = time.perf_counter() - t0
        print(f"   {n_frames} frames | static {n_static} gated | classified {n_clf} "
              f"-> {n_vis} visible / {n_clf-n_vis} hidden | uncertain {n_unc} "
              f"| saved {n_saved} | {dt:.1f}s", flush=True)

    summ.close()
    print("-" * 64)
    print(f"Done. Per-video JSON + scan_summary.csv in {OUT_DIR.relative_to(PROJECT)}/")
    print(f"Saved frames split into {FRAMES_DIR.relative_to(PROJECT)}/{{visible,hidden}}/")
    print(f"Registry now has {reg['count']} processed videos.")


if __name__ == "__main__":
    main()
