"""
Local octopus clip + caption pipeline (video in -> clips + captions out), optimized.

Same result as `local_video_to_captions.ipynb` / `extract_octopus_clips.py` (same gates,
same 20 s non-overlapping clips, same training-matched top-N CLAHE frames), but faster:

  A) SCAN passes run CONCURRENTLY — octopus detection (CLIP on the GPU) and motion
     detection (ffmpeg + numpy on the CPU) overlap instead of running back-to-back.
  B) CAPTIONING REUSES the scan's per-second p_visible — it no longer re-extracts dense
     frames and re-runs CLIP per clip just to pick the best frames. It picks the best-N
     seconds straight from the scan scores and only enhances those frames for the VLM.

Importable (the UI drives `process_video(..., on_stage=, on_clip=)`) and runnable as a CLI:
    python3 local_pipeline.py /path/to/video.mp4 [--camera Right_Top]
"""
import argparse, json, platform, subprocess, sys, tempfile, time, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from caption_openrouter import load_detector, enhance, N_KEEP, IMG_MAXSIDE, PRESENT_MIN

# repo root + default caption-student locations (bundled in src/ if packaged, else repo models/)
REPO = HERE.parent
BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"   # LoRA base (HF backend downloads this)

def _first_existing(cands):
    return next((p for p in cands if p.exists()), cands[-1])

# Apple-Silicon path: MLX 4-bit merged model.
DEFAULT_MLX = _first_existing([HERE / "qwen3vl2b_caption_v1_mlx_4bit",
                               REPO / "models" / "qwen3vl2b_caption_v1_mlx_4bit"])
# Cross-platform path: base Qwen3-VL-2B + this LoRA adapter (PEFT).
DEFAULT_ADAPTER = _first_existing([HERE / "qwen3vl2b_caption_v1_lora",
                                   REPO / "models" / "qwen3vl2b_caption_v1_lora"])


def _is_apple_silicon():
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def _mlx_available():
    try:
        import mlx_vlm  # noqa: F401
        return True
    except Exception:
        return False


def pick_backend():
    """'mlx' on Apple Silicon (if mlx-vlm + the MLX model are present), else 'hf'."""
    if _is_apple_silicon() and _mlx_available() and DEFAULT_MLX.exists():
        return "mlx"
    return "hf"

# ── pipeline params (defaults match extract_octopus_clips.py) ────────────────────
SAMPLE_FPS       = 1.0
CLIP_LEN         = 20
MIN_VISIBLE_FRAC = 0.50
VIS_THRESH       = 0.60
MOTION_THRESH    = 0.008
MOTION_PIX       = 25
SIZE, BATCH      = 224, 64
CAP_PROMPT = ("These frames are from one short aquarium clip of Nity, an octopus, in time order. "
              "Describe in ONE sentence what the octopus is doing.")


# ── models (loaded once, reused across videos) ──────────────────────────────────
_MODELS = None

def load_models(mlx_model_path=DEFAULT_MLX, adapter_path=DEFAULT_ADAPTER, backend=None):
    """Load the CLIP+MLP detector and the caption student once; cached module-globally.

    backend='mlx'  -> Apple-Silicon MLX 4-bit model (fast, laptop-local).
    backend='hf'   -> cross-platform: base Qwen3-VL-2B + LoRA adapter via transformers
                      (CUDA / CPU / non-Apple). Auto-picked when None.
    """
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    backend = backend or pick_backend()
    cm, pre, clf, vis, dev = load_detector()
    _MODELS = {"cm": cm, "pre": pre, "clf": clf, "vis": vis, "dev": dev, "backend": backend}
    if backend == "mlx":
        from mlx_vlm import load as mlx_load
        _MODELS["mlx"], _MODELS["proc"] = mlx_load(str(mlx_model_path))
    else:
        _MODELS.update(_load_hf_student(adapter_path))
    return _MODELS


def _load_hf_student(adapter_path):
    """Cross-platform caption student: base Qwen3-VL-2B + LoRA adapter (PEFT).
    4-bit (bitsandbytes) on CUDA, otherwise fp16/fp32 on the best available device."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
    kw = {"dtype": dtype}
    if cuda:                                      # bitsandbytes 4-bit — CUDA only
        try:
            from transformers import BitsAndBytesConfig
            kw = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
                "device_map": "auto"}
        except Exception:
            kw["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL_ID, **kw)
    ap = Path(adapter_path)
    if ap.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ap))
    else:
        print(f"WARNING: LoRA adapter not found at {ap} — running the BASE model (run download_model.sh).",
              file=sys.stderr)
    if not cuda:
        model = model.to(dev)
    model.eval()
    proc = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    return {"hf_model": model, "proc": proc, "hf_dev": dev}


# ── scan: ONE decode feeds both octopus + motion  [SPEEDUP A] ────────────────────
# The old pipeline decoded the whole video twice (once for CLIP, once for motion). Video
# decode is CPU-bound and already multi-threaded, so running the two decodes *concurrently*
# just oversubscribes the cores and is slower. Instead we decode ONCE: ffmpeg does the heavy
# downscale to a fit-224 frame, then per frame Python cheaply (a) pads it to 224² for CLIP and
# (b) stretches it to 224² grey for the motion diff — same geometry as the two original passes.
import cv2

def _probe_scaled_size(path):
    """Native W,H via ffprobe, and the fit-inside-224 size ffmpeg will emit (aspect preserved)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
                         capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in out.split("x")[:2])
    s = SIZE / max(w, h)
    return w, h, max(1, round(w * s)), max(1, round(h * s))


def scan_video(path, M, on_stage=None):
    """[A] Single decode → per-second p_visible (CLIP) AND absolute motion, in one pass."""
    if on_stage: on_stage("scanning", "single decode → octopus + motion")
    _, _, sw, sh = _probe_scaled_size(path)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path),
           "-vf", f"fps={SAMPLE_FPS},scale={sw}:{sh}",   # ffmpeg does the expensive downscale once
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fsize = sw * sh * 3
    cm, pre, clf, vis, dev = M["cm"], M["pre"], M["clf"], M["vis"], M["dev"]
    y0, x0 = (SIZE - sh) // 2, (SIZE - sw) // 2
    mrow, mcol = int(SIZE * 0.88), int(SIZE * 0.60)     # burned-in timestamp mask (as in scan_motion_area)
    pv, motion, buf = [], [], []
    prev_g = None

    def flush():
        if not buf: return
        batch = torch.stack([pre(im) for im in buf]).to(dev)
        with torch.no_grad():
            f = cm.encode_image(batch).float(); f = f / f.norm(dim=-1, keepdim=True)
            p = torch.softmax(clf(f), dim=1)[:, vis]
        pv.extend(p.cpu().tolist()); buf.clear()

    while True:
        raw = proc.stdout.read(fsize)
        if len(raw) < fsize: break
        arr = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
        # (a) octopus: pad the fit-224 frame to 224² (== letterbox, no extra resize)
        cv_img = np.full((SIZE, SIZE, 3), 128, np.uint8); cv_img[y0:y0+sh, x0:x0+sw] = arr
        buf.append(Image.fromarray(cv_img))
        # (b) motion: stretch to 224² grey, absolute changed-pixel fraction with timestamp masked
        g = cv2.cvtColor(cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev_g is None:
            motion.append(0.0)
        else:
            diff = np.abs(g - prev_g); diff[mrow:, mcol:] = 0.0
            motion.append(float((diff > MOTION_PIX).mean()))
        prev_g = g
        if len(buf) >= BATCH: flush()
    flush(); proc.stdout.close(); proc.wait()
    return np.array(pv, np.float32), np.array(motion, np.float32)


# ── windows + extraction ────────────────────────────────────────────────────────

def find_windows(pv, motion):
    L = int(CLIP_LEN * SAMPLE_FPS); N = len(pv); out = []; s = 0
    while s + L <= N:
        wp, wm = pv[s:s + L], motion[s:s + L]
        vf = float((wp >= VIS_THRESH).mean()); mm = float(wm.mean())
        if vf > MIN_VISIBLE_FRAC and mm >= MOTION_THRESH:
            out.append({"start": int(s / SAMPLE_FPS), "end": int((s + L) / SAMPLE_FPS),
                        "visible_frac": round(vf, 3), "mean_motion": round(mm, 5)})
            s += L                        # non-overlapping
        else:
            s += 1
    return out


def extract_clip(path, start, end, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10000:
        return True
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-to", str(end),
                        "-i", str(path), "-c:v", "copy", "-c:a", "aac", str(out_path)],
                       capture_output=True, text=True)
    return r.returncode == 0 and out_path.exists()


# ── caption: reuse scan scores to pick frames  [SPEEDUP B] ───────────────────────

def _extract_frames_at_768(clip_path, tmp):
    """One ffmpeg decode of the (short) clip -> 1 fps frames at <=768px. No CLIP here."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_path),
                    "-vf", f"fps={SAMPLE_FPS},scale='min({IMG_MAXSIDE},iw)':-2", "-q:v", "3",
                    f"{tmp}/f_%03d.jpg"], capture_output=True)
    return sorted(str(p) for p in Path(tmp).glob("f_*.jpg"))


def caption_window(video_path, start, pv, M):
    """[B, no-clip variant] Caption a window WITHOUT keeping a clip file: byte-copy the 20 s window
    to a temp mp4 (robust — same cut the save_clips path uses; deep per-frame seeks into the full
    video fail near truncated stream ends), caption it, then discard the temp clip."""
    with tempfile.TemporaryDirectory() as tmp:
        cp = Path(tmp) / "w.mp4"
        if not extract_clip(video_path, start, start + CLIP_LEN, cp):
            return {"caption": None, "status": "noframes"}
        return caption_clip(cp, start, pv, M)


def caption_clip(clip_path, start, pv, M):
    """[B] Pick the best-N seconds from the whole-video scan scores (no per-clip CLIP re-run),
    CLAHE-enhance just those frames, and caption with the student (MLX or HF backend)."""
    win = pv[start:start + CLIP_LEN]
    maxp = float(win.max()) if len(win) else 0.0
    if maxp < PRESENT_MIN:                                   # presence gate (skip the VLM)
        return {"caption": "octopus not present", "max_p_visible": round(maxp, 3), "status": "absent"}
    with tempfile.TemporaryDirectory() as tmp:
        frames = _extract_frames_at_768(clip_path, tmp)
        if not frames:
            return {"caption": None, "status": "noframes"}
        n = min(len(frames), len(win))
        scores = win[:n]
        order = sorted(range(n), key=lambda k: scores[k], reverse=True)[:N_KEEP]
        best = [frames[k] for k in sorted(order)]            # chronological
        prepped = []
        for j, f in enumerate(best):                         # CLAHE == training / teacher input
            im = Image.open(f).convert("RGB"); im.thumbnail((IMG_MAXSIDE, IMG_MAXSIDE)); im = enhance(im)
            outp = f"{tmp}/best_{j:02d}.jpg"; im.save(outp, quality=90); prepped.append(outp)
        cap = (_caption_mlx if M.get("backend") == "mlx" else _caption_hf)(prepped, M)
        return {"caption": cap, "max_p_visible": round(maxp, 3), "status": "captioned"}


def _caption_mlx(image_paths, M):
    """Apple-Silicon MLX generation."""
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    fmt = apply_chat_template(M["proc"], M["mlx"].config, CAP_PROMPT, num_images=len(image_paths))
    out = mlx_generate(M["mlx"], M["proc"], fmt, image_paths, max_tokens=80, temperature=0.0, verbose=False)
    return (out.text if hasattr(out, "text") else out).strip()


def _caption_hf(image_paths, M):
    """Cross-platform transformers generation (base Qwen3-VL-2B + LoRA)."""
    model, proc, dev = M["hf_model"], M["proc"], M["hf_dev"]
    content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
    content.append({"type": "text", "text": CAP_PROMPT})
    messages = [{"role": "user", "content": content}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ── orchestration ────────────────────────────────────────────────────────────────

def process_video(video_path, out_dir, M=None, camera="cam", on_stage=None, on_clip=None,
                  save_clips=True):
    """Full pipeline. Calls on_stage(stage, detail) and on_clip(i, total, record) for the UI.
    save_clips=False captions straight from the full video and writes no clip mp4s (the demo UI
    seeks the full video, so per-clip files aren't needed)."""
    M = M or load_models()
    video_path = str(video_path); out_dir = Path(out_dir)
    stem = Path(video_path).stem
    clips_dir = out_dir / "clips"
    if save_clips: clips_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pv, motion = scan_video(video_path, M, on_stage)
    if on_stage: on_stage("scanned", f"{len(pv)}s | mean p_visible {pv.mean():.2f} | "
                                      f"motion {motion.mean():.4f} | {time.time()-t0:.0f}s")

    windows = find_windows(pv, motion)
    if on_stage: on_stage("windows", f"{len(windows)} clips pass both gates")

    recs = []
    for i, w in enumerate(windows, 1):
        rec = {**w, "video_timeline": f"{w['start']//60:02d}:{w['start']%60:02d}-"
                                      f"{w['end']//60:02d}:{w['end']%60:02d}"}
        if save_clips:
            cp = clips_dir / f"{camera}_{stem}_{w['start']:04d}-{w['end']:04d}.mp4"
            if not extract_clip(video_path, w["start"], w["end"], cp):
                continue
            rec["clip_path"] = str(cp); rec["clip_name"] = cp.name
            rec.update(caption_clip(cp, w["start"], pv, M))
        else:
            rec.update(caption_window(video_path, w["start"], pv, M))
        recs.append(rec)
        if on_clip: on_clip(i, len(windows), rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"video": video_path, "camera": camera,
              "processed_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "caption_model": "qwen3vl2b_caption_v1_mlx_4bit" if M.get("backend") == "mlx"
                               else "qwen3vl2b_caption_v1_lora",
              "caption_backend": M.get("backend"),
              "elapsed_sec": round(time.time() - t0, 1),
              "params": {"clip_len": CLIP_LEN, "vis_thresh": VIS_THRESH,
                         "min_visible_frac": MIN_VISIBLE_FRAC, "motion_thresh": MOTION_THRESH},
              "n_clips": len(recs), "clips": recs}
    out_json = out_dir / f"{stem}_captions.json"
    json.dump(result, open(out_json, "w"), indent=2)
    if on_stage: on_stage("done", f"{len(recs)} clips in {result['elapsed_sec']:.0f}s -> {out_json.name}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--out", default=str(REPO / "local_pipeline_out"))
    ap.add_argument("--mlx", default=str(DEFAULT_MLX), help="Apple-Silicon MLX 4-bit model dir")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="LoRA adapter dir (HF backend)")
    ap.add_argument("--backend", choices=["mlx", "hf"], default=None,
                    help="force caption backend (default: auto — mlx on Apple Silicon, else hf)")
    args = ap.parse_args()
    M = load_models(args.mlx, args.adapter, backend=args.backend)
    print(f"caption backend: {M['backend']}", flush=True)
    def stage(s, d): print(f"[{s}] {d}", flush=True)
    def clip(i, n, r): print(f"  [{i}/{n}] {r['video_timeline']} ({r['status']}) {r['caption']}", flush=True)
    res = process_video(args.video, args.out, M, camera=args.camera, on_stage=stage, on_clip=clip)
    print(f"\nDONE: {res['n_clips']} clips in {res['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
