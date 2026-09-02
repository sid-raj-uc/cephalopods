"""
Diversity dedup — drop near-duplicate clips using CLIP embeddings.

Per clip: mean-pooled CLIP embedding over a few frames. Greedy keep: a clip
survives only if its cosine similarity to every already-kept clip is below the
threshold. Reports survivor counts:
  - within-video (dedup inside each source video/segment/camera) — kills adjacency dups
  - global (dedup across the whole set) — also kills cross-video repeats

Embeddings are cached to src/clip_embeddings.npz. Non-destructive: only reports
(and can optionally tag the index with keep/dup_of — off by default).

Run: python3 dedup_clips.py                 # report survivors at a few thresholds
     python3 dedup_clips.py --write 0.93     # also write keep/dup_of into the index at 0.93
"""
import argparse, json, subprocess, tempfile, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

HERE    = Path(__file__).resolve().parent
PROJECT = HERE.parent
INDEX   = HERE / "octopus_clips_verified.json"
CKPT    = HERE / "clip_mlp_hardneg_v2.pt"
CACHE   = HERE / "clip_embeddings.npz"
N_FRAMES = 4
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def letterbox(img, size=224, fill=(128, 128, 128)):
    w, h = img.size; s = size / max(w, h); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    cv = Image.new("RGB", (size, size), fill); cv.paste(img, ((size - nw) // 2, (size - nh) // 2)); return cv

def resolve(cp: str) -> Path:
    if cp.startswith("data/"):
        return PROJECT / cp                    # old clips live in repo data/
    return HERE / cp                           # new clips live under src/


def _prep_frames(clip_path, pre):
    """ffmpeg extract + load N_FRAMES preprocessed CPU tensors. Runs in worker THREADS
    (ffmpeg + PIL + preprocess are CPU/IO — safe off-main-thread). NO GPU/MPS here."""
    with tempfile.TemporaryDirectory() as t:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(resolve(clip_path)),
                        "-vf", "fps=1", "-q:v", "3", f"{t}/f_%03d.jpg"], capture_output=True)
        fs = sorted(Path(t).glob("f_*.jpg"))
        if not fs:
            return None
        idx = np.linspace(0, len(fs) - 1, min(N_FRAMES, len(fs))).round().astype(int)
        return torch.stack([pre(letterbox(Image.open(fs[k]).convert("RGB"))) for k in idx])


def embed_all(clips, cm, pre, workers=8):
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {p: v for p, v in zip(list(z["paths"]), z["embs"])}
    todo = [c for c in clips if c["clip_path"] not in cache and resolve(c["clip_path"]).exists()]
    print(f"embedding {len(todo)} clips ({len(cache)} cached) | {workers} prep-workers ...", flush=True)

    # threads do ffmpeg/image prep in parallel (ordered stream); MPS encode stays on the MAIN thread
    # (PyTorch MPS hangs if encode is called from a worker thread).
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for c, ims in zip(todo, ex.map(lambda cc: _prep_frames(cc["clip_path"], pre), todo)):
            done += 1
            if ims is not None:
                with torch.no_grad():                  # encode on main thread
                    f = cm.encode_image(ims.to(device)).float(); f = f / f.norm(dim=-1, keepdim=True)
                    v = f.mean(0); v = (v / v.norm()).cpu().numpy()
                cache[c["clip_path"]] = v
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)
                np.savez(CACHE, paths=list(cache), embs=np.array(list(cache.values()), np.float32))
    np.savez(CACHE, paths=list(cache), embs=np.array(list(cache.values()), np.float32))
    return cache


def greedy_keep(items, emb, thresh):
    """items: list of (clip_path, sort_key). Keep if cos-sim to all kept < thresh."""
    kept, kept_vecs = [], []
    for cp, _ in sorted(items, key=lambda x: x[1]):
        v = emb[cp]
        if not kept_vecs or max(float(np.dot(v, kv)) for kv in kept_vecs) < thresh:
            kept.append(cp); kept_vecs.append(v)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=float, default=None, help="write keep/dup_of into the index at this threshold (within-video)")
    ap.add_argument("--out", type=str, default=None, help="write dedup results to this SEPARATE json (index untouched)")
    ap.add_argument("--out-thresh", type=float, default=0.93, help="within-video threshold for per-clip keep in --out")
    ap.add_argument("--workers", type=int, default=8, help="parallel ffmpeg frame-extraction workers")
    args = ap.parse_args()

    try:
        import pkg_resources, packaging, packaging.version, packaging.specifiers, packaging.requirements
        pkg_resources.packaging = packaging
    except Exception: pass
    import clip as clip_lib
    ck = torch.load(CKPT, map_location=device); cm, pre = clip_lib.load(ck["clip_model"], device=device); cm.eval()

    index = json.load(open(INDEX)); clips = index["clips"]
    have_file = [c for c in clips if resolve(c["clip_path"]).exists()]
    print(f"index clips: {len(clips)} | files present: {len(have_file)} | device {device}")
    emb = embed_all(have_file, cm, pre, workers=args.workers)
    clips_e = [c for c in have_file if c["clip_path"] in emb]
    print(f"embedded: {len(clips_e)}\n" + "-" * 60)

    groups = defaultdict(list)
    for c in clips_e:
        groups[(c["date"], c["segment"], c["camera"])].append((c["clip_path"], c.get("start_sec", 0)))

    print(f"{'thresh':>7} | {'within-video survivors':>22} | {'global survivors':>16}")
    survivors = {}
    for T in [0.90, 0.93, 0.95, 0.97]:
        wv = sum(len(greedy_keep(v, emb, T)) for v in groups.values())
        gl = len(greedy_keep([(c["clip_path"], c.get("start_sec", 0)) for c in clips_e], emb, T))
        survivors[f"{T:.2f}"] = {"within_video": wv, "global": gl}
        print(f"{T:7.2f} | {wv:22d} | {gl:16d}")
    print(f"\n(total embedded clips: {len(clips_e)} | source videos: {len(groups)})")

    if args.out is not None:
        import datetime
        T = args.out_thresh
        keep = set()
        for v in groups.values():
            keep.update(greedy_keep(v, emb, T))
        results = {
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "index": INDEX.name, "n_embedded": len(clips_e), "n_videos": len(groups),
            "survivors_by_threshold": survivors,
            "keep_thresh_within_video": T, "n_keep": len(keep),
            "clips": [{"clip_path": c["clip_path"], "date": c["date"], "segment": c["segment"],
                       "camera": c["camera"], "start_sec": c.get("start_sec", 0),
                       "keep": c["clip_path"] in keep} for c in clips_e],
        }
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nwrote dedup results -> {args.out}  ({len(keep)} kept / {len(clips_e)} at within-video {T})")

    if args.write is not None:
        T = args.write
        keep_set = set()
        for v in groups.values():
            keep_set.update(greedy_keep(v, emb, T))
        for c in clips:
            if c["clip_path"] in emb:
                c["keep"] = c["clip_path"] in keep_set
        json.dump(index, open(INDEX, "w"), indent=2)
        print(f"\nwrote keep flag at within-video thresh {T}: {len(keep_set)} kept / {len(clips_e)}")


if __name__ == "__main__":
    main()
