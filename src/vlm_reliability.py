#!/usr/bin/env python3
"""vlm_reliability.py — VLM-250 benchmark: is the structured behavioural extractor self-consistent?

Every headline behavioural result (activity budget, circadian profile, human-presence stimulus
response, kinematics x behaviour) is grouped by labels this extractor produced, and those labels
have never been validated. This runs the extractor a SECOND time on the same clips using a
DISJOINT set of input frames (ranks N_KEEP..2*N_KEEP by detector score instead of the top N_KEEP).

What that measures, precisely: **frame-sampling sensitivity** — does the label survive being shown
a different set of clear frames from the same clip. It is a CONSISTENCY measure. Consistency bounds
accuracy but does not establish it; accuracy is assessed separately against human labels
(src/vlm_reliability_stats.py).

Safety: reads data/behaviour_records.json, never writes it (a backup is taken anyway); all output
goes to data/behaviour_records_retest.json. Resumable.

Usage:
  venv/bin/python3 src/vlm_reliability.py --sample          # build+freeze data/vlm_250.json
  venv/bin/python3 src/vlm_reliability.py --run [--limit N] # run condition B
"""
import argparse, collections, glob, json, os, random, shutil, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import caption_openrouter as C
import extract_behaviour_records as E

SRC = ROOT / "data" / "behaviour_records.json"
OUT = ROOT / "data" / "behaviour_records_retest.json"
SAMPLE = ROOT / "data" / "vlm_250.json"
WORKERS = 2                      # referee: keep local ffmpeg/CLIP contention low
io_lock, clip_lock = threading.Lock(), threading.Lock()


def source_video(rel):
    p = Path(rel); return f"{p.parent.parent.name}/{p.parent.name}"


def build_sample(n_total=250, per_video=2, seed=11):
    """Stratified by behaviour; <=per_video per source video; Right_Left is its OWN stratum
    (excluded elsewhere by pipeline convention, so it must not silently dominate); `uncertain`
    and not-present are INCLUDED so abstention is measured too."""
    recs = json.load(open(SRC))
    rows = []
    for rel, r in recs.items():
        if not (ROOT / "src" / rel).exists():
            continue
        st = r.get("struct") or {}
        beh = st.get("behavior", "uncertain")
        if not st.get("present"):
            beh = "__not_present__"
        rows.append({"clip": rel, "video": source_video(rel), "camera": r.get("camera", "?"),
                     "stratum": ("RL/" if "Right_Left" in rel else "") + beh})
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    strata = sorted(by, key=lambda s: -len(by[s]))
    quota = max(1, n_total // max(1, len(strata)))
    chosen, per_vid = [], collections.Counter()
    for s in strata:
        pool = by[s][:]
        rng.shuffle(pool)
        took = 0
        for r in pool:
            if took >= quota or len(chosen) >= n_total:
                break
            if per_vid[r["video"]] >= per_video:
                continue
            chosen.append(r); per_vid[r["video"]] += 1; took += 1
    # top up to n_total respecting the per-video cap
    if len(chosen) < n_total:
        rest = [r for r in rows if r not in chosen]
        rng.shuffle(rest)
        for r in rest:
            if len(chosen) >= n_total:
                break
            if per_vid[r["video"]] < per_video:
                chosen.append(r); per_vid[r["video"]] += 1
    stats = collections.Counter(r["stratum"] for r in chosen)
    json.dump({"seed": seed, "per_video": per_video, "n": len(chosen),
               "n_videos": len({r["video"] for r in chosen}),
               "strata": dict(sorted(stats.items())), "clips": [r["clip"] for r in chosen]},
              open(SAMPLE, "w"), indent=1)
    print(f"VLM-250: {len(chosen)} clips / {len({r['video'] for r in chosen})} videos -> {SAMPLE}")
    for k, v in sorted(stats.items()):
        print(f"  {k:42s} {v}")


def run(limit=0):
    if not SAMPLE.exists():
        sys.exit("no sample — run --sample first")
    if SRC.exists():                                   # referee: protect the $2.22 corpus
        bak = SRC.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copy(SRC, bak); print(f"backed up corpus -> {bak}")
    clips = json.load(open(SAMPLE))["clips"]
    done = json.load(open(OUT)) if OUT.exists() else {}
    todo = [c for c in clips if c not in done]
    if limit:
        todo = todo[:limit]
    print(f"condition B (disjoint frames): {len(todo)} to run, {len(done)} done", flush=True)
    cm, pre, clf, vis, dev = C.load_detector(); print("detector loaded", flush=True)
    state = {"done": 0, "err": 0, "cost": 0.0}

    def work(rel):
        p = str(ROOT / "src" / rel); tmp = tempfile.mkdtemp()
        try:
            grey = E.greyscale(p)
            frames = C.extract_frames(p, tmp)
            if not frames:
                raise RuntimeError("no frames")
            with clip_lock:
                sc = C.score(frames, cm, pre, clf, vis, dev)
            ranked = sorted(range(len(frames)), key=lambda k: sc[k], reverse=True)
            # DISJOINT frame set: skip the top N_KEEP that condition A used
            order = ranked[C.N_KEEP:2 * C.N_KEEP] or ranked[:C.N_KEEP]
            order = sorted(order)
            urls = [C.b64_image(frames[k]) for k in order]
            txt, u = E.call(urls, E.prompt(not grey))
            return rel, {"struct": E.validate(E.pjson(txt)), "grey": grey,
                         "n_frames_used": len(order), "cost": float(u.get("cost", 0) or 0)}, None
        except Exception as ex:
            return rel, None, str(ex)[:120]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(work, c) for c in todo]):
            rel, rec, err = fut.result()
            with io_lock:
                if rec:
                    done[rel] = rec; state["done"] += 1; state["cost"] += rec["cost"]
                else:
                    state["err"] += 1
                n = state["done"] + state["err"]
                if n % 10 == 0 or n == len(todo):
                    json.dump(done, open(OUT, "w"), indent=1)
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)}  ok={state['done']} err={state['err']} "
                          f"${state['cost']:.3f}  {el/60:.1f}min", flush=True)
    json.dump(done, open(OUT, "w"), indent=1)
    print(f"done: {state['done']} ok / {state['err']} err / ${state['cost']:.3f} -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.sample:
        build_sample(a.n)
    if a.run:
        run(a.limit)
