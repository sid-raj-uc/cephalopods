"""caption_local_235b.py — one UNIFORM Qwen3-VL-235B caption+ethogram pass over every clip
that exists on local disk, written to a STANDALONE json.

WHY. Stage 3's input labelling is currently inconsistent: of the 3,455 index clips whose mp4 is
actually on this machine, only ~500 carry a 235B caption and ~2,888 carry only the 30B one. The two
models disagree enormously on presence — on the same 847 clips the 235B called 63% "octopus not
present" and the 30B only 11% (a ~6x gap) — so any filter built on "the caption" means something
different depending on which clip you look at. This produces one verdict, from one model, for the
whole local corpus.

DELIBERATE CHOICES
  * Reuses `caption_openrouter.process_one` and `build_prompt()` VERBATIM — same prompt, same
    N_KEEP=6 best-frame selection, same CLAHE, same PRESENT_MIN=0.5 skip. So these results are
    directly comparable to the existing `caption_235b` field rather than being a new distribution.
  * Writes `data/local_235b_labels.json`, keyed by the last three path components
    (`date/segment/file.mp4`). It does **NOT** touch `octopus_clips_verified.json` — the index backs
    the paper's frozen sets, and this is a re-labelling experiment until we decide otherwise.
  * Clips that ALREADY have a `caption_235b` in the index are copied across, not re-called (same
    model, same prompt), so the output covers all on-disk clips for ~1/6 the API calls.
  * Resumable: keys already present in the output json are skipped.

NOT FIXED BY THIS. Coverage. Only 26% of the 13,342 index entries have a surviving mp4; the rest were
extracted on machines that no longer hold them. This labels what exists, and cannot say anything
about the missing 74%.

Usage
  venv/bin/python3 src/caption_local_235b.py --limit 5      # smoke test
  venv/bin/python3 src/caption_local_235b.py --workers 8
"""
import argparse, datetime, glob, json, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import caption_openrouter as C

INDEX = REPO / "src" / "octopus_clips_verified.json"
ROOTS = [REPO / "src" / "octopus_clips_verified", REPO / "data" / "octopus_clips_verified"]
OUT = REPO / "data" / "local_235b_labels.json"

CAP_KEY, ETHO_KEY, MODEL_KEY = "caption_235b", "ethogram_label_235b", "caption_235b_model"
KEEP = (CAP_KEY, ETHO_KEY, MODEL_KEY, f"{CAP_KEY}_max_p", f"{CAP_KEY}_at")

io_lock = threading.Lock()
state = {"api": 0, "copied": 0, "absent": 0, "err": 0}


def rel3(p):
    return "/".join(str(p).strip("/").split("/")[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    d = json.load(open(INDEX))
    entries = [x for x in (d if isinstance(d, list) else d.get("clips", [])) if isinstance(x, dict)]

    # which clips actually exist, and under which root
    disk = {}
    for r in ROOTS:
        for f in glob.glob(str(r) + "/**/*.mp4", recursive=True):
            disk.setdefault(rel3(f), r)
    on_disk = [e for e in entries if e.get("clip_path") and rel3(e["clip_path"]) in disk]
    print(f"index {len(entries)} clips | on local disk {len(on_disk)} ({len(on_disk)/len(entries):.0%})")

    out = json.load(open(args.out)) if Path(args.out).exists() else {}
    print(f"output already has {len(out)} clips")

    todo, copy = [], []
    for e in on_disk:
        k = rel3(e["clip_path"])
        if k in out:
            continue
        (copy if e.get(CAP_KEY) else todo).append(e)
    if args.limit:
        todo = todo[:args.limit]
    print(f"to copy from index (already 235B): {len(copy)} | to CALL the API: {len(todo)}")

    # 1. copy existing 235B labels straight across -- same model, same prompt, no reason to re-pay
    for e in copy:
        out[rel3(e["clip_path"])] = {**{k: e.get(k) for k in KEEP if e.get(k) is not None},
                                     "camera": e.get("camera"), "date": e.get("date"),
                                     "source": "index_existing"}
        state["copied"] += 1
    if copy:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"copied {len(copy)} existing 235B labels")

    if not todo:
        print("nothing to call. done."); return

    det = C.load_detector()
    prompt = C.build_prompt()
    print(f"detector loaded; calling {C.OR_MODEL} with {args.workers} workers", flush=True)
    t0 = time.time()

    def work(e):
        root = disk[rel3(e["clip_path"])]
        try:
            e2, status = C.process_one(dict(e), root, CAP_KEY, ETHO_KEY, MODEL_KEY, prompt, det)
        except Exception as ex:
            with io_lock:
                state["err"] += 1
            return
        with io_lock:
            if status.startswith("apifail") or status in ("missing", "noframes"):
                state["err"] += 1
                return
            state["absent" if status == "absent" else "api"] += 1
            out[rel3(e["clip_path"])] = {**{k: e2.get(k) for k in KEEP if e2.get(k) is not None},
                                         "camera": e2.get("camera"), "date": e2.get("date"),
                                         "source": f"fresh_{status}"}
            n = state["api"] + state["absent"]
            if n % 25 == 0:
                json.dump(out, open(args.out, "w"), indent=1)
                el = time.time() - t0
                print(f"  [{n}/{len(todo)}] api={state['api']} absent={state['absent']} "
                      f"err={state['err']} | {n/max(el,1)*60:.1f} clips/min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nDONE. api={state['api']} absent={state['absent']} copied={state['copied']} "
          f"err={state['err']} | total in file {len(out)} | {time.time()-t0:.0f}s")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
