"""refetch_clips.py — re-extract clips that are IN the index but whose mp4 is gone from disk.

WHY. The index records 13,342 extraction decisions but only ~3,455 of those clips still exist on
this machine; the rest were extracted on boxes that no longer hold them. Every missing entry still
carries `video_url` + `start_sec`/`end_sec`, so the clip can be rebuilt exactly by an ffmpeg
byte-range copy from the server — no re-scanning, no re-deciding, byte-identical extraction.

CAMERA SPREAD, not random. 71% of the missing clips are `Right_Top` (IR), which is also the largest
mean file (8.65 MB) and the camera the segmentation work cannot use. A uniform random draw would
therefore spend most of the bandwidth and disk on the least useful footage. Default is round-robin
across cameras, spread across dates within each camera. `--random` restores a plain shuffle and
`--cameras` restricts the pool.

Writes into `src/octopus_clips_verified/{date}/{segment}/{file}.mp4` — the layout the labelling
scripts already scan, so refetched clips are picked up with no further wiring.

Resumable: an existing file >10 kB is skipped. Concurrency defaults to 3 — the footage server caps
aggregate throughput around 5 MB/s and collapses under sustained parallelism (see memory
`server-throttling-sustained-load`), so more workers buy nothing and risk the whole pull.

Usage
  set -a; . ./.env; set +a          # extract_clip reads OCTOPUS_USER/OCTOPUS_PASS from the env
  venv/bin/python3 src/refetch_clips.py --n 1000
  venv/bin/python3 src/refetch_clips.py --n 50 --cameras Right_Front Right_Back
"""
import argparse, collections, glob, json, random, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from harvest_stream import extract_clip          # proven byte-range copy; reads AUTH from env

INDEX = REPO / "src" / "octopus_clips_verified.json"
DEST = REPO / "src" / "octopus_clips_verified"
ROOTS = [REPO / "src" / "octopus_clips_verified", REPO / "data" / "octopus_clips_verified"]

lock = threading.Lock()
state = {"ok": 0, "fail": 0, "bytes": 0}


def rel3(p):
    return "/".join(str(p).strip("/").split("/")[-3:])


def pick(missing, n, mode):
    """Round-robin over cameras, and within a camera spread evenly across dates."""
    if mode == "random":
        random.Random(0).shuffle(missing)
        return missing[:n]
    bycam = collections.defaultdict(list)
    for e in missing:
        bycam[e.get("camera")].append(e)
    for cam, lst in bycam.items():
        bydate = collections.defaultdict(list)
        for e in lst:
            bydate[e.get("date")].append(e)
        for v in bydate.values():
            random.Random(0).shuffle(v)
        # interleave dates so a camera's quota is not all one day
        out, keys = [], sorted(bydate)
        while any(bydate[k] for k in keys):
            for k in keys:
                if bydate[k]:
                    out.append(bydate[k].pop())
        bycam[cam] = out
    order, cams = [], sorted(bycam)
    while len(order) < n and any(bycam[c] for c in cams):
        for c in cams:
            if bycam[c] and len(order) < n:
                order.append(bycam[c].pop(0))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--random", action="store_true", help="plain shuffle instead of camera round-robin")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = json.load(open(INDEX))
    entries = [x for x in (d if isinstance(d, list) else d.get("clips", [])) if isinstance(x, dict)]
    disk = set()
    for r in ROOTS:
        for f in glob.glob(str(r) + "/**/*.mp4", recursive=True):
            disk.add(rel3(f))
    missing = [x for x in entries
               if x.get("clip_path") and rel3(x["clip_path"]) not in disk
               and x.get("video_url") and x.get("start_sec") is not None]
    if args.cameras:
        missing = [x for x in missing if x.get("camera") in args.cameras]
    print(f"index {len(entries)} | on disk {len(disk)} | missing & refetchable {len(missing)}")

    sel = pick(missing, args.n, "random" if args.random else "spread")
    print(f"selected {len(sel)}")
    print("  by camera:", dict(collections.Counter(x.get('camera') for x in sel).most_common()))
    print("  by date  :", dict(collections.Counter(x.get('date') for x in sel).most_common()))
    if args.dry_run:
        print("[dry-run] nothing fetched."); return

    t0 = time.time()

    def work(e):
        rel = rel3(e["clip_path"])
        out = DEST / rel
        if out.exists() and out.stat().st_size > 10000:
            return
        ok = extract_clip(e["video_url"], e["start_sec"], e["end_sec"], out)
        with lock:
            if ok:
                state["ok"] += 1; state["bytes"] += out.stat().st_size
            else:
                state["fail"] += 1
            n = state["ok"] + state["fail"]
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}/{len(sel)}] ok={state['ok']} fail={state['fail']} "
                      f"{state['bytes']/1e9:.2f} GB | {state['bytes']/1e6/max(el,1):.2f} MB/s "
                      f"| {n/max(el,1)*60:.1f} clips/min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, sel))

    el = time.time() - t0
    print(f"\nDONE. ok={state['ok']} fail={state['fail']} | {state['bytes']/1e9:.2f} GB "
          f"in {el/60:.1f} min ({state['bytes']/1e6/max(el,1):.2f} MB/s)")
    print(f"-> {DEST}")


if __name__ == "__main__":
    main()
