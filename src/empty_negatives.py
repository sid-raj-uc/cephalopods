#!/usr/bin/env python3
"""empty_negatives.py — build EMPTY-V2: a properly-powered, detector-independent empty-tank set.

WHY. SEG-TEST's 19 empty-tank negatives come from **2 source videos, 18 of them from one recording**,
so the presence AUC of 0.794 the paper reports is effectively a single-video estimate: no CI can be
attached to it and it cannot be ordered against any many-video estimate. Every presence claim resting
on those 19 frames inherits the problem, including R8's temporal-fusion presence gain
(0.794 -> 0.9685 ema / 0.9495 flow). Fixing the negative set is the highest-value repair available to
the presence benchmark.

TWO DESIGN RULES, both load-bearing:

1. **Detector-independent sampling.** Frames are drawn at UNIFORM RANDOM TIMESTAMPS from whole source
   videos on the server, never from clips the extraction pipeline selected. Extracted clips exist only
   because the CLIP detector fired on them, so a negative set built from them is enriched for that
   detector's false positives — which biases a detector-vs-segmenter comparison in the segmenter's
   favour. (That bias is present in REFL-28 and is recorded as a caveat on R10; EMPTY-V2 must not
   repeat it.)

2. **Zero-shot for the segmenter.** Every video whose session appears in thin768's 142-video training
   partition is excluded, as is every session in the CLIP detector's training manifest. Sessions are
   matched on `(date, HHMM)`: the server names a recording `095421--vv-1.mp4` while the local clip tree
   calls the same session `095420`, so second-level matching silently fails. HHMM matching over-excludes
   if two recordings start in the same minute, which is the safe direction.

Frames are STAGED ONLY. `verified` starts null; nothing may be scored until a frame is reviewed,
because a uniformly-sampled frame very often DOES contain the animal — that is the point of sampling
uniformly, and it is why this file cannot assume its own frames are empty.

Usage:
  venv/bin/python3 src/empty_negatives.py --sample --videos 60 --per-video 2
  venv/bin/python3 src/empty_negatives.py --contact-sheet
"""
import argparse, collections, json, os, random, re, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
from server_creds import USER, PASS
os.environ.setdefault("OCTOPUS_USER", USER); os.environ.setdefault("OCTOPUS_PASS", PASS)
import harvest_stream as H

OUTDIR = REPO / "data" / "empty_negatives"
INDEX = OUTDIR / "index.json"
TRAIN_VIDEOS = REPO / "data" / "thin768_train_videos.json"
DET_MANIFEST = REPO / "data" / "frames" / "manifest.csv"
MAXSIDE = 1024


def hhmm(date, seg):
    return (date, seg[:4])


def excluded_sessions():
    """thin768 training sessions + CLIP-detector training sessions, as (date, HHMM)."""
    if not TRAIN_VIDEOS.exists():
        sys.exit(f"missing {TRAIN_VIDEOS} — run the leakage audit first; no list, no experiment")
    seg = set()
    for v in json.load(open(TRAIN_VIDEOS)):
        d, s = v.split("/")
        seg.add(hhmm(d, s))
    det, bad, tot = set(), 0, 0
    import csv
    for r in csv.DictReader(open(DET_MANIFEST)):
        tot += 1
        f = r["path"].split("/")[-1]
        m = re.search(r"(20\d\d-\d\d-\d\d)_(\d{6})_", f) or re.search(r"(20\d\d-\d\d-\d\d)_(\d{4})_", f)
        if not m:
            bad += 1
            continue
        det.add(hhmm(m.group(1), m.group(2)))
    if bad:
        sys.exit(f"LEAKAGE CHECK ABORTED: {bad}/{tot} detector-manifest rows unparsed")
    print(f"exclusions: {len(seg)} thin768 training sessions + {len(det)} detector sessions "
          f"({tot} manifest rows, 0 unparsed) -> {len(seg | det)} unique")
    return seg | det


def parse_url(url):
    """-> (camera, date, segment) from .../{Camera}/Local/{date}/{HHMMSS}--vv-1.mp4"""
    m = re.search(r"/([^/]+)/Local/(\d{4}-\d{2}-\d{2})/(\d{6})", url.replace("%20", " "))
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def grab(url, t, dst, timeout=90):
    """One full frame at time t via fast input-seek (small ranged download), scaled to <=MAXSIDE.

    Never raises: the footage server stalls intermittently (it throttles hard under sustained load), and
    an uncaught TimeoutExpired killed a whole sampling run at frame 28. A slow video is skipped, not
    fatal.
    """
    auth = "Basic " + __import__("base64").b64encode(f"{USER}:{PASS}".encode()).decode()
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t),
           "-headers", f"Authorization: {auth}\r\n", "-i", url, "-frames:v", "1",
           "-vf", f"scale='min({MAXSIDE},iw)':-2", "-y", str(dst)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    return dst.exists() and cv2.imread(str(dst)) is not None


def sample(n_videos=60, per_video=2, seed=23, collections_=None, outdir=None):
    """collections_: restrict the crawl to specific server collections.

    DOMAIN MATCHING MATTERS MORE THAN VOLUME. The two Nity collections are two different physical
    setups: `O-vulgaris-Nity-2026-2-20--` is the lab tank the SEG-TEST positives come from (Camera 2,
    lab shelving), while `O-vulgaris-Nity-2025-9-17--` is a different room and tank entirely (Cameras
    7/9, a blue reef tank with curtains and a table). A first run crawled both and drew mostly the
    2025 setup; separating "2026 tank containing an octopus" from "2025 room containing no octopus"
    would have measured SCENE DIFFERENCE, not octopus presence, and produced a flatteringly high AUC
    for the wrong reason. Match the collection to the positives for the headline number; the
    cross-setup frames are a separate, genuinely useful question (false positives in an unseen
    environment) and must be reported as such, never pooled.
    """
    global OUTDIR, INDEX
    if outdir:
        OUTDIR = REPO / "data" / outdir
        INDEX = OUTDIR / "index.json"
    excl = excluded_sessions()
    plan = []
    for coll in (collections_ or H.NITY_COLLECTIONS):
        try:
            dates = H.list_dates(H.BASE + coll)
        except Exception as e:
            print(f"  crawl failed for {coll}: {e}")
            continue
        for (cam, date), url in sorted(dates.items()):
            plan.append((cam, date, url))
    print(f"crawled {len(plan)} (camera,date) listings")

    rng = random.Random(seed)
    rng.shuffle(plan)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows, used_sessions, skipped = [], set(), collections.Counter()
    if INDEX.exists():                                   # resumable: a stalled server should not cost a rerun
        prev = json.load(open(INDEX))
        rows = prev.get("rows", [])
        for r in rows:
            d, s = r["video"].split("/")
            used_sessions.add(hhmm(d, s))
        print(f"resuming: {len(rows)} frames / {len(used_sessions)} videos already staged")
    # Cap videos taken per (camera, date) listing. Without this the first listing alone supplies every
    # recording asked for: a first run drew all 40 frames from 20 recordings on ONE date and ONE camera
    # — reproducing precisely the single-video concentration this benchmark exists to fix. Spread across
    # cameras and dates is the whole point, so listings are visited round-robin.
    MAX_PER_LISTING = 2
    taken = collections.Counter()
    for _round in range(MAX_PER_LISTING):
      for cam, date, listing in plan:
        if len(used_sessions) >= n_videos:
            break
        if taken[(cam, date)] >= MAX_PER_LISTING:
            continue
        try:
            vids = H.list_videos(listing)
        except Exception:
            skipped["listing_error"] += 1
            continue
        rng.shuffle(vids)
        for vurl in vids:
            if len(used_sessions) >= n_videos or taken[(cam, date)] >= MAX_PER_LISTING:
                break
            c, d, s = parse_url(vurl)
            if d is None:
                skipped["unparsed_url"] += 1
                continue
            key = hhmm(d, s)
            if key in excl:
                skipped["excluded_training_session"] += 1
                continue
            if key in used_sessions:
                continue
            dur = H.probe_duration(vurl)
            if dur < 60:
                skipped["short_or_unreadable"] += 1
                continue
            got = 0
            for j in range(per_video):
                t = rng.uniform(0.05 * dur, 0.95 * dur)          # UNIFORM: no model chose this frame
                k = f"empty_{len(rows):04d}"
                dst = OUTDIR / f"{k}.jpg"
                if grab(vurl, t, dst):
                    rows.append({"key": k, "image": dst.name, "camera": c,
                                 "video": f"{d}/{s}", "url": vurl, "t": round(t, 1),
                                 "verified": None, "review": None})
                    got += 1
            if got:
                used_sessions.add(key)
                taken[(cam, date)] += 1
                if len(used_sessions) % 5 == 0:
                    print(f"  {len(used_sessions)} videos / {len(rows)} frames", flush=True)
                    _save(rows, used_sessions, excl, skipped, seed, per_video)
    _save(rows, used_sessions, excl, skipped, seed, per_video)
    print(f"\nstaged {len(rows)} frames / {len(used_sessions)} source videos -> {INDEX}")
    print(f"skipped: {dict(skipped)}")


def _save(rows, used, excl, skipped, seed, per_video):
    json.dump({"n": len(rows), "n_videos": len(used), "seed": seed, "per_video": per_video,
               "sampling": "UNIFORM random timestamps from whole source videos — detector-independent",
               "exclusions": f"{len(excl)} sessions (thin768 training + CLIP-detector training), "
                             "matched on (date, HHMM)",
               "verification_status": "PENDING — no frame may be scored until verified is True",
               "skipped": dict(skipped), "rows": rows}, open(INDEX, "w"), indent=1)


def contact_sheet(cols=2, cw=760, ch=470, per=6):
    idx = json.load(open(INDEX))
    rows = idx["rows"]
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        nr = (len(chunk) + cols - 1) // cols
        sheet = np.full((nr * ch, cols * cw, 3), 25, np.uint8)
        for i, r in enumerate(chunk):
            im = cv2.imread(str(OUTDIR / r["image"]))
            if im is None:
                continue
            h, w = im.shape[:2]
            sc = min((cw - 14) / w, (ch - 34) / h)
            im = cv2.resize(im, (int(w * sc), int(h * sc)))
            y, x = (i // cols) * ch, (i % cols) * cw
            sheet[y + 30:y + 30 + im.shape[0], x + 7:x + 7 + im.shape[1]] = im
            cv2.putText(sheet, f"#{s+i} {r['camera']}", (x + 10, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 235, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUTDIR / f"sheet_{s//per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"wrote {(len(rows)+per-1)//per} sheets to {OUTDIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--contact-sheet", action="store_true")
    ap.add_argument("--videos", type=int, default=60)
    ap.add_argument("--per-video", type=int, default=2)
    ap.add_argument("--collection", default="", help="restrict crawl to one server collection "
                    "(match the positives' setup — see the docstring on domain matching)")
    ap.add_argument("--outdir", default="", help="output dir under data/ (default empty_negatives)")
    a = ap.parse_args()
    if a.outdir:
        OUTDIR = REPO / "data" / a.outdir
        INDEX = OUTDIR / "index.json"
    if a.sample:
        sample(a.videos, a.per_video,
               collections_=[a.collection] if a.collection else None,
               outdir=a.outdir or None)
    if a.contact_sheet:
        contact_sheet()
