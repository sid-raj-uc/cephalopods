#!/usr/bin/env python3
"""vlm_reliability_stats.py — how self-consistent is the structured behavioural extractor?

Every headline behavioural result in this project (activity budget, circadian profile,
human-presence stimulus response, kinematics x behaviour) groups clips by labels this extractor
produced, and until now those labels had never been checked. `vlm_reliability.py --run` re-ran the
extractor over the frozen VLM-250 sample using a **disjoint set of input frames** (detector ranks
N_KEEP..2*N_KEEP instead of the top N_KEEP). This scores the two runs against each other.

WHAT THIS MEASURES, PRECISELY: **frame-sampling sensitivity** — does a label survive being shown a
different set of clear frames from the same clip. That is CONSISTENCY, not accuracy. Consistency
upper-bounds accuracy (a label that changes when you look at different frames cannot be reliably
correct), but a perfectly consistent extractor can still be consistently wrong. Say "self-consistent",
never "accurate", when reporting these numbers.

Statistics: Cohen's kappa plus raw agreement, per field. CIs are cluster-bootstrapped BY SOURCE VIDEO
(clips from one recording are not independent — the same discipline as kinematics_stats.py).
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A_PATH = ROOT / "data" / "behaviour_records.json"
B_PATH = ROOT / "data" / "behaviour_records_retest.json"
OUT = ROOT / "data" / "vlm_reliability_stats.json"
FIELDS = ["present", "behavior", "posture", "activity", "location", "context",
          "body_color", "color_or_texture_change"]


def source_video(rel):
    p = Path(rel)
    return f"{p.parent.parent.name}/{p.parent.name}"


def kappa(a, b):
    """Cohen's kappa for two label sequences."""
    labs = sorted(set(a) | set(b))
    if len(labs) < 2:
        return 1.0 if list(a) == list(b) else 0.0
    idx = {l: i for i, l in enumerate(labs)}
    n = len(a)
    obs = sum(x == y for x, y in zip(a, b)) / n
    ca = collections.Counter(a); cb = collections.Counter(b)
    exp = sum((ca[l] / n) * (cb[l] / n) for l in labs)
    return 1.0 if exp == 1 else (obs - exp) / (1 - exp)


def boot_by_video(pairs, fn, iters=2000, seed=3):
    """pairs: list of (video, a, b). Resample VIDEOS with replacement."""
    rng = np.random.default_rng(seed)
    byv = collections.defaultdict(list)
    for v, x, y in pairs:
        byv[v].append((x, y))
    keys = list(byv)
    vals = []
    for _ in range(iters):
        s = [p for k in rng.choice(keys, len(keys)) for p in byv[k]]
        if not s:
            continue
        vals.append(fn([x for x, _ in s], [y for _, y in s]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    A = json.load(open(A_PATH)); B = json.load(open(B_PATH))
    keys = [k for k in B if k in A]
    print(f"VLM-250 self-consistency: {len(keys)} clips / "
          f"{len({source_video(k) for k in keys})} source videos")
    print("Condition A = top-N frames by detector score; B = ranks N..2N (DISJOINT frames).")
    print("This is CONSISTENCY (frame-sampling sensitivity), NOT accuracy.\n")

    res = {"n_clips": len(keys), "n_videos": len({source_video(k) for k in keys}), "fields": {}}
    print(f"{'field':26s}{'n':>5}{'agree':>8}{'kappa':>8}   {'kappa CI95 (by video)':>24}")
    for f in FIELDS:
        pairs = []
        for k in keys:
            sa = (A[k].get("struct") or {}); sb = (B[k].get("struct") or {})
            if f not in sa or f not in sb:
                continue
            pairs.append((source_video(k), str(sa[f]), str(sb[f])))
        if len(pairs) < a.min_n:
            continue
        xs = [p[1] for p in pairs]; ys = [p[2] for p in pairs]
        agr = sum(x == y for x, y in zip(xs, ys)) / len(xs)
        kp = kappa(xs, ys)
        lo, hi = boot_by_video(pairs, kappa)
        # A field can score perfect agreement without the model having judged anything: if its value
        # is fixed by a deterministic preprocessing gate (colour fields are forced to `uncertain` on
        # greyscale/IR clips), both runs agree trivially. `color_or_texture_change` is 100% explained
        # by the greyscale gate, so its kappa of 1.000 measures the gate, NOT reliability. Flag such
        # fields instead of reporting them as excellent.
        gate = None
        if agr > 0.999:
            det = sum(1 for k in keys
                      if (B[k].get("grey") and str((B[k].get("struct") or {}).get(f)) == "uncertain")
                      or ((not B[k].get("grey")) and str((B[k].get("struct") or {}).get(f)) != "uncertain"))
            if det == len(keys):
                gate = "value fully determined by the greyscale gate — kappa is not a reliability measure"
        res["fields"][f] = {"n": len(pairs), "agreement": round(agr, 4), "kappa": round(kp, 4),
                            "kappa_ci95": [round(lo, 4), round(hi, 4)], "gate_artifact": gate}
        note = "  <-- ARTIFACT: gate-determined, not a reliability measure" if gate else ""
        print(f"{f:26s}{len(pairs):>5}{agr:>8.3f}{kp:>8.3f}   [{lo:+.3f}, {hi:+.3f}]{note}")

    # behaviour confusion + which classes are unstable
    pb = [(source_video(k), str((A[k].get('struct') or {}).get('behavior')),
           str((B[k].get('struct') or {}).get('behavior'))) for k in keys]
    pb = [p for p in pb if p[1] != 'None' and p[2] != 'None']
    per = collections.defaultdict(lambda: [0, 0])
    for _, x, y in pb:
        per[x][1] += 1
        if x == y:
            per[x][0] += 1
    print("\nper-class stability (A's label -> same label in B):")
    for c, (ok, tot) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"   {c:34s} {ok:3d}/{tot:3d} = {100*ok/tot:5.1f}%")
    res["behavior_per_class"] = {c: {"stable": ok, "n": tot, "rate": round(ok / tot, 4)}
                                 for c, (ok, tot) in per.items()}

    # abstention: how often the model declines
    unc = sum(1 for k in keys if str((B[k].get('struct') or {}).get('behavior')).lower() == 'uncertain')
    res["uncertain_rate_B"] = round(unc / len(keys), 4)
    print(f"\nabstention ('uncertain') in condition B: {unc}/{len(keys)} = {100*unc/len(keys):.1f}%")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")
