#!/usr/bin/env python3
"""behaviour_uncertainty.py — uncertainty for the behavioural findings.

The paper's vision results all carry CIs cluster-bootstrapped by source video, but the BEHAVIOURAL
findings (activity budget, circadian profile, human-presence stimulus response) were reported as bare
point estimates with no interval and, for the stimulus response, no test at all. That inconsistency is
the kind a reviewer notices immediately: the most quotable claim in the paper ("human presence nearly
doubles movement") had the least statistical support.

The independent unit here is the RECORDING DAY, not the clip. All 3,083 present clips come from only
**seven** recording days, so clips within a day share lighting, the animal's state, whether a human was
in the room, and any gate bias for that day. Bootstrapping over clips would treat 3,083 correlated
observations as independent and produce absurdly tight intervals.

n=7 clusters is genuinely few, so intervals will be wide. That width IS the finding: it quantifies how
much a one-week window can support, and it is the honest argument for extending to the harvested
~209-day corpus.

Usage: venv/bin/python3 src/behaviour_uncertainty.py
"""
import collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECS = ROOT / "data" / "behaviour_records.json"
OUT = ROOT / "data" / "behaviour_uncertainty.json"

ACT = {"still": 0.0, "low": 0.33, "moderate": 0.66, "high": 1.0}
POS = {"contracted": 0.0, "neutral": 0.3, "arms_extended": 0.6, "climbing": 0.8,
       "flattened_spread": 1.0, "uncertain": 0.3}


def arousal(s):
    return 0.6 * ACT.get(s.get("activity"), 0.33) + 0.4 * POS.get(s.get("posture"), 0.3)


def boot_days(days, stat, iters=4000, seed=7):
    """Resample recording DAYS with replacement; `stat` maps {day: [items]} -> value."""
    rng = np.random.default_rng(seed)
    keys = list(days)
    vals = []
    for _ in range(iters):
        pick = rng.choice(keys, len(keys))
        merged = {}
        for j, k in enumerate(pick):
            merged[f"{k}#{j}"] = days[k]
        v = stat(merged)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


if __name__ == "__main__":
    recs = json.load(open(RECS))
    pres = {k: v for k, v in recs.items() if (v.get("struct") or {}).get("present")}
    byday = collections.defaultdict(list)
    for k, v in pres.items():
        byday[v.get("date") or k.split("/")[1]].append(v)
    print(f"{len(pres)} present clips / {len(byday)} recording days "
          f"({min(byday)} .. {max(byday)})")
    print(f"clips per day: {dict(sorted((d, len(v)) for d, v in byday.items()))}\n")
    res = {"n_clips": len(pres), "n_days": len(byday),
           "clips_per_day": {d: len(v) for d, v in sorted(byday.items())}}

    # ── 1. activity budget, CI by day
    print("=== ACTIVITY BUDGET (share of present clips), CI95 by recording day ===")
    classes = [c for c, _ in collections.Counter(
        r["struct"]["behavior"] for r in pres.values()).most_common()]
    res["activity_budget"] = {}
    for c in classes:
        pt = np.mean([r["struct"]["behavior"] == c for r in pres.values()])
        lo, hi = boot_days(byday, lambda d, c=c: np.mean(
            [r["struct"]["behavior"] == c for v in d.values() for r in v]))
        # per-day spread, the more interpretable number
        perday = [np.mean([r["struct"]["behavior"] == c for r in v]) for v in byday.values()]
        res["activity_budget"][c] = {"share": round(float(pt), 4),
                                     "ci95": [round(lo, 4), round(hi, 4)],
                                     "per_day_min": round(float(min(perday)), 4),
                                     "per_day_max": round(float(max(perday)), 4)}
        print(f"  {c:34s} {100*pt:5.1f}%  CI95 [{100*lo:4.1f}, {100*hi:4.1f}]  "
              f"per-day range {100*min(perday):4.1f}-{100*max(perday):4.1f}%")

    # ── 2. circadian, with MATCHED exposure.
    # Two traps here, both hit on the first attempt:
    #  (a) `video_timeline` is a clip OFFSET (mm:ss-mm:ss), not a clock time. Parsing it as an hour
    #      makes every hour look identical. Absolute clock time comes from the index `segment`
    #      (HHMMSS) plus the window's start_sec.
    #  (b) the denominator must be restricted to the SAME recording days as the numerator. The index
    #      holds 13,342 extracted clips but only 3,205 were behaviourally analysed; leaving off-date
    #      clips in the denominator deflates the rate. (Here it is a 2.6% effect, but the shape of the
    #      curve is the headline claim, so the exposure has to be matched rather than assumed benign.)
    idx = json.load(open(ROOT / "src" / "octopus_clips_verified.json"))["clips"]
    meta = {c["clip_path"]: c for c in idx}
    rec_dates = set(byday)

    def abshour(c):
        s = str(c.get("segment") or "")
        if len(s) < 6:
            return None
        try:
            hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:6])
        except Exception:
            return None
        return ((hh * 3600 + mm * 60 + ss + int(c.get("start_sec", 0))) // 3600) % 24

    den = collections.defaultdict(collections.Counter)
    num = collections.defaultdict(collections.Counter)
    for c in idx:
        parts = c["clip_path"].split("/")
        d = parts[1] if len(parts) > 2 else None
        if d not in rec_dates:
            continue
        h = abshour(c)
        if h is not None:
            den[d][h] += 1
    for k in pres:
        c = meta.get(k)
        if not c:
            continue
        h = abshour(c)
        if h is not None:
            num[k.split("/")[1]][h] += 1

    def rate(days, hours):
        n = sum(num[d][h] for d in days for h in hours)
        m = sum(den[d][h] for d in days for h in hours)
        return (n / m) if m else None

    AFT, NIGHT = range(13, 20), range(0, 6)
    A, N = rate(list(den), AFT), rate(list(den), NIGHT)
    print("\n=== CIRCADIAN (matched exposure: numerator and denominator on the same 7 days) ===")
    print(f"  afternoon 13-19h {100*A:.1f}%   overnight 00-05h {100*N:.1f}%   ratio {A/N:.1f}x")
    hourly = {}
    for h in range(24):
        n = sum(num[d][h] for d in den); m = sum(den[d][h] for d in den)
        if m >= 5:
            hourly[h] = {"present": n, "extracted": m, "rate": round(n / m, 4),
                         "per_day": sorted(round(num[d][h] / den[d][h], 3)
                                           for d in den if den[d][h] >= 5)}
    peak = max(hourly, key=lambda h: hourly[h]["rate"])
    print(f"  peak hour {peak:02d}h at {100*hourly[peak]['rate']:.1f}%  "
          f"(per-day {[round(100*x) for x in hourly[peak]['per_day']]})")
    ok = tot = 0
    for d in sorted(den):
        a, n_ = rate([d], AFT), rate([d], NIGHT)
        if a is None or n_ is None:
            continue
        if sum(den[d][h] for h in AFT) >= 10 and sum(den[d][h] for h in NIGHT) >= 10:
            tot += 1; ok += (a > n_)
    print(f"  afternoon > overnight on {ok}/{tot} days with enough exposure in both windows")
    res["circadian"] = {"afternoon_rate": round(A, 4), "overnight_rate": round(N, 4),
                        "ratio": round(A / N, 2), "peak_hour": peak,
                        "peak_rate": hourly[peak]["rate"],
                        "days_afternoon_higher": ok, "n_days_testable": tot,
                        "hourly": hourly}

    # ── 3. stimulus response: human present vs absent, day-clustered
    print("\n=== STIMULUS RESPONSE: human present vs absent (day-clustered) ===")
    def split(v):
        return "human" if (v.get("struct") or {}).get("context") == "human_present" else "other"
    for metric, fn in (("mean_motion", lambda r: r.get("mean_motion")),
                       ("arousal", lambda r: arousal(r["struct"]))):
        dd = {}
        for d, vs in byday.items():
            h = [fn(r) for r in vs if split(r) == "human" and fn(r) is not None]
            o = [fn(r) for r in vs if split(r) == "other" and fn(r) is not None]
            if h and o:
                dd[d] = (float(np.mean(h)), float(np.mean(o)))
        if len(dd) < 3:
            print(f"  {metric}: only {len(dd)} days have both conditions — not testable"); continue
        diffs = [a - b for a, b in dd.values()]
        pt = float(np.mean(diffs))
        rng = np.random.default_rng(11)
        bs = [float(np.mean(rng.choice(diffs, len(diffs)))) for _ in range(4000)]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        # exact sign test over days (paired, distribution-free, n = days)
        pos = sum(1 for x in diffs if x > 0)
        from math import comb
        p = sum(comb(len(diffs), i) for i in range(pos, len(diffs) + 1)) / 2 ** len(diffs)
        mh = float(np.mean([a for a, _ in dd.values()])); mo = float(np.mean([b for _, b in dd.values()]))
        print(f"  {metric:12s} human {mh:.4f} vs other {mo:.4f} | paired diff {pt:+.4f} "
              f"CI95 [{lo:+.4f}, {hi:+.4f}] | higher on {pos}/{len(diffs)} days, sign-test p={p:.4f}")
        res[f"stimulus_{metric}"] = {"n_days": len(dd), "mean_human": round(mh, 4),
                                     "mean_other": round(mo, 4), "paired_diff": round(pt, 4),
                                     "ci95": [round(lo, 4), round(hi, 4)],
                                     "days_higher": pos, "sign_test_p": round(p, 4)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")
