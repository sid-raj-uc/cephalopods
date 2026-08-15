"""kinematics_stats.py — does skeleton kinematics separate the VLM behaviour classes? (with statistics)

Answers the two reviewer objections the n=41 version could not:
  (a) "n=2 is anecdote"      -> behaviour-stratified sample, and the PRIMARY test is at
                                SOURCE-VIDEO level (clip-level would be pseudo-replication:
                                clips from one recording are not independent).
  (b) "px/s just means the   -> every test is repeated on a SCALE-INVARIANT speed,
      animal was closer"        tip speed / arm spread  (body-lengths per second).

Tests: Kruskal-Wallis across classes (+ epsilon^2 effect size), pairwise resting-vs-each
Mann-Whitney with Holm correction, and cluster-bootstrap 95% CIs on the per-class medians.

Usage: venv/bin/python3 src/kinematics_stats.py [--motion data/skeleton_motion.json]
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
CAMS = ("Right_Front", "Right_Back", "Right_Right")
RESTING = "Resting / stationary"


def source_video(rel):
    p = Path(rel); return f"{p.parent.parent.name}/{p.parent.name}"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (order preserved)."""
    idx = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        adj[i] = min(1.0, run)
    return adj


def cluster_bootstrap_ci(by_video, n=5000, seed=0):
    """95% CI of the median of per-video values, resampling VIDEOS (the independent unit)."""
    v = np.asarray(by_video, float)
    if len(v) < 3:
        return None
    rng = np.random.RandomState(seed)
    meds = [np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(n)]
    return [round(float(np.percentile(meds, 2.5)), 1), round(float(np.percentile(meds, 97.5)), 1)]


def collect(motion_path):
    br = json.load(open(REPO / "data" / "behaviour_records.json"))
    mot = json.load(open(motion_path))
    rows = []
    for rel, k in mot.items():
        if rel.startswith("_") or not isinstance(k, dict):
            continue
        if not k.get("activity_px_s") or not k.get("arm_spread_px"):
            continue
        if not any(c in rel for c in CAMS):
            continue
        rec = br.get(rel) or {}
        beh = (rec.get("struct") or {}).get("behavior")
        if not beh or beh == "uncertain":
            continue
        spread = k["arm_spread_px"]["mean"]
        raw = k["activity_px_s"]["mean"]
        rows.append({"clip": rel, "video": source_video(rel), "behavior": beh,
                     "raw": raw,                                   # px/s
                     "norm": (raw / spread) if spread else None,   # body-lengths/s
                     "occluded_frac": k.get("occluded_frac"),
                     "cfg": k.get("_cfg", {})})
    return rows


def analyse(rows, key, label):
    """Video-level analysis for one speed definition."""
    print(f"\n=== {label} ===")
    per_video = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r[key] is not None:
            per_video[r["behavior"]][r["video"]].append(r[key])
    groups, summary = {}, {}
    for beh, vids in per_video.items():
        vals = [float(np.median(v)) for v in vids.values()]   # one number per video
        if len(vals) >= 3:
            groups[beh] = vals
            summary[beh] = {"n_videos": len(vals),
                            "n_clips": sum(len(v) for v in vids.values()),
                            "median": round(float(np.median(vals)), 2),
                            "ci95": cluster_bootstrap_ci(vals)}
    if len(groups) < 2:
        print("  not enough classes with >=3 videos"); return None
    order = sorted(summary, key=lambda b: summary[b]["median"])
    for b in order:
        s = summary[b]
        print(f"  {b:32s} median {s['median']:8.2f}  CI95 {s['ci95']}  "
              f"({s['n_videos']} videos / {s['n_clips']} clips)")

    H, p = stats.kruskal(*[groups[b] for b in order])
    N = sum(len(v) for v in groups.values()); k = len(groups)
    eps2 = (H - k + 1) / (N - k) if N > k else float("nan")
    print(f"  Kruskal-Wallis H={H:.2f}  p={p:.2e}  eps^2={eps2:.3f}  (N={N} videos, k={k})")

    pw = {}
    if RESTING in groups:
        others = [b for b in order if b != RESTING]
        raws = [stats.mannwhitneyu(groups[RESTING], groups[b], alternative="two-sided").pvalue
                for b in others]
        adj = holm(raws)
        print(f"  resting vs each (Mann-Whitney, Holm-adjusted):")
        for b, r0, a in zip(others, raws, adj):
            print(f"    vs {b:30s} p={r0:.4f}  p_holm={a:.4f}  {'*' if a < 0.05 else 'ns'}")
            pw[b] = {"p": float(r0), "p_holm": float(a), "sig": bool(a < 0.05)}
    return {"summary": summary, "kruskal": {"H": float(H), "p": float(p), "eps2": float(eps2),
                                            "n_videos": int(N), "k": int(k)},
            "resting_vs": pw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default=str(REPO / "data" / "skeleton_motion.json"))
    ap.add_argument("--out", default=str(REPO / "data" / "kinematics_stats.json"))
    args = ap.parse_args()
    rows = collect(args.motion)
    cfgs = {json.dumps(r["cfg"], sort_keys=True) for r in rows if r["cfg"]}
    print(f"{len(rows)} clips / {len({r['video'] for r in rows})} videos")
    print(f"config stamps present: {len(cfgs)} -> {list(cfgs)[:2]}")
    if len(cfgs) > 1:
        print("  WARNING: mixed pipeline configs in this motion file — results are not comparable")

    res = {"n_clips": len(rows), "n_videos": len({r["video"] for r in rows}),
           "configs": sorted(cfgs),
           "raw_px_s": analyse(rows, "raw", "RAW arm-tip speed (px/s)"),
           "norm_bodylen_s": analyse(rows, "norm", "SCALE-INVARIANT speed (body-lengths/s)")}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")

    r_, n_ = res["raw_px_s"], res["norm_bodylen_s"]
    if r_ and n_:
        print("\nVERDICT")
        print(f"  raw  : p={r_['kruskal']['p']:.2e} eps^2={r_['kruskal']['eps2']:.3f}")
        print(f"  norm : p={n_['kruskal']['p']:.2e} eps^2={n_['kruskal']['eps2']:.3f}")
        kill = (r_["kruskal"]["p"] > 0.05) or (r_["kruskal"]["eps2"] < 0.06)
        print("  KILL CRITERION MET — record as a negative" if kill
              else "  separation holds at video level")


if __name__ == "__main__":
    main()
