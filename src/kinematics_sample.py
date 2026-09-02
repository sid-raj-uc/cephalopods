"""kinematics_sample.py — build the behaviour-stratified clip sample for the kinematics×VLM study.

Referee requirements honoured:
  * maximise DISTINCT SOURCE VIDEOS per class (stats are video-level; clip-level would be
    pseudo-replication) — at most `--per-video` clips from any one recording;
  * spread across dates;
  * colour cameras only (the mask model is colour-trained), present clips only, on-disk only;
  * deterministic (seeded) and written to a committed clip list so the run is reproducible.

Usage: venv/bin/python3 src/kinematics_sample.py --per-class 25 --out data/kinematics_sample.json
"""
import argparse, collections, json, random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BR = REPO / "data" / "behaviour_records.json"
CAMS = ("Right_Front", "Right_Back", "Right_Right")


def source_video(rel):
    p = Path(rel); return f"{p.parent.parent.name}/{p.parent.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--per-video", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=str(REPO / "data" / "kinematics_sample.json"))
    args = ap.parse_args()

    br = json.load(open(BR))
    pool = collections.defaultdict(list)
    for rel, rec in br.items():
        st = rec.get("struct") or {}
        if not st.get("present"):
            continue
        beh = st.get("behavior")
        if not beh or beh == "uncertain":
            continue
        if not any(c in rel for c in CAMS):
            continue
        if not (REPO / "src" / rel).exists():
            continue
        pool[beh].append(rel)

    rng = random.Random(args.seed)
    chosen, stats = [], {}
    for beh, rels in sorted(pool.items(), key=lambda kv: -len(kv[1])):
        byvid = collections.defaultdict(list)
        for r in rels:
            byvid[source_video(r)].append(r)
        vids = sorted(byvid)
        rng.shuffle(vids)
        picked, per_vid = [], collections.Counter()
        # round-robin over videos so we spend the budget on breadth, not one recording
        for rnd in range(args.per_video):
            for v in vids:
                if len(picked) >= args.per_class:
                    break
                cands = [r for r in byvid[v] if r not in picked]
                if not cands or per_vid[v] > rnd:
                    continue
                picked.append(rng.choice(cands)); per_vid[v] += 1
            if len(picked) >= args.per_class:
                break
        chosen += picked
        stats[beh] = {"clips": len(picked), "videos": len({source_video(r) for r in picked}),
                      "pool_clips": len(rels), "pool_videos": len(vids)}

    json.dump({"seed": args.seed, "per_class": args.per_class, "per_video": args.per_video,
               "cameras": list(CAMS), "stats": stats, "clips": chosen},
              open(args.out, "w"), indent=1)
    tot_v = len({source_video(r) for r in chosen})
    print(f"sampled {len(chosen)} clips / {tot_v} distinct videos -> {args.out}")
    for b, s in stats.items():
        print(f"  {b:32s} {s['clips']:3d} clips / {s['videos']:3d} videos   "
              f"(pool {s['pool_clips']}/{s['pool_videos']})")


if __name__ == "__main__":
    main()
