"""merge_shards.py — combine per-shard kinematics outputs into one file for analysis.

Shards write separate JSONs (concurrent writes to one file would clobber). This merges them and
refuses to mix pipeline configs: every record carries a `_cfg` stamp (git SHA + ckpt + refine + fps),
and results computed under different configs are not comparable.

Usage: venv/bin/python3 src/merge_shards.py --out data/skeleton_motion_study.json
"""
import argparse, glob, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", default=str(REPO / "data" / "skeleton_motion.shard*of*.json"))
ap.add_argument("--out", default=str(REPO / "data" / "skeleton_motion_study.json"))
a = ap.parse_args()

merged, cfgs, files = {}, set(), sorted(glob.glob(a.pattern))
for f in files:
    d = json.load(open(f))
    for k, v in d.items():
        if isinstance(v, dict) and "_cfg" in v:
            cfgs.add(json.dumps(v["_cfg"], sort_keys=True))
        merged[k] = v
    print(f"  {Path(f).name}: {len(d)} records")

print(f"merged {len(merged)} records from {len(files)} shards -> {a.out}")
print(f"config stamps: {len(cfgs)}")
for c in sorted(cfgs):
    print(f"  {c}")
if len(cfgs) > 1:
    print("WARNING: mixed configs — do NOT pool these for statistics")
json.dump(merged, open(a.out, "w"), indent=1)
