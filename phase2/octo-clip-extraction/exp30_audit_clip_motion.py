"""
Exp 30 — Re-audit the verified clips with the new absolute motion method
(scan_motion_area). Flags which clips have REAL motion vs. passed the old
normalized gate on flicker.

Survive rule: mean changed-pixel fraction >= SURV_THRESH at pix_thresh=PIX, fps=FPS.
Writes data/clips_motion_audit.json (all per-clip values + survive flag) and prints
a per-camera summary.

Usage: venv/bin/python3 phase2/exp30_audit_clip_motion.py
"""
import sys, json, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]   # repo root (file is phase2/octo-clip-extraction/)
from motion_detector import scan_motion_area   # same folder; on sys.path as the script dir

CLIPS_DIR = PROJECT / "data" / "octopus_clips_verified"
OUT_JSON  = PROJECT / "data" / "clips_motion_audit.json"

FPS         = 5.0
PIX         = 25
SURV_THRESH = 0.005   # mean changed-pixel fraction (0.5% of frame)


def main():
    clips = sorted(CLIPS_DIR.rglob("*.mp4"))
    print(f"{len(clips)} verified clips to audit  (fps={FPS}, pix_thresh={PIX}, survive>= {SURV_THRESH})")
    print("-" * 64)

    results, survivors = [], []
    tot = defaultdict(int); surv = defaultdict(int)

    for i, c in enumerate(clips, 1):
        cam = c.stem.rsplit("_", 1)[0]
        tot[cam] += 1
        _, area = scan_motion_area(str(c), fps=FPS, pix_thresh=PIX)
        if len(area) == 0:
            continue
        mean_a = float(area.mean()); max_a = float(area.max())
        frac1  = float((area > 0.01).mean())
        survived = mean_a >= SURV_THRESH
        rec = {"clip_path": str(c.relative_to(PROJECT)),
               "camera": cam, "date": c.parts[-3], "segment": c.parts[-2],
               "mean_area": round(mean_a, 5), "max_area": round(max_a, 5),
               "frac_frames_gt1pct": round(frac1, 3), "survive": survived}
        results.append(rec)
        if survived:
            survivors.append(rec["clip_path"]); surv[cam] += 1

        if i % 50 == 0 or i == len(clips):
            json.dump({"method": "scan_motion_area", "fps": FPS, "pix_thresh": PIX,
                       "survive_threshold": SURV_THRESH,
                       "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                       "count": len(results), "survivors": len(survivors),
                       "clips": results},
                      open(OUT_JSON, "w"), indent=1)
            print(f"  [{i}/{len(clips)}]  survivors so far: {len(survivors)}", flush=True)

    # also write a plain survivor path list
    (PROJECT / "data" / "clips_motion_survivors.txt").write_text("\n".join(survivors) + "\n")

    print("-" * 64)
    print(f"Audited {len(results)} clips -> {OUT_JSON.relative_to(PROJECT)}")
    print(f"SURVIVORS (real motion): {len(survivors)} / {len(results)}")
    print(f"\n{'camera':12s}{'total':>7s}{'survive':>9s}{'drop%':>8s}")
    for cam in sorted(tot):
        dropped = tot[cam] - surv[cam]
        print(f"{cam:12s}{tot[cam]:7d}{surv[cam]:9d}{100*dropped/tot[cam]:7.0f}%")
    print(f"{'TOTAL':12s}{sum(tot.values()):7d}{sum(surv.values()):9d}"
          f"{100*(sum(tot.values())-sum(surv.values()))/max(1,sum(tot.values())):7.0f}%")
    print(f"\nsurvivor list -> data/clips_motion_survivors.txt")


if __name__ == "__main__":
    main()
