# Experiment 2: MOG2 Background Subtraction on Left_Top Full 30-min Video

**Date:** 2026-06-10  
**Script:** `phase2/exp02_blob_motion.py`  
**Camera:** Left_Top (overhead)  
**Video:** `data/aquarium/full/2026-02-20/095420/Left_Top.mp4` (30 min)

---

## What We Did

Used OpenCV MOG2 background subtractor at 2fps on the full 30-minute session.
- MOG2 history=200 frames to learn stable background (tank walls, substrate, equipment)
- Filtered blobs by octopus-plausible size: 0.3%–15% of frame area
- Processed 3595 frames (~30 min × 2fps), took ~8 min

---

## Results

**Timeline:** One dominant event at t=17:04 (1024s), everything else near zero.

| Rank | Time | Score | Assessment |
|---|---|---|---|
| 1 | 17:04 (1024s) | **0.824** | ❌ Aquarist leaning over the tank — large human blob fills foreground |
| 2 | 5:28 (328s) | 0.051 | Tiny transient, likely water disturbance |
| 3–6 | various | <0.005 | Background noise |

---

## Did It Work?

**No — the octopus is invisible to overhead MOG2.**

Root cause: Nity is camouflaged against the substrate. The overhead Left_Top camera shows the tank floor where the octopus blends into rocks and sand. MOG2 cannot separate a camouflaged animal from the background it evolved to mimic.

The only detectable event is the aquarist entering the frame at 17 min, which also corresponds to the 985–1045s clip (t=985s ≈ 16:25 from session start). This confirms that window is a **feeding/interaction event** — the aquarist is actively working with the tank.

---

## Key Cross-Experiment Finding

Triangulating Exp 1 and Exp 2:
- GDino on Right_Front at t=35–58s in the 985_1045 clip (= absolute t=1020–1043s) → persistent blob, score 0.44–0.59 ✅
- MOG2 on Left_Top full video → aquarist at t=1024s ✅
- High-resolution manual inspection of Right_Front at t=1025s and t=1035s → **organic mottled blob consistent with octopus mantle visible against back wall**

**Verdict: Nity is visible in Right_Front during the t=1020–1043s window.** The body/mantle is pressed against the back wall in the mid-tank water column. The texture pattern changes between frames — consistent with active chromatophore display during human interaction.

---

## Next Step → Experiment 3

Scan all five timestamps' Right_Front videos with GDino to find more confirmed windows beyond this single 60s session. Also extract clean high-resolution crops of confirmed frames for use as positive training examples.
