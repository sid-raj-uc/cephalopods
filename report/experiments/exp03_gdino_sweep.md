# Experiment 3: GroundingDINO Sweep — All 5 Timestamps, Right_Front, ROI-Filtered

**Date:** 2026-06-10  
**Script:** `phase2/exp03_gdino_sweep.py`  
**Camera:** Right_Front (best camera from Exp 1)  
**Videos:** All 5 full 30-min sessions — 095420, 102421, 112421, 122421, 132421  
**Improvement over Exp 1:** Tank-interior ROI crop (x=0.15–0.85, y=0.28–0.80) to exclude aquarist at frame edges

---

## What We Did

Ran GDino (SwinT, threshold=0.30) at 0.5fps across all five 30-min Right_Front videos.
- ~4500 frames total, ~90 min wall time
- ROI filter: only kept detections whose center fell inside the tank glass area
- Saved top-3 frames per timestamp where score ≥ 0.30

---

## Results

| Timestamp | Session time | Detections | Max score | Verdict |
|---|---|---|---|---|
| 095420 | 09:54 AM | 3 frames at t=1022–1026s | **0.620** | ✅ In-tank blob — confirmed octopus |
| 102421 | 10:24 AM | 1 frame at t=34s | 0.307 | ⚠️ Barely above threshold, unclear |
| 112421 | 11:24 AM | **0** | 0.000 | ❌ Nothing detected |
| 122421 | 12:24 PM | 16 frames at t=526–544s | **0.885** | ❌ Aquarist standing center-frame |
| 132421 | 01:24 PM | **0** | 0.000 | ❌ Nothing detected |

---

## Did It Work?

**No — ROI filtering insufficient.** The 122421 "detections" are all the aquarist standing directly in front of the glass, blocking the center of the frame. The ROI boundary excluded edges (x < 0.15 and x > 0.85) but couldn't exclude a human blocking the center.

The only confirmed in-tank detection remains **095420 t=1022–1026s** from Experiment 1.

### Key pattern
GDino fires reliably (0.80–0.89) whenever the aquarist is at the tank, and fires weakly (0.56–0.62) on the octopus inside the tank during the same event. Outside of human-interaction windows, the octopus is invisible to GDino — camouflage is too effective.

---

## What We Learned

1. **Feeding/interaction events are the best windows to look for Nity** — she comes to the front of the tank during these events and becomes partially visible.
2. **GDino zero-shot cannot separate human from octopus** — humans always outscore the octopus.
3. **112421 and 132421 have zero signal** — Nity was either denning or in the overhead-invisible position during those sessions.
4. **The confirmed octopus crop from 095420 is valuable** — it's a real aquarium-IR exemplar of Nity's body.

---

## Next Step → Experiment 4

Use the confirmed octopus crop as a **DINOv2 exemplar query**:
1. Extract a tight crop of the Nity detection from 095420 t=1024s
2. Compute its DINOv2 patch embedding
3. Slide that query across frames from all sessions — find windows with high patch cosine similarity
4. This sidesteps GDino entirely and doesn't require text-based queries that confuse human with octopus
