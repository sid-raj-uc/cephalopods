# Experiment 1: GroundingDINO Zero-Shot Detection on 985–1045s Candidate Clips

**Date:** 2026-06-10  
**Script:** `phase2/exp01_gdino_detection.py`  
**Cameras:** Left_Top, Right_Front (low-bias cameras)  
**Clip:** `data/aquarium/clips/2026-02-20/095420/985_1045/` (60s, the strongest prior candidate)

---

## What We Did

Ran GroundingDINO (SwinT) with query `"octopus"` at 1fps on both clips.
- 60 frames × 2 cameras = 120 frames processed
- Thresholds: box=0.25, text=0.20 (deliberately low to catch partial views)
- Saved top-6 globally ranked frames with bounding boxes drawn

---

## Results

**Total frames with ≥1 detection:** 117 / 120 (nearly all frames trigger at low threshold)

### Top detections:

| Rank | Camera | Time | Score | Assessment |
|---|---|---|---|---|
| 1 | Right_Front | t=21s | **0.812** | ❌ False positive — aquarist arm outside tank (blurry white object at left edge) |
| 2 | Right_Front | t=40s | 0.592 | ⚠️ Blob inside tank, mid-upper area |
| 3 | Right_Front | t=39s | 0.590 | ⚠️ Same blob, same location |
| 4 | Right_Front | t=41s | 0.579 | ⚠️ Same blob |
| 5 | Right_Front | t=38s | 0.579 | ⚠️ Same blob |
| 6 | Right_Front | t=37s | 0.570 | ⚠️ Same blob |
| 7 | Left_Top    | t=39s | 0.552 | ⚠️ Detection in overhead view, uncorroborated |

### Zoomed crop of Right_Front t=39s detected region:

The persistent cluster at **t=35–58s** on Right_Front all land on the same pixel location (cx≈0.61, cy≈0.64 in normalized coords → pixel box ~459,257→523,318 at 800px). The zoomed crop shows a **rounded, blob-like shape** in the mid-tank water column — organic in appearance and consistent across 20+ consecutive seconds.

---

## Did It Work?

**Partially.** Key findings:

1. **Rank 1 is a false positive.** Score 0.812 fired on the aquarist's arm/clothing at the left edge of frame — not inside the tank. GDino triggered on a high-contrast blurry blob near the camera edge. This is the same human false positive pattern we've seen before.

2. **The t=35–58s cluster on Right_Front is the interesting signal.** Scores 0.44–0.59, consistently in the same tank location, with a rounded blob shape in the zoomed crop. This could be:
   - **Octopus mantle** resting against the back wall of the tank (most likely given the shape and consistency)
   - A decoration or equipment artifact (possible — tank has pipes and objects)
   - Water reflection/glare (unlikely — stable across 20+ seconds)

3. **Left_Top is noisy.** Detections spread uniformly across the whole 60s with no strong cluster — suggests mostly false positives on rocks/substrate pattern.

---

## Conclusion

GDino zero-shot at this threshold is too noisy (117/120 frames trigger). The high-value finding is the **persistent blob at t=35–58s on Right_Front** (absolute video time t=1020–1043s).

**UPDATE (post Exp 2):** High-resolution zoom of the detected region at t=1025s and t=1035s confirms an organic, mottled blob with texture that changes between frames — consistent with octopus chromatophore activity. **This is likely Nity**, mantle pressed against the back wall of the tank during the aquarist interaction event.

---

## Next Step → Experiment 2

Run motion-based blob detection on Left_Top and Right_Front across the **full 30-min video** (`data/aquarium/full/2026-02-20/095420/`) to find windows where a blob of octopus-like size appears and moves slowly in the substrate area. Motion provides a more reliable signal than GDino zero-shot in IR footage.
