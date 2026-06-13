# Experiment 4: DINOv2 Exemplar Search (CLS Similarity)

**Date:** 2026-06-10  
**Script:** `phase2/exp04_dinov2_exemplar.py`  
**Camera:** Right_Front, all 5 timestamps  
**Query:** Confirmed Nity crop from 095420 Right_Front t=1024s

---

## What We Did

Extracted a ~124×120px crop of the confirmed octopus region (GDino bbox cx=0.614, cy=0.640) from the reference frame. Computed its DINOv2 ViT-B/14 CLS embedding (768-d). Scanned all 5 timestamps at 0.5fps, cropping the same region in every frame and computing cosine similarity to the query vector.

---

## Results

| Timestamp | Max sim | Mean sim | Frames above threshold |
|---|---|---|---|
| 095420 (reference) | **0.9954** | 0.8785 | 896/897 |
| 102421 | 0.9116 | 0.8768 | 724/724 |
| 112421 | 0.8532 | 0.8242 | 715/715 |
| 122421 | 0.8246 | 0.7740 | 470/685 |
| 132421 | 0.7925 | 0.7493 | 120/866 |
| Baseline (empty 132421 t=300s) | 0.7188 | — | — |

---

## Did It Work?

**No.** The approach failed because DINOv2 CLS encodes the **global scene content** of the crop — dominated by the tank background (lights, wall texture, equipment) which is present in all timestamps. Almost every frame scores above threshold because they're all the same camera looking at the same section of tank.

The declining max similarity from 0.9954 → 0.7925 across sessions reflects **time-of-day lighting drift** and scene changes, not octopus presence.

The `"frames_above_threshold=896/897"` for the reference session itself confirms the method isn't discriminative — it can't distinguish the 1 octopus frame from the 896 background frames in the same session.

---

## What We Learned

CLS embedding at this crop scale (~120×120px) doesn't work because:
1. The octopus occupies maybe 20–30% of the crop area
2. The remaining 70–80% is identical background across all frames
3. DINOv2 CLS aggregates globally — background wins

**Fix:** Use a **tighter crop** (<50×50px, just the octopus blob) and compare **patch token** cosine similarity at that exact location rather than CLS.

---

## Next Step → Experiment 5

Tight-patch DINOv2 + pixel residual:
1. Extract only the core octopus blob (~50×50px from the confirmed bbox center)
2. For each candidate frame, compare patch tokens at the same location
3. Additionally subtract a "median background frame" before comparing to suppress static background
