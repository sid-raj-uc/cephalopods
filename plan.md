# Cephalopod Behavior Analysis — Revised Plan

## Goal
Classify octopus behavior in video clips according to a predefined ethogram, with no labeled training data and no API budget.

## Pipeline

```
clips → filter (CLIP) → features (DINOv2 + VideoMAE) → cluster (HDBSCAN) → label representatives → propagate labels
```

---

## Stage 1 — Filter clips (CLIP zero-shot)
- Run CLIP on the center frame of each clip
- Compute cosine similarity to `"an octopus"`
- Discard clips below threshold (no octopus present)
- Fast, free, no GroundingDINO or SAM 2 needed
- Mark surviving clips as `detected` in the manifest

## Stage 2 — Feature Extraction
For each detected clip:

**Appearance (DINOv2)**
- Run DINOv2 on each frame
- Mean-pool CLS tokens across all frames
- Output: appearance vector per clip

**Motion (VideoMAE)**
- Run VideoMAE on the frame sequence
- Use the CLS token output
- Output: motion vector per clip

**Fusion**
- Concatenate appearance + motion vectors
- L2-normalize
- Save as `.npy` in `data/features/`
- Register in manifest

## Stage 3 — Clustering (HDBSCAN)
- Load all feature vectors
- Run HDBSCAN (set min_cluster_size based on ethogram size)
- Also try k-means with k = number of ethogram behaviors as a comparison
- Save cluster assignments per clip
- Identify noise points (HDBSCAN label = -1) for separate review

## Stage 4 — Manual Labeling
- For each cluster, pick 3-5 clips closest to the centroid
- Human reviews those clips and assigns an ethogram label
- ~30-50 labeling decisions instead of 400
- Noise/outlier clips reviewed individually

## Stage 5 — Label Propagation
- Assign cluster label to all clips in that cluster
- Flag low-confidence clips (far from centroid) for review
- Output: labeled clip dataset

---

## What was dropped from the original plan
- **SAM 2 segmentation** — not needed, too slow, adds complexity
- **GroundingDINO** — replaced by CLIP for filtering
- **VLM weak labeling (Gemini)** — no API budget; replaced by clustering + manual labeling
- **Continued SSL pretraining** — use pretrained DINOv2 + VideoMAE directly

## Key assumptions
- DINOv2 attention focuses on foreground (octopus) sufficiently without masking
- Behavioral clusters will be coherent enough to label with one ethogram entry per cluster
- If clusters are not behaviorally coherent, revisit with masked features

## Open questions
- Exact ethogram behaviors (needed before labeling step)
- Threshold for CLIP filtering
- Whether to use raw frames or center-crop for DINOv2
