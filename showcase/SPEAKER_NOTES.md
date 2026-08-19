# Nity — Octopus Behaviour Intelligence · Speaking Notes

Walk the showcase page top-to-bottom. Live demos are pre-warmed. ~8–12 min.

---

## 0 · Open (15s)
- "We taught a machine to read an octopus — to turn months of raw aquarium footage into a quantified record of one animal's behaviour."
- "Frontier models teach; tiny distilled models do the work on a laptop."
- Point at hero stats: **3,205 clips → behaviour records · ~209 days of footage · a 3.2M-param edge model · 5 live tools running right now.**

## 1 · The problem (20s)
- "Aquariums run cameras 24/7 — almost none of that footage is ever analysed. Welfare is still guessed at by eye."
- "And the octopus is the hardest possible subject: soft-bodied, camouflaging, hides in a den, and the cameras are full of reflections."

## 2 · The pipeline — five stages, one loop (30s)
- Detect → Extract → Caption → Structure → Segment.
- "The whole design is **distillation**: GroundingDINO, SAM2 and a 235-billion-param VLM never ship — they only label data. What we deploy is a 3.2M segmenter and a 1.7GB caption model that runs with no GPU."

## 3 · Dataset (40s)
- "Five fixed cameras on the den. The catch — **colour is camera-gated**: three are colour, one pure infrared, one is a reflection artefact we throw away."
- Numbers: **3,986 verified clips · ~9,600 labelled frames · 3,205 behaviour records · 4,412 image-mask pairs.**
- Honest point: "We only used **7 days** of the ~209 available — every model is footage-diversity-limited, which is why we built a streaming harvester to mine the rest."

## 4 · Detection (40s)
- "A frozen CLIP backbone plus a tiny MLP probe — visible vs hidden."
- **The insight that mattered:** "CLIP's default crop throws away 33–44% of a wide frame — and the octopus is usually at the edge. We switched to letterbox, keep the whole frame. That one preprocessing change fixed field performance. Not architecture — preprocessing."
- **The lesson:** "We mined the model's confident mistakes as hard negatives — but verified them by hand. Of 232, **166 actually had the octopus** — the model was right. Only 66 were real negatives. Training on the unverified set gave a higher score and a worse model. Always verify labels."
- Result: hidden false-positives **24% → 3%**, recall held at 0.97.

## 5 · Extraction pipeline (30s)
- "Two gates: the octopus is visible in >50% of frames, AND there's real motion. Slide a 20-second window, keep the good ones."
- **The bug:** "Our first motion metric was normalized per-video — so a static tank with a flickering IR lamp normalized up to 'motion' and passed. Fix: absolute changed-pixel fraction, and mask out the ticking clock."
- **The kicker:** "When we ran the 235B VLM over our 'verified' clips, **63% had no octopus** — the detector was firing on reflections. A VLM is a better presence filter than the detector — which is exactly what the segmentation gate later automates."

## 6 · Captioning (40s)  ▶ DEMO :8000, then :8017
- "Each clip gets a one-sentence caption and a 7-class behaviour label. A large VLM writes the training signal; we distil it into a 2B model."
- Distillation lift: embedding similarity **0.70 → 0.83**, ROUGE-L **0.27 → 0.46**.
- "Deployed as a 1.7GB 4-bit model — **captions a clip in ~3 seconds on a 16GB Mac, no CUDA.**"
- **▶ Open :8000 (Captioning)** — "here's a clip next to the caption and behaviour label the model generated."
- **▶ Open :8017 (Base vs our model vs 235B)** — "and here's the proof it worked: the base model, our distilled student, and the 235B teacher side by side, on the exact frames the model saw."

## 7 · Segmentation (40s)  ▶ DEMO :8012
- "Pixel masks — deliberately the smallest model that still works."
- "Teacher: GroundingDINO finds the box, SAM2 makes the mask, propagated across the whole clip for temporal consistency. Student: a 3.2M-param LR-ASPP."
- **The deployment win:** "Trained with negatives, the mask becomes a presence detector — **AUC 0.86, and 0.99 against reflections**, zero reflection false-positives. It beats the CLIP gate exactly where it was weakest."
- **Be honest:** "Mask IoU plateaus around 0.47 — a video-diversity gap, only 62 distinct training videos. Not an architecture problem; it needs more footage."
- **▶ Open :8012 (Segmentation)** — "the tiny model overlaying the octopus mask and its live area percentage."

## 8 · Findings — the payoff (40s)  ▶ DEMO :8011
- "The point was never a better captioner — it's a behavioural time-series."
- Activity budget: **41% exploration, 33% resting, 14% human interaction, 9% reaching out.**
- Circadian: near-zero overnight, climbing to a **45% peak at 17:00.**
- **The headline result:** "When a human is present, the octopus's movement **nearly doubles** — 0.045 to 0.095 — and a transparent arousal index goes 0.46 to 0.68. A measurable, repeatable welfare signal, extracted with **zero manual labelling.**"
- **▶ Open :8011 (End-to-end run)** — "full 30-minute video on the left, the captions the pipeline produced synced on the right — click any caption to jump there."

## 9 · Presence data (optional demo)  ▶ :8004
- **▶ Open :8004** if asked about data quality — "the frame-level octopus / no-octopus review set behind the detector."

## 10 · Models & close (20s)  ▶ HF links
- "Everything's open." — click **Open models on Hugging Face** (bottom of page).
  - Detector + segmenter: `huggingface.co/sidraj000/octopus-nity-segmentation`
  - Caption model: `huggingface.co/sidraj000/octopus-nity-caption-qwen3vl2b-mlx-4bit`
- Close: "One octopus, months of footage, a fully-distilled pipeline that runs on a laptop — and a welfare signal you couldn't get by watching. This generalises to any animal on a camera. Selected to present at IEEE OCEANS 2026."

---

## Live demo quick-reference
| Say it during | Port | Open | One-liner |
|---|---|---|---|
| Captioning | **:8000** | http://localhost:8000 | clip + generated caption/label |
| Captioning proof | **:8017** | http://localhost:8017 | base vs our model vs 235B |
| Segmentation | **:8012** | http://localhost:8012 | mask overlay + live area% (pre-rendered) |
| Findings | **:8011** | http://localhost:8011 | full video + synced captions |
| Data quality (opt) | **:8004** | http://localhost:8004 | octopus/no-octopus review set |

**If a server is down:** re-run `bash scratchpad/launch_uis.sh` isn't in the repo — the 5 are already up; if the machine slept, ask me to relaunch. **8012 is pre-rendered**, so it plays instantly; the others are light.

## Numbers cheat-sheet
- Footage ~209 days · used 7 · 5 cameras (3 colour / 1 IR / 1 reflection-dropped)
- 3,986 clips · ~9,600 frames · 3,205 records (3,083 present) · 4,412 masks
- Detector: CLIP ViT-B/32 + MLP, letterbox · 96.8% · FP 24%→3% · recall 0.97 · hard-neg 232→66
- Caption student: Qwen3-VL-2B QLoRA · emb-sim 0.70→0.83 · 1.7GB MLX · ~3s/clip, no GPU
- Segmenter: LR-ASPP 3.2M · presence AUC 0.86 / 0.99 vs reflections / 0% reflection FP · mask IoU ~0.47
- Behaviour: 41/33/14/9/2/1 budget · circadian peak 45%@17:00 · human presence motion ×2 (0.045→0.095), arousal 0.46→0.68
