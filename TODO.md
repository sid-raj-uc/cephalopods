# TODO — tank-scope hardening before OCEANS submission

Scope decision (2026-08-20): **we stay in the tank ecosystem.** The wild-footage probe (R21) showed
both presence signals go near-chance out of domain; the pipeline is positioned as a
**tank-instrumentation system**, not a wild-footage tool. No further OOD work.

Ordering rule: things that unblock *your* manual time come first, so your labelling runs in parallel
with my compute. "Owner" = who does the work. Each item states how we know it is DONE.

---

## T1 — Behaviour-label ACCURACY (the paper's biggest hole) — IN PROGRESS
**Why.** R15 measured *consistency only, never accuracy*: `present` κ=0.413 [0.229,0.579],
behavior 0.552, posture 0.511, location 0.511, context 0.585, activity 0.592. Every headline
behavioural claim is bounded by these, and **no field has an accuracy number**. Meanwhile we spent
**513 human labels on masks** and **4 on behaviour** — the human-label budget is in the wrong place
relative to where the claims live.

- [ ] **T1a** (me) stratified sampler: ~200 clips, sampled **BY SOURCE VIDEO**, stratified across the
      7 behaviour classes + cameras, over-sampling rare classes. Frozen list, committed.
- [ ] **T1b** (me) **BLIND** labelling UI. Critical: `review_captions.py` shows the VLM's label, so it
      yields approve/reject with anchoring, NOT independent labels. The human must not see the model's
      answer. Writes `data/behaviour_human_labels.json`.
- [ ] **T1c** (you) label the set, ~20 s/clip ≈ 1.5–2 h.
- [ ] **T1d** (me) accuracy + per-class confusion vs the VLM, CIs clustered by source video; fold into
      the paper as the accuracy figure R15 could not give.
- **DONE when** the paper can state per-field accuracy vs human, not just self-consistency.
- **NOT a fix for** the failed behaviour classifier (45% vs 50% majority). Recorded lesson: static
  pooled CLIP features can't do it; it needs temporal/motion features. Annotation fixes the
  *evaluation*, not that model.

## T2 — Demote or hard-qualify the activity budget (free, writing only)
**Why.** R17: resting ranges **16–73% across the 7 days** — the one headline behavioural finding with
no robust support. Circadian (13.4x, 4/4 days) and stimulus response (7/7 days, sign-test p=0.0078)
both hold; the budget does not.
- [ ] decide: demote from the headline, or state the per-day range inline every time it is quoted.
- **DONE when** no sentence in the paper implies the budget is as solid as the other two findings.

## T3 — Retire the `enrichment_object` contrast (cheap)
**Why.** It fires on ~66% of clips; it means "object in tank", not active enrichment. The clean
stimulus contrast is none-vs-`human_present`, which is already the strong result.
- [ ] drop the enrichment contrast from the results, or rename the field so it cannot be misread.
- **DONE when** the only stimulus claim rests on the human-presence contrast.

## T4 — IR / `Right_Top`: the biggest coverage hole
**Why.** ~1,391 IR clips unlabelled on what the paper calls our largest deployment camera.
GroundingDINO is rarely confident on greyscale (**13% clip acceptance, 87 of 653**) and SAM2
over-segments bright tools (median mask area 8.5% vs 2.9% colour). Needs the Phase-0 IR fix.
- [ ] try: CLAHE/contrast pre-processing into GroundingDINO; IR-specific prompt; brightness-aware
      negative points; lower seed gate + stricter area continuity.
- **DONE when** IR clip acceptance is materially above 13% with mask area in the colour range, or we
  record a measured negative and state IR as out of scope.

## T5 — Cleaner teacher labels for the mask plateau
**Why.** R19 bounds it: the student sits ~0.08 under the teacher's own operating-point quality
(0.6415 vs 0.726), and more clips will NOT move it. Label quality is the lever.
- [ ] **T5a** run the **propagated**-teacher arm on SEG-TEST to bound the true ceiling
      (122 clips x 40 frames = 4,880 GD calls, ~3.4 h locally — run on cbox or Modal, not the Mac).
- [ ] **T5b** if the ceiling is meaningfully above 0.726, re-auto-label at a raised seed gate and retrain.
- **DONE when** we can state the true teacher ceiling, and either beat 0.6415 or explain the residual.

## T6 — Presence gate still attenuates rather than removes
**Why.** At the deployed operating point (area ≥ 0.01), **15.5%** of human-verified empty frames and
**22.2%** of reflection frames still fire.
- [ ] cheap first: re-train the negatives/presence variant on the **merged** diverse set (never done —
      v3 predates the merge).
- **DONE when** FP@deployed drops with present-recall held, or we record the negative.

## T7 — Decide what to do with the 156 newly harvested clips
**Why.** The 2026-2-20 campaign produced 156 clips over dates reaching 2026-04-12, extending the
record ~5 weeks past the analysed corpus. They are raw.
- [ ] **T7a** settle the denominator question FIRST: the circadian curve is *present ÷ all extracted
      windows per hour*, and harvested clips use a different regime (visibility-only gate, 2/video,
      probe-first). They **cannot** be pooled into the existing exposure normalisation.
- [ ] **T7b** if the denominator can be reconstructed, run structured extraction and test whether the
      circadian peak and stimulus contrasts hold outside the original week.
- **DONE when** either a multi-week replication exists, or we record why pooling is unsound.

## T8 — Submission blockers (not research)
- [ ] **confirm the OCEANS 2026 page limit.** v2 is at **8 pages**; v1 was deliberately held at 7,
      which suggests 7 may be the target. If the limit is 6 or 7, v2 needs trimming, not more content.
- [ ] author list + affiliations are still placeholders in v2's header.
- [ ] push the local commits (several unpushed); cbox's checkout is one scp'd file dirty vs `cb28373`.
