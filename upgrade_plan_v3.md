# Plan: Upgrade Qaida Pronunciation Scoring v3 — Fix Advanced Lessons

## Context
After v2 (XLSR-53 + SupCon+Triplet, see `upgrade_plan_v2.md`), the system reaches **100% Top-1 / Top-3 / MRR on the studio val split** and works well on real microphone input for **Lesson 1 (alphabet)**, where items have distinct base letters (ز vs ض vs غ).

**Problem.** Real-mic accuracy collapses on advanced lessons:
- **L3 harakat** (بَ vs بِ vs بُ — same consonant, different ~80 ms vowel)
- **L4 long voices** (ثٰ vs ثٖ vs ثٗ — vowel length distinctions)
- **L5 tanween** (ـً ـٍ ـٌ — same consonant, different ending)
- **L9 madd** (سَا vs سَ — vowel duration)
- **L10 leen letters** (ـَيْ vs ـَوْ — diphthong direction)

**Root causes.**
1. **Mean-pooling kills harakat.** v2 mean-pools the entire utterance into one 128-d vector. The discriminating signal for harakat is concentrated in ~80–150 ms of vowel; averaging over a 500 ms+ clip dilutes it.
2. **Single-speaker references.** All 692 references come from one teacher. Embeddings encode that speaker's pitch/timbre as much as the phoneme — different student voices push examples across the decision boundary.
3. **Studio-vs-real-mic domain gap.** v2 augmentations (pitch/speed/noise/bandpass/reverb) approximate but don't match real laptop/phone mics, codec compression, room conditions, child voices.
4. **No length normalization on user input.** A user's recording with 500 ms of leading silence drags the mean pool toward "silence", flattening the embedding.
5. **Global thresholds.** `PASS=0.6 / GOOD=0.8` are not calibrated per lesson — L10 needs a tighter margin than L1.

## What Changes

| Tier | Change | Retraining? | Effort | Expected impact |
|------|--------|-------------|--------|-----------------|
| 1 | VAD + silence trim, length-matching, lesson-restricted retrieval, per-lesson thresholds | No | ~2 h | Medium — fixes "wrong silence dominates pool" failures |
| 2 | Frame-level DTW pooling instead of mean-pool; multi-prototype references | No | ~1 day | **High** — directly addresses harakat dilution |
| 3 | Auxiliary phoneme-CTC head; stronger real-mic augmentation; Arabic-specialist backbone option | Yes | 1–3 days | High — forces encoder to keep harakat detail |
| 4 | Collect multi-speaker student recordings per advanced item | Yes | ongoing | Decisive — current single-speaker ceiling |

## Execution Steps

### Tier 1 — Inference-side quick wins (no retraining)

#### Step 1.1: VAD + silence trim
**File**: `qaida_project/embeddings/embedding_model.py` — `extract()`
- After `librosa.load`, apply energy-based silence trim (`librosa.effects.trim(top_db=30)`) or webrtcvad if available.
- Apply to *both* reference (during embedding extraction) and student (during inference) for consistency.

#### Step 1.2: Length matching to reference
**File**: `qaida_project/inference/acoustic_scorer.py` — `compute_score()`
- Compute reference duration. If student is >1.3× or <0.7× reference duration, time-stretch to fit (librosa.effects.time_stretch). This is critical for L9 (madd) where length *is* the signal — but normalizing length first makes other lessons more robust.

#### Step 1.3: Lesson-restricted retrieval in demo
**File**: `qaida_project/scripts/gradio_demo.py`
- Already has `lesson_num` selected. Pass `lesson_num` filter into the scorer so cosine search runs only against same-lesson references.
- Report the **in-lesson margin** (top1 − top2) explicitly in the UI. Users learn to trust the score.

#### Step 1.4: Per-lesson thresholds
**File**: `qaida_project/config/settings.py`, `inference/acoustic_scorer.py`
- Replace global `PASS_THRESHOLD / GOOD_THRESHOLD` with a `LESSON_THRESHOLDS` dict, calibrated from `eval_acoustic_xlsr-finetuned.json`:
  - L1, L11: pass 0.55, good 0.75 (large margins)
  - L2, L13, L14, L15: pass 0.60, good 0.80
  - L3, L4, L5, L9: pass 0.65, good 0.85 (subtle harakat)
  - L10, L16: pass 0.70, good 0.88 (very tight margins — leen, hurof mukatat)

### Tier 2 — Frame-level DTW + multi-prototype (no retraining of XLSR)

#### Step 2.1: Frame-level embeddings
**File**: `qaida_project/embeddings/embedding_model.py` — new method `extract_frames()`
- Skip the projection head's mean-pool. Apply the projection head per-frame (or use the encoder's `last_hidden_state` directly), L2-normalize each frame.
- Returns `(T_frames, 128)` instead of `(128,)`.

#### Step 2.2: Cosine-DTW similarity
**File**: `qaida_project/inference/acoustic_scorer.py` — new function `dtw_similarity(ref_frames, stu_frames)`
- Compute frame-by-frame cosine distance matrix.
- Run DTW (use `librosa.sequence.dtw` or `dtaidistance`) to find best monotonic alignment.
- Score = `1 - (DTW_path_cost / path_length)`, normalized to [0, 1].
- This preserves the harakat region — the alignment-aware path penalizes mismatched vowels even when the overall consonant is correct.

#### Step 2.3: Hybrid scoring
- Final score = `0.4 * mean_pool_cosine + 0.6 * dtw_similarity`
- Mean-pool is still useful for "wrong letter entirely" cases; DTW handles "right letter, wrong harakat".

#### Step 2.4: Multi-prototype references
- For each `audio_id`, store **3 prototype embeddings** instead of 1:
  - Original reference
  - Reference + pitch-shift +2 semitones (simulate higher voice / child)
  - Reference + pitch-shift -2 semitones (simulate lower voice / adult)
- Score = max similarity over the 3 prototypes.
- File: `embeddings/extract_embeddings.py` — generate `_p0`, `_p1`, `_p2` keys. `inference/acoustic_scorer.py` — `max(sim(stu, ref_p0), sim(stu, ref_p1), sim(stu, ref_p2))`.

### Tier 3 — Retraining with phoneme awareness

#### Step 3.1: Auxiliary phoneme-CTC head
**File**: `qaida_project/training/wav2vec2_contrastive.py`
- Add an Arabic phoneme classification head (CTC) on top of XLSR-53's frame-level output, parallel to the projection head.
- Phoneme labels: derive from Arabic transcript using a phonemizer (e.g. `epitran`, `phonemizer-arabic`, or hand-mapped Buckwalter).
- Joint loss: `loss = supcon + 0.5*triplet + 0.3*ctc_phoneme`
- The CTC objective forces the encoder to retain *frame-level* phoneme information, which directly fixes harakat dilution.

#### Step 3.2: Stronger real-mic augmentation
**File**: `qaida_project/training/wav2vec2_contrastive.py` — `augment_audio()`
- Add real-world impulse responses (free OpenAIR / MIT IR Survey) — replace the synthetic exponential reverb.
- Add lossy codec round-trip simulation (encode-decode through Opus 16 kbps via `pydub` or `ffmpeg-python`).
- Add real background noise from MUSAN (TV, babble, keyboard).
- Add RIR-convolution from BUT Reverb DB.
- Probabilities: 0.4 each, applied independently.

#### Step 3.3: Arabic-specialist backbone option
- Add config flag `BACKBONE_VARIANT = "xlsr53"` (current) | `"arabic_xlsr"` | `"mms_arabic"`.
- `arabic_xlsr`: `elgeish/wav2vec2-large-xlsr-53-arabic` — already CTC-trained on Arabic, so the encoder representations are phoneme-aware before contrastive fine-tuning.
- `mms_arabic`: `facebook/mms-1b-fl102` (Arabic head) — newer, generally stronger.
- Train one of each and pick the best on the *advanced-lessons-only* val subset.

#### Step 3.4: Validation by lesson group
**File**: `qaida_project/training/wav2vec2_contrastive.py` — `validate()`
- Currently reports overall Top-1/Top-3. Split into:
  - Easy group (L1, L2, L11, L13–15)
  - Hard group (L3, L4, L5, L9, L10)
- Save best by **hard-group Top-1**, not overall — overall is already saturated.

### Tier 4 — Data collection (the real ceiling)

#### Step 4.1: Multi-speaker references
- Recruit 3–5 additional Qaida teachers (mixed male/female/age) to re-record all 692 items. Even 3 speakers × 692 = 2076 references and 4× the in-class positives for SupCon.
- Critical for the "same letter, different voices, same harakat" cluster the model must learn.

#### Step 4.2: Real student recordings
- Add an in-app "donate recording" toggle. Each correct attempt (verified by teacher or by high cross-prototype margin) becomes a new positive.
- Active learning loop: every 50 new recordings, retrain projection head only (not XLSR-53), takes ~30 min.

#### Step 4.3: Hard-negative mining from confusion matrix
- Run inference on a held-out hard-cases set, log all confused pairs (predicted ≠ correct).
- Add explicit triplets `(anchor=correct, positive=correct, negative=confused)` to the next training batch with higher weight.

## Files Summary

| File | Change | Tier | Priority |
|------|--------|------|----------|
| `embeddings/embedding_model.py` | VAD/silence trim; new `extract_frames()` | 1, 2 | P0 |
| `inference/acoustic_scorer.py` | Length-match; DTW similarity; hybrid score; per-lesson thresholds | 1, 2 | P0 |
| `scripts/gradio_demo.py` | Lesson-restricted retrieval; show in-lesson margin | 1 | P0 |
| `config/settings.py` | `LESSON_THRESHOLDS` dict; backbone variant flag | 1, 3 | P0 |
| `embeddings/extract_embeddings.py` | Multi-prototype (pitch-shift) refs | 2 | P1 |
| `training/wav2vec2_contrastive.py` | Phoneme-CTC head; stronger augmentation; per-group validation | 3 | P1 |
| `data/prepare_dataset.py` | Phoneme-label generation from Arabic transcript | 3 | P1 |
| `data/` | Multi-speaker recording set | 4 | P2 (ongoing) |
| `evaluation/evaluate_acoustic.py` | Split by easy/hard lesson group; real-mic eval set | 1, 3 | P0 |

## Estimated Timeline
- **Tier 1**: ~2 h. Ship same-day.
- **Tier 2**: ~1 day (frame-level extraction + DTW + hybrid score + multi-prototype).
- **Tier 3**: ~3 days (phoneme labeling + retrain w/ CTC + Arabic backbone trial).
- **Tier 4**: ongoing — needs human recording effort.
- **Total to a usable advanced-lesson system: ~5 days** + data collection.

## Verification

1. **Hard-cases real-mic eval set**: record 50 new student samples covering L3/L4/L5/L9/L10. This is the test that matters — studio val is already saturated.
2. **Per-lesson Top-1 on real-mic set** before/after each tier:
   - Baseline (v2 today): record numbers for L1, L3, L5, L9, L10.
   - After Tier 1: target +10–15 pts on hard lessons.
   - After Tier 2: target +20–25 pts on hard lessons.
   - After Tier 3: target ≥80% on all advanced lessons.
3. **Gradio behavior tests**:
   - بَ recorded → returns بَ with score >0.7, top-2 margin >0.05
   - بِ recorded against بَ reference → returns بِ correctly (cross-harakat detection)
   - User says nothing / says wrong letter → score <0.4, low confidence flag

## Notes / Open Questions
- Should we drop Whisper entirely? Currently `STT_WEIGHT=0.0`, code is dead weight. Decide before v3 ships.
- Is the API consumer OK with per-lesson scoring contract change (return `lesson_pass` instead of global pass)? Coordinate with frontend.
- Phoneme inventory: Buckwalter vs IPA vs Quranic-specific (with hamzat-wasl, sukoon, shaddah as own labels)? Quranic-specific is more accurate but harder to source labels for.
- Multi-prototype storage: 3× embedding count is fine in pkl (~few MB), but if we go to 5+ prototypes per item, switch to a proper vector index (FAISS).
