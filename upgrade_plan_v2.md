# Plan: Upgrade Qaida Pronunciation Scoring — XLSR-53 + Better Training

## Context
The v1 contrastive system (wav2vec2-base, 500 steps, NT-Xent) is working but accuracy is poor:
- **Detection wrong** even for correct pronunciation — model picks wrong reference item
- **Scores unreliable** — thresholds are arbitrary, not calibrated
- **Root cause**: wav2vec2-base was pre-trained on **English speech only** — it doesn't understand Arabic phonemes. The contrastive training can't overcome this fundamental limitation with only 500 steps.

**Solution**: Switch backbone to `facebook/wav2vec2-large-xlsr-53` (pre-trained on 53 languages **including Arabic**), use better loss function, stronger augmentations, and longer training.

## What Changes

| Area | Current (v1) | Upgraded (v2) |
|------|-------------|---------------|
| Backbone | wav2vec2-base (English, 95M, 768-dim) | **XLSR-53 (Arabic+52 langs, 315M, 1024-dim)** |
| Loss | NT-Xent (needs huge batches) | **SupCon + Hard Negative Triplet** |
| Projection | 768→256 | **1024→512→128** (with BatchNorm) |
| Augmentations | 4 types (pitch, speed, noise, volume) | **7 types** (+bandpass, reverb, silence) |
| Frozen layers | 6/12 | **16/24** |
| Training steps | 500 | **2000** |
| Optimizer | AdamW fp32 | **PagedAdamW 8-bit** |
| Grad checkpointing | No | **Yes** |
| Validation | None | **Top-1/Top-3 accuracy during training** |
| VRAM usage | ~1.5GB | **~1.4GB** (fits fine in 4GB) |

## Execution Steps

### Phase 0: Baseline Evaluation (before any changes)
**NEW file**: `qaida_project/evaluation/evaluate_acoustic.py`
- Run current model on 104 val samples
- For each: extract embedding, find rank of correct item within same lesson
- Report: Top-1 accuracy, Top-3 accuracy, per-lesson breakdown
- Also test pretrained XLSR-53 **without** fine-tuning (zero-shot baseline)
- This gives us numbers to compare against

### Phase 1: Download XLSR-53
```bash
py -3.12 -c "from transformers import Wav2Vec2Model, Wav2Vec2Processor; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53'); Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53')"
```
~1.2GB download, cached in HuggingFace cache.

### Phase 2: Update Config
**File**: `qaida_project/config/settings.py`
- Add `XLSR53_MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"`
- Add `XLSR53_FINETUNED_DIR` path
- Point active embedding model to XLSR-53

### Phase 3: Rewrite Training Script
**File**: `qaida_project/training/wav2vec2_contrastive.py` (major rewrite)

Architecture:
```
XLSR-53 encoder (24 layers, layers 0-15 frozen, 16-23 trainable)
  → mean pool over time → 1024-dim
  → Linear(1024, 512) + BatchNorm + ReLU + Linear(512, 128)
  → L2 normalize
```

**Loss function** — Combined SupCon + Hard Negative Triplet:
- SupCon: augmented views of same audio are positives, all others negative
- Hard Negative Triplet: within same lesson, push same-base-letter items apart (margin=0.3)
- Combined: `loss = supcon_loss + 0.5 * triplet_loss`

**New augmentations** (on top of existing 4):
- Bandpass filter (100-300Hz to 4-7kHz) — simulates phone/laptop mic (40% prob)
- Room reverb simulation (50-150ms decay) — simulates room acoustics (30% prob)
- Random silence insertion (50-200ms) — simulates hesitation (20% prob)

**Training config**:
- Batch size: 16 (×2 views = 32 embeddings)
- LR: 2e-5 (lower for larger model)
- Steps: 2000
- Optimizer: PagedAdamW8bit (saves ~400MB VRAM)
- Gradient checkpointing: enabled
- Temperature: 0.1
- Warmup: 100 steps, cosine decay
- Validation every 200 steps (Top-1/Top-3 on val set)
- Save best model by Top-1 accuracy

**Lesson-aware batch sampler**:
- Each batch: half from one lesson, half from others
- Ensures within-lesson hard negatives appear every batch
- Cycles through all 16 lessons

### Phase 4: Update Embedding Model
**File**: `qaida_project/embeddings/embedding_model.py`
- Read `hidden_size` from model config dynamically (not hardcoded 768)
- Update default model name to XLSR-53
- `ContrastiveWav2Vec2` already parameterized — just ensure config.json drives it

### Phase 5: Train (~4-5 hours)
```bash
cd C:\Users\PC\Desktop\Qaida_audios
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 py -3.12 -u -m qaida_project.training.wav2vec2_contrastive
```
Monitor: loss should drop from ~4.0→<1.0, val Top-1 should climb to 70%+

### Phase 6: Re-extract Embeddings
**File**: `qaida_project/embeddings/extract_embeddings.py`
- Backup old embeddings first
- Re-extract all 692 with new XLSR-53 model → 128-dim vectors

### Phase 7: Recalibrate Scoring
**File**: `qaida_project/inference/acoustic_scorer.py`
- Run evaluation to see new similarity distribution
- Adjust absolute score mapping and blend ratio based on real numbers
- XLSR-53 should have wider margin between correct/incorrect → more reliable absolute scores

### Phase 8: Evaluate & Test
- Run `evaluate_acoustic.py` with new model
- Compare Top-1/Top-3/MRR against Phase 0 baseline
- Launch Gradio demo, test with real microphone

## Files Summary

| File | Change | Priority |
|------|--------|----------|
| `training/wav2vec2_contrastive.py` | Major rewrite: XLSR-53, SupCon+Triplet, new augments, grad ckpt, 8-bit optim, validation | P0 |
| `config/settings.py` | Add XLSR-53 paths | P0 |
| `embeddings/embedding_model.py` | Dynamic hidden_size, update defaults | P0 |
| `evaluation/evaluate_acoustic.py` | NEW: acoustic evaluation script | P0 |
| `embeddings/extract_embeddings.py` | Backup before overwrite | P1 |
| `inference/acoustic_scorer.py` | Threshold recalibration | P1 |
| `inference/combined_scorer.py` | Use new settings constants | P1 |

## Estimated Timeline
- Phase 0 (baseline eval): ~10 min
- Phase 1 (download): ~5 min
- Phase 2-4 (code changes): ~30 min
- Phase 5 (training): ~4-5 hours
- Phase 6-8 (extract + eval + calibrate): ~20 min
- **Total: ~5-6 hours** (mostly training time)

## Verification
1. Baseline: record current Top-1 accuracy per lesson
2. After training: Top-1 accuracy should improve significantly (target: >70%)
3. Gradio test: record correct pronunciation → should detect correctly + score >0.8
4. Gradio test: record wrong letter → should detect the wrong letter + score <0.4
