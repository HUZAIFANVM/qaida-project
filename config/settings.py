"""Central settings for the Qaida pronunciation checker project."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.join(BASE_DIR, "qaida_project")

# Data paths
DATA_DIR = os.path.join(PROJECT_DIR, "data")
UNIFIED_DATASET_JSON = os.path.join(DATA_DIR, "unified_dataset.json")
SPLIT_INDICES_JSON = os.path.join(DATA_DIR, "split_indices.json")

# Model paths
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")

# Whisper-large-v3 + LoRA (QLoRA with 4-bit quantization)
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
WHISPER_OUTPUT_DIR = os.path.join(CHECKPOINTS_DIR, "whisper-large-v3-qaida")
WHISPER_FINAL_DIR = os.path.join(MODELS_DIR, "whisper-large-v3-qaida-final")

# Embedding model (v1: wav2vec2-base, kept for reference)
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
WAV2VEC2_FINETUNED_DIR = os.path.join(MODELS_DIR, "wav2vec2-qaida-contrastive")

# Embedding model v2: XLSR-53 (pre-trained on 53 languages including Arabic)
XLSR53_MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
XLSR53_FINETUNED_DIR = os.path.join(MODELS_DIR, "xlsr53-qaida-contrastive")

# Active embedding model (point to XLSR-53 for v2)
EMBEDDING_MODEL_NAME = XLSR53_MODEL_NAME
EMBEDDING_FINETUNED_DIR = XLSR53_FINETUNED_DIR
REFERENCE_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "reference_embeddings.pkl")

# Audio settings
TARGET_SAMPLE_RATE = 16000
ORIGINAL_SAMPLE_RATE = 44100

# LoRA hyperparameters
LORA_R = 16              # rank of adaptation matrices
LORA_ALPHA = 32          # scaling factor
LORA_DROPOUT = 0.1
USE_4BIT = True          # QLoRA: load base model in 4-bit

# Training hyperparameters
TRAIN_BATCH_SIZE = 2     # small batch for 4GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch size = 16
LEARNING_RATE = 1e-4     # higher LR for LoRA (only training adapters)
MAX_STEPS = 1000
WARMUP_STEPS = 50
EVAL_STEPS = 100
SAVE_STEPS = 100
WEIGHT_DECAY = 0.01

# Scoring weights (acoustic-only: Whisper dropped for phoneme-level scoring)
STT_WEIGHT = 0.0
ACOUSTIC_WEIGHT = 1.0
PASS_THRESHOLD = 0.6
GOOD_THRESHOLD = 0.8

# v3 scoring config
USE_DTW_SCORING = True       # frame-level DTW similarity in addition to mean-pool
DTW_WEIGHT = 0.6             # weight of DTW similarity in final acoustic score
MEAN_WEIGHT = 0.4            # weight of mean-pool similarity
USE_MULTI_PROTOTYPE = True   # use pitch-shifted prototype refs (orig + ±2 semitones)
DTW_TOP_K = 5                # rerank only top-K mean-pool candidates with DTW

# Per-lesson DTW config: (dtw_weight, mean_weight, tail_fraction, skip_fraction, min_margin)
#
#   tail_fraction:  use only last N% of frames for DTW (vowel-final lessons)
#   skip_fraction:  skip first N% of frames for DTW (identical-consonant-prefix lessons)
#   min_margin:     hybrid score margin below which detection is flagged as uncertain
#
# Root cause of wrong detection: consonant frames (identical within a lesson) dominate
# the DTW path, drowning out the vowel tail where the actual distinction lives.
LESSON_DTW_CONFIG = {
    # (dtw_w, mean_w, tail_frac, skip_frac, min_margin)
    1:  (0.50, 0.50, 1.00, 0.00, 0.010),  # alphabet — distinct base letters, easy
    2:  (0.50, 0.50, 1.00, 0.00, 0.010),  # shapes
    3:  (0.85, 0.15, 0.50, 0.00, 0.020),  # harakat — vowel at end (بَ بِ بُ)
    4:  (0.85, 0.15, 0.50, 0.00, 0.020),  # long voices — vowel length at end (ثٰ ثٖ ثٗ)
    5:  (0.85, 0.15, 0.50, 0.00, 0.020),  # tanween — nasalized vowel at end
    6:  (0.75, 0.25, 0.80, 0.15, 0.018),  # sukoon — skip initial alif, focus mid+tail
    7:  (0.75, 0.25, 0.80, 0.15, 0.018),  # shaddah — same structure as sukoon
    8:  (0.60, 0.40, 1.00, 0.00, 0.010),  # qalqalah — longer phrases, full DTW
    9:  (0.85, 0.15, 0.50, 0.00, 0.020),  # madd — length/vowel distinction
    10: (0.85, 0.15, 0.50, 0.00, 0.020),  # leen letters — diphthong direction
    11: (0.50, 0.50, 1.00, 0.00, 0.010),  # heavy letters — distinct words
    12: (0.75, 0.25, 0.50, 0.00, 0.020),  # heavy/light alif — short clips
    13: (0.60, 0.40, 1.00, 0.00, 0.010),  # heavy/light laam
    14: (0.60, 0.40, 1.00, 0.00, 0.010),  # heavy/light raa
    15: (0.60, 0.40, 1.00, 0.00, 0.010),  # gunnah
    16: (0.65, 0.35, 1.00, 0.00, 0.015),  # hurof mukatat
}

# Per-lesson thresholds (v3): tighter for advanced lessons where margins are smaller.
# Calibrated from eval_acoustic_xlsr-finetuned.json (avg margin 0.0965 globally).
# Falls back to PASS_THRESHOLD/GOOD_THRESHOLD if lesson_num is missing.
LESSON_THRESHOLDS = {
    # easy: distinct base letters, large margins
    1:  (0.55, 0.75),  # alphabet
    2:  (0.60, 0.80),  # shapes
    11: (0.55, 0.75),  # heavy letters
    13: (0.60, 0.80),  # heavy/light laam
    14: (0.60, 0.80),  # heavy/light raa
    15: (0.60, 0.80),  # gunnah
    # hard: subtle harakat/length/diphthong distinctions
    3:  (0.65, 0.85),  # harakat
    4:  (0.65, 0.85),  # long voices
    5:  (0.65, 0.85),  # tanween
    6:  (0.62, 0.82),  # sukoon
    7:  (0.62, 0.82),  # shaddah
    8:  (0.62, 0.82),  # qalqalah
    9:  (0.65, 0.85),  # madd
    10: (0.70, 0.88),  # leen letters
    12: (0.62, 0.82),  # heavy/light alif
    16: (0.65, 0.85),  # hurof mukatat
}
