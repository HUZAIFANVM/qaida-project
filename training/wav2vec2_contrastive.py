"""
Contrastive fine-tuning of XLSR-53 for Qaida phoneme-aware embeddings (v3).

Upgrades over v2:
- Auxiliary phoneme-CTC head: forces frame-level encoder output to retain
  phoneme identity, fixing the harakat dilution that v2 mean-pooling causes
- Stronger real-mic-style augmentation: telephone-band resample, harsher
  bandpass, more realistic noise + reverb combinations
- Per-group validation: splits val into easy (L1, L2, L11, L13-15) and hard
  (L3, L4, L5, L9, L10) groups; saves best by hard-group Top-1

Usage:
    python -m qaida_project.training.wav2vec2_contrastive
"""

import os
import sys
import json
import random
import unicodedata
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qaida_project.config.settings import (
    UNIFIED_DATASET_JSON, SPLIT_INDICES_JSON,
    XLSR53_MODEL_NAME, XLSR53_FINETUNED_DIR,
    MODELS_DIR, TARGET_SAMPLE_RATE,
)

OUTPUT_DIR = XLSR53_FINETUNED_DIR


# ============================================================
# Augmentation (7 types)
# ============================================================

def augment_audio(audio, sr=16000):
    """v3 augmentations — stronger real-mic simulation.

    Same 7 base ops as v2 plus:
    8. Telephone-band downsample (8 kHz roundtrip, simulates phone/VoIP codec)
    9. Pink-ish noise (background hum)
    Probabilities tuned higher so each clip gets more transformation.
    """
    aug = audio.copy()

    # 1. Pitch shift: ±3 semitones (wider for varied voices)
    if random.random() < 0.75:
        n_steps = random.uniform(-3, 3)
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=n_steps)

    # 2. Speed change: 0.8x - 1.2x (wider for madd/short variations)
    if random.random() < 0.75:
        rate = random.uniform(0.8, 1.2)
        aug = librosa.effects.time_stretch(aug, rate=rate)

    # 3. Gaussian noise: SNR 10-30dB (lower min for noisier mics)
    if random.random() < 0.7:
        snr_db = random.uniform(10, 30)
        signal_power = np.mean(aug ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(aug))
        aug = aug + noise.astype(aug.dtype)

    # 4. Volume change: ±9dB
    if random.random() < 0.6:
        gain_db = random.uniform(-9, 9)
        aug = aug * (10 ** (gain_db / 20))

    # 5. Aggressive bandpass (simulate cheap mic)
    if random.random() < 0.5:
        try:
            from scipy.signal import butter, sosfilt
            low_cut = random.uniform(100, 400)
            high_cut = random.uniform(3000, 7000)
            sos = butter(4, [low_cut, high_cut], btype='bandpass', fs=sr, output='sos')
            aug = sosfilt(sos, aug).astype(np.float32)
        except Exception:
            pass

    # 6. Room reverb (synthetic)
    if random.random() < 0.35:
        try:
            from scipy.signal import fftconvolve
            reverb_len = int(sr * random.uniform(0.05, 0.18))
            decay = np.exp(-np.linspace(0, 5, reverb_len))
            impulse = np.random.randn(reverb_len).astype(np.float32) * decay.astype(np.float32)
            impulse[0] = 1.0
            impulse = impulse / np.abs(impulse).max()
            aug = fftconvolve(aug, impulse, mode='full')[:len(audio)].astype(np.float32)
        except Exception:
            pass

    # 7. Random silence insertion
    if random.random() < 0.2:
        silence_len = int(sr * random.uniform(0.05, 0.2))
        pos = random.randint(0, max(1, len(aug) - silence_len))
        aug[pos:pos + silence_len] = 0.0

    # 8. Telephone-band roundtrip (simulate VoIP/WhatsApp codec)
    if random.random() < 0.3:
        try:
            tel = librosa.resample(aug, orig_sr=sr, target_sr=8000)
            aug = librosa.resample(tel, orig_sr=8000, target_sr=sr)[:len(audio)]
            if len(aug) < len(audio):
                aug = np.pad(aug, (0, len(audio) - len(aug)))
        except Exception:
            pass

    # 9. Pink-ish background hum
    if random.random() < 0.3:
        try:
            n = len(aug)
            white = np.random.normal(0, 1, n)
            # cumulative sum approximates pink-ish low-freq drift
            pink = np.cumsum(white)
            pink = pink - pink.mean()
            pmax = np.abs(pink).max() + 1e-9
            pink = pink / pmax
            snr_db = random.uniform(15, 30)
            signal_power = np.mean(aug ** 2) + 1e-10
            noise_power = signal_power / (10 ** (snr_db / 10))
            aug = aug + (pink * np.sqrt(noise_power)).astype(aug.dtype)
        except Exception:
            pass

    aug = np.clip(aug, -1.0, 1.0)
    return aug


# ============================================================
# Dataset
# ============================================================

def strip_diacritics(s):
    """Remove Arabic diacritics to get base letter."""
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ============================================================
# Phoneme vocab (v3) — character-level over Arabic transcripts
# Includes diacritics so the CTC head learns harakat distinctions.
# ============================================================

class PhonemeVocab:
    """Character-level vocab built from all Arabic transcripts.

    Index 0 reserved as CTC blank. Whitespace is dropped.
    """
    BLANK_IDX = 0

    def __init__(self, transcripts):
        chars = set()
        for t in transcripts:
            for c in unicodedata.normalize("NFC", t):
                if c.isspace():
                    continue
                chars.add(c)
        self.itos = ["<blank>"] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi[c] for c in unicodedata.normalize("NFC", text)
                if not c.isspace() and c in self.stoi]


class ContrastiveQaidaDataset(Dataset):
    """Each __getitem__ returns two augmented views of the same audio + phoneme labels."""

    def __init__(self, entries, vocab=None, sr=16000):
        self.entries = entries
        self.sr = sr
        self.vocab = vocab or PhonemeVocab([e["arabic_text"] for e in entries])

        print("Pre-loading audio...")
        self.audio_cache = {}
        for e in entries:
            audio, _ = librosa.load(e["audio_path"], sr=sr, mono=True)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio * (0.95 / peak)
            self.audio_cache[e["audio_id"]] = audio

        self.labels = [e["arabic_text"] for e in entries]
        self.base_letters = [strip_diacritics(e["arabic_text"]) for e in entries]
        self.lesson_nums = [e["lesson_num"] for e in entries]
        # Phoneme labels (per-sample list of int indices)
        self.phoneme_labels = [self.vocab.encode(e["arabic_text"]) for e in entries]

        self.base_to_indices = defaultdict(list)
        for i, bl in enumerate(self.base_letters):
            self.base_to_indices[bl].append(i)

        self.lesson_to_indices = defaultdict(list)
        for i, ln in enumerate(self.lesson_nums):
            self.lesson_to_indices[ln].append(i)

        print(f"Loaded {len(entries)} samples, {len(self.base_to_indices)} base letters, "
              f"{len(self.lesson_to_indices)} lessons, vocab={self.vocab.size}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio = self.audio_cache[entry["audio_id"]]

        view1 = augment_audio(audio, self.sr)
        view2 = augment_audio(audio, self.sr)

        return {
            "view1": view1,
            "view2": view2,
            "label_idx": idx,
            "lesson_num": entry["lesson_num"],
            "phoneme_label": self.phoneme_labels[idx],
        }


def collate_contrastive(batch, processor, sr=16000):
    """Collate batch: process both views through feature extractor + flatten CTC labels."""
    all_audios = []
    label_indices = []
    lesson_nums = []
    phoneme_seqs = []
    phoneme_lengths = []

    for item in batch:
        all_audios.append(item["view1"])
        all_audios.append(item["view2"])
        label_indices.append(item["label_idx"])
        label_indices.append(item["label_idx"])
        lesson_nums.append(item["lesson_num"])
        lesson_nums.append(item["lesson_num"])
        # Same phoneme label for both views
        phoneme_seqs.append(item["phoneme_label"])
        phoneme_seqs.append(item["phoneme_label"])
        phoneme_lengths.append(len(item["phoneme_label"]))
        phoneme_lengths.append(len(item["phoneme_label"]))

    inputs = processor(
        all_audios,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    # Flatten phoneme labels for CTC (concatenated)
    flat_targets = torch.tensor([t for seq in phoneme_seqs for t in seq], dtype=torch.long)

    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.get("attention_mask"),
        "labels": torch.tensor(label_indices, dtype=torch.long),
        "lesson_nums": torch.tensor(lesson_nums, dtype=torch.long),
        "ctc_targets": flat_targets,
        "ctc_target_lengths": torch.tensor(phoneme_lengths, dtype=torch.long),
    }


# ============================================================
# Lesson-Aware Batch Sampler
# ============================================================

class LessonAwareSampler:
    """Sample batches with within-lesson hard negatives.

    Each batch: ~half from one lesson, ~half from others.
    Ensures within-lesson discrimination signal every batch.
    """

    def __init__(self, dataset, batch_size=16):
        self.batch_size = batch_size
        self.lesson_to_indices = dict(dataset.lesson_to_indices)
        self.lessons = list(self.lesson_to_indices.keys())
        self.total = len(dataset)

    def __iter__(self):
        random.shuffle(self.lessons)
        all_batches = []

        for lesson in self.lessons:
            indices = self.lesson_to_indices[lesson].copy()
            random.shuffle(indices)

            half = max(4, self.batch_size // 2)
            for start in range(0, len(indices), half):
                batch = indices[start:start + half]

                # Fill remaining slots from other lessons
                other_indices = []
                for other_l in self.lessons:
                    if other_l != lesson:
                        other_indices.extend(self.lesson_to_indices[other_l])
                random.shuffle(other_indices)
                remaining = self.batch_size - len(batch)
                batch.extend(other_indices[:remaining])

                if len(batch) >= 4:
                    all_batches.append(batch)

        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return sum(
            max(1, (len(ids) + self.batch_size // 2 - 1) // (self.batch_size // 2))
            for ids in self.lesson_to_indices.values()
        )


# ============================================================
# Model
# ============================================================

class ContrastiveXLSR(nn.Module):
    """XLSR-53 with projection head + CTC phoneme head (v3) for contrastive learning."""

    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53",
                 proj_dim=128, freeze_layers=16, vocab_size=None):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze CNN feature extractor
        self.encoder.feature_extractor._freeze_parameters()

        # Freeze first N transformer layers
        for i in range(freeze_layers):
            for param in self.encoder.encoder.layers[i].parameters():
                param.requires_grad = False

        self.encoder.gradient_checkpointing_enable()

        hidden_size = self.encoder.config.hidden_size  # 1024

        # Projection head: 1024 → 512 → 128 with BatchNorm
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

        # v3: CTC phoneme head — applied per frame
        self.has_ctc = vocab_size is not None
        if self.has_ctc:
            self.ctc_dropout = nn.Dropout(0.1)
            self.ctc_head = nn.Linear(hidden_size, vocab_size)

    def _frame_lengths(self, attention_mask, hidden):
        return self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1).long()
        )

    def forward(self, input_values, attention_mask=None, return_ctc=False):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T_reduced, H)

        if attention_mask is not None:
            output_lengths = self._frame_lengths(attention_mask, hidden)
            reduced_mask = torch.zeros(
                hidden.shape[:2], dtype=hidden.dtype, device=hidden.device
            )
            for i, length in enumerate(output_lengths):
                reduced_mask[i, :length] = 1.0
            mask = reduced_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
            output_lengths = None

        projected = self.projection(pooled)
        embedding = F.normalize(projected, p=2, dim=1)

        if return_ctc and self.has_ctc:
            ctc_logits = self.ctc_head(self.ctc_dropout(hidden))  # (B, T, V)
            return embedding, ctc_logits, output_lengths
        return embedding


# ============================================================
# Loss Functions
# ============================================================

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Positives: augmented views of same audio (same label).
    Negatives: all other samples.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

    labels_col = labels.unsqueeze(0)
    labels_row = labels.unsqueeze(1)
    pos_mask = (labels_col == labels_row).float()

    eye = torch.eye(batch_size, device=device)
    pos_mask = pos_mask - eye
    logits_mask = 1.0 - eye

    num_pos = pos_mask.sum(dim=1)
    valid = num_pos > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / num_pos.clamp(min=1)
    loss = -mean_log_prob[valid].mean()
    return loss


def hard_negative_triplet_loss(embeddings, identity_labels, lesson_labels, margin=0.3):
    """Triplet loss with hardest in-lesson negatives.

    For each anchor: find hardest positive (augmented view) and
    hardest negative (different item, same lesson).
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    dist_matrix = 1.0 - torch.mm(embeddings, embeddings.t())

    losses = []
    for i in range(batch_size):
        # Positive: same identity
        pos_mask = (identity_labels == identity_labels[i]) & \
                   (torch.arange(batch_size, device=device) != i)
        if pos_mask.sum() == 0:
            continue
        pos_dist = dist_matrix[i][pos_mask].max()

        # Negative: same lesson, different identity
        neg_mask = (lesson_labels == lesson_labels[i]) & \
                   (identity_labels != identity_labels[i])
        if neg_mask.sum() == 0:
            continue
        neg_dist = dist_matrix[i][neg_mask].min()

        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ============================================================
# Validation
# ============================================================

def validate(model, val_entries, all_entries, processor, device, sr=16000):
    """Compute Top-1/Top-3 accuracy on validation set."""
    model.eval()

    # Extract embeddings for all items
    all_embs = {}
    for e in all_entries:
        audio, _ = librosa.load(e["audio_path"], sr=sr, mono=True)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.95 / peak)

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model(**inputs).squeeze().cpu().numpy()
        all_embs[e["audio_id"]] = {"embedding": emb, "lesson_num": e["lesson_num"]}

    correct_top1 = 0
    correct_top3 = 0
    total = 0

    # v3: split val into easy/hard groups
    EASY_LESSONS = {1, 2, 11, 13, 14, 15}
    HARD_LESSONS = {3, 4, 5, 9, 10}

    counts = {"all": [0, 0, 0], "easy": [0, 0, 0], "hard": [0, 0, 0]}  # [c1, c3, total]

    for entry in val_entries:
        audio_id = entry["audio_id"]
        lesson_num = entry["lesson_num"]
        student_emb = all_embs[audio_id]["embedding"]

        lesson_items = [(e["audio_id"], all_embs[e["audio_id"]]["embedding"])
                        for e in all_entries if e["lesson_num"] == lesson_num]

        sims = [(aid, float(np.dot(student_emb, emb))) for aid, emb in lesson_items]
        sims.sort(key=lambda x: -x[1])

        top_ids = [s[0] for s in sims[:3]]
        c1 = 1 if audio_id == top_ids[0] else 0
        c3 = 1 if audio_id in top_ids else 0

        counts["all"][0] += c1; counts["all"][1] += c3; counts["all"][2] += 1
        if lesson_num in EASY_LESSONS:
            counts["easy"][0] += c1; counts["easy"][1] += c3; counts["easy"][2] += 1
        if lesson_num in HARD_LESSONS:
            counts["hard"][0] += c1; counts["hard"][1] += c3; counts["hard"][2] += 1

    model.train()
    def acc(group):
        c1, c3, t = counts[group]
        return (c1 / max(1, t), c3 / max(1, t), t)

    a1, a3, _ = acc("all")
    e1, e3, _ = acc("easy")
    h1, h3, _ = acc("hard")
    return {
        "top1_acc": a1, "top3_acc": a3,
        "easy_top1": e1, "easy_top3": e3,
        "hard_top1": h1, "hard_top3": h3,
    }


# ============================================================
# Training
# ============================================================

def main():
    print("=" * 60)
    print("XLSR-53 CONTRASTIVE FINE-TUNING (v3 — phoneme-CTC + stronger aug)")
    print("=" * 60)

    # Config
    BATCH_SIZE = 16
    LR = 2e-5
    MAX_STEPS = 2000
    LOG_EVERY = 50
    SAVE_EVERY = 200
    EVAL_EVERY = 200
    PROJ_DIM = 128
    FREEZE_LAYERS = 16
    TEMPERATURE = 0.1
    TRIPLET_MARGIN = 0.3
    TRIPLET_WEIGHT = 0.5
    CTC_WEIGHT = 0.3              # v3: aux phoneme-CTC loss weight
    WARMUP_STEPS = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU: {gpu} ({vram:.1f}GB)")

    # Load dataset
    with open(UNIFIED_DATASET_JSON, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    with open(SPLIT_INDICES_JSON, "r") as f:
        split = json.load(f)

    val_entries = [all_entries[i] for i in split["val_indices"]]
    print(f"Dataset: {len(all_entries)} total, {len(val_entries)} val")

    # v3: build phoneme vocab once and share with dataset
    vocab = PhonemeVocab([e["arabic_text"] for e in all_entries])
    print(f"Phoneme vocab: {vocab.size} symbols (incl. CTC blank)")

    dataset = ContrastiveQaidaDataset(all_entries, vocab=vocab, sr=TARGET_SAMPLE_RATE)

    # Feature extractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(XLSR53_MODEL_NAME)

    # Dataloader with lesson-aware sampling
    sampler = LessonAwareSampler(dataset, batch_size=BATCH_SIZE)

    def collate_fn(batch):
        return collate_contrastive(batch, processor, sr=TARGET_SAMPLE_RATE)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    print(f"\nLoading XLSR-53 (freezing first {FREEZE_LAYERS}/24 layers)...")
    model = ContrastiveXLSR(
        model_name=XLSR53_MODEL_NAME,
        proj_dim=PROJ_DIM,
        freeze_layers=FREEZE_LAYERS,
        vocab_size=vocab.size,
    )
    model = model.to(device)
    ctc_loss_fn = nn.CTCLoss(blank=PhonemeVocab.BLANK_IDX, zero_infinity=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.1f}%)")

    # 8-bit optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=0.01,
        )
        print("Using PagedAdamW8bit optimizer")
    except ImportError:
        print("bitsandbytes not available, using standard AdamW")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=0.01,
        )

    # LR scheduler: linear warmup then cosine decay
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nTraining for {MAX_STEPS} steps...")
    print(f"  Batch: {BATCH_SIZE} (×2 views = {BATCH_SIZE * 2} embeddings)")
    print(f"  LR: {LR}, Temperature: {TEMPERATURE}")
    print(f"  Projection: 1024→512→{PROJ_DIM}")
    print(f"  Loss: SupCon + {TRIPLET_WEIGHT}×Triplet(margin={TRIPLET_MARGIN}) + {CTC_WEIGHT}×CTC")
    print()

    model.train()
    step = 0
    running_loss = 0.0
    running_supcon = 0.0
    running_triplet = 0.0
    running_ctc = 0.0
    best_hard_top1 = 0.0  # v3: select model by hard-group accuracy

    while step < MAX_STEPS:
        for batch in dataloader:
            if step >= MAX_STEPS:
                break

            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            lesson_nums = batch["lesson_nums"].to(device)
            ctc_targets = batch["ctc_targets"].to(device)
            ctc_target_lengths = batch["ctc_target_lengths"].to(device)

            # Forward
            with torch.amp.autocast('cuda', enabled=True):
                embeddings, ctc_logits, frame_lengths = model(
                    input_values, attention_mask, return_ctc=True
                )

                loss_supcon = supervised_contrastive_loss(
                    embeddings, labels, temperature=TEMPERATURE
                )
                loss_triplet = hard_negative_triplet_loss(
                    embeddings, labels, lesson_nums, margin=TRIPLET_MARGIN
                )

                # CTC loss: needs (T, B, V) log-probs in fp32
                log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
                if frame_lengths is None:
                    # Without attention_mask, all frames are valid
                    T_full = log_probs.shape[0]
                    input_lengths = torch.full(
                        (log_probs.shape[1],), T_full, dtype=torch.long, device=device,
                    )
                else:
                    input_lengths = frame_lengths.to(device).long().clamp(max=log_probs.shape[0])

                # Guard: drop CTC if any input length < target length (CTC requires T >= L)
                valid = (input_lengths >= ctc_target_lengths)
                if valid.all():
                    loss_ctc = ctc_loss_fn(
                        log_probs, ctc_targets, input_lengths, ctc_target_lengths
                    )
                else:
                    # Compute on valid subset only
                    keep_idx = valid.nonzero(as_tuple=True)[0]
                    if len(keep_idx) > 0:
                        # Reconstruct flat targets for the kept batch
                        starts = torch.cumsum(
                            torch.cat([torch.tensor([0], device=device), ctc_target_lengths]), 0
                        )
                        kept_targets = torch.cat([
                            ctc_targets[starts[i]:starts[i + 1]] for i in keep_idx
                        ])
                        loss_ctc = ctc_loss_fn(
                            log_probs[:, keep_idx, :],
                            kept_targets,
                            input_lengths[keep_idx],
                            ctc_target_lengths[keep_idx],
                        )
                    else:
                        loss_ctc = torch.tensor(0.0, device=device, requires_grad=True)

                loss = loss_supcon + TRIPLET_WEIGHT * loss_triplet + CTC_WEIGHT * loss_ctc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_supcon += loss_supcon.item()
            running_triplet += loss_triplet.item()
            running_ctc += float(loss_ctc.item())
            step += 1

            if step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                avg_sc = running_supcon / LOG_EVERY
                avg_tr = running_triplet / LOG_EVERY
                avg_ctc = running_ctc / LOG_EVERY
                lr = scheduler.get_last_lr()[0]
                vram_used = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
                print(f"  Step {step}/{MAX_STEPS} | Loss: {avg_loss:.4f} "
                      f"(SC:{avg_sc:.3f} TR:{avg_tr:.3f} CTC:{avg_ctc:.3f}) | "
                      f"LR: {lr:.2e} | VRAM: {vram_used:.1f}GB")
                running_loss = 0.0
                running_supcon = 0.0
                running_triplet = 0.0
                running_ctc = 0.0

            if step % EVAL_EVERY == 0:
                print(f"  Validating...")
                val_result = validate(model, val_entries, all_entries,
                                      processor, device, sr=TARGET_SAMPLE_RATE)
                print(f"  Val Top-1: {val_result['top1_acc']:.4f} | "
                      f"Top-3: {val_result['top3_acc']:.4f} | "
                      f"easy={val_result['easy_top1']:.3f} hard={val_result['hard_top1']:.3f}")

                if val_result["hard_top1"] > best_hard_top1:
                    best_hard_top1 = val_result["hard_top1"]
                    save_path = os.path.join(OUTPUT_DIR, "best")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
                    print(f"  New best (hard-group) model! hard_top1={best_hard_top1:.4f}")

            if step % SAVE_EVERY == 0 and step != MAX_STEPS:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))

    # Save final model
    print(f"\nSaving final model to {OUTPUT_DIR}")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))

    # Save config
    config = {
        "base_model": XLSR53_MODEL_NAME,
        "proj_dim": PROJ_DIM,
        "freeze_layers": FREEZE_LAYERS,
        "hidden_size": 1024,
        "vocab_size": vocab.size,
        "version": 3,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save phoneme vocab for downstream usage
    with open(os.path.join(OUTPUT_DIR, "phoneme_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, ensure_ascii=False, indent=2)

    # Save feature extractor
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\nTraining complete! Best hard-group Top-1: {best_hard_top1:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
