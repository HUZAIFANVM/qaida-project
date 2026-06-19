"""
Extract reference audio embeddings (v3) for all 692 Qaida samples.

v3 stores per audio_id:
  - "embedding": mean-pool L2-normalized projected embedding (proj_dim,)
  - "prototypes": (3, proj_dim) — orig, +2 semitones, -2 semitones
  - "frame_embedding": (T, hidden_size) fp16 — per-frame encoder embeddings
                        for cosine-DTW scoring against the original prototype

Usage:
    python -m qaida_project.embeddings.extract_embeddings
"""

import os
import sys
import json
import pickle
import shutil
import time
import numpy as np
import librosa
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qaida_project.config.settings import (
    UNIFIED_DATASET_JSON, MODELS_DIR, REFERENCE_EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME, EMBEDDING_FINETUNED_DIR,
    USE_MULTI_PROTOTYPE,
)
from qaida_project.embeddings.embedding_model import ReferenceEmbeddingExtractor


PITCH_SHIFTS = [0, +2, -2]  # semitones


def _load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def _embed_audio_array(extractor, audio):
    """Mean-pool embedding from a numpy audio array."""
    audio_norm = extractor._normalize_audio(audio)
    inputs = extractor._audio_to_inputs(audio_norm)
    with torch.no_grad():
        if extractor.use_finetuned:
            emb = extractor.model(**inputs).squeeze().cpu().float().numpy()
        else:
            outputs = extractor.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().float().numpy()
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
    return emb


def _frames_audio_array(extractor, audio):
    return extractor.extract_frames(audio, return_fp16=True)


def main():
    print("=" * 60)
    print("REFERENCE EMBEDDING EXTRACTION (v3)")
    print("=" * 60)

    # --- Backup existing pkl if present ---
    if os.path.exists(REFERENCE_EMBEDDINGS_PATH):
        backup = REFERENCE_EMBEDDINGS_PATH.replace(".pkl", "_v2_backup.pkl")
        if not os.path.exists(backup):
            shutil.copy2(REFERENCE_EMBEDDINGS_PATH, backup)
            print(f"Backed up old pkl -> {backup}")

    with open(UNIFIED_DATASET_JSON, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} entries")

    print(f"Loading model: {EMBEDDING_MODEL_NAME}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    extractor = ReferenceEmbeddingExtractor(
        model_name=EMBEDDING_MODEL_NAME,
        finetuned_dir=EMBEDDING_FINETUNED_DIR,
    )

    store = {}
    t0 = time.time()
    for i, entry in enumerate(entries):
        audio = _load_audio(entry["audio_path"])

        # Original mean-pool + frames
        orig_emb = _embed_audio_array(extractor, audio)
        frame_emb = _frames_audio_array(extractor, audio)

        # Multi-prototype: orig + pitch-shifted variants
        if USE_MULTI_PROTOTYPE:
            protos = [orig_emb]
            for n_steps in PITCH_SHIFTS:
                if n_steps == 0:
                    continue
                try:
                    shifted = librosa.effects.pitch_shift(audio, sr=16000, n_steps=n_steps)
                    protos.append(_embed_audio_array(extractor, shifted))
                except Exception as e:
                    print(f"  pitch-shift n={n_steps} failed for {entry['audio_id']}: {e}")
            prototypes = np.stack(protos).astype(np.float32)
        else:
            prototypes = orig_emb[None, :].astype(np.float32)

        store[entry["audio_id"]] = {
            "embedding": orig_emb.astype(np.float32),
            "prototypes": prototypes,
            "frame_embedding": frame_emb,  # fp16
            "word_id": entry["word_id"],
            "lesson_id": entry["lesson_id"],
            "lesson_num": entry["lesson_num"],
            "type": entry["type"],
            "arabic_text": entry["arabic_text"],
            "transliteration": entry["transliteration"],
            "audio_path": entry["audio_path"],
        }

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(entries) - i - 1)
            print(f"  {i + 1}/{len(entries)} | elapsed {elapsed:.0f}s | eta {eta:.0f}s")

    print(f"  Done: {len(entries)}/{len(entries)} in {time.time() - t0:.0f}s")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REFERENCE_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(store, f)
    pkl_mb = os.path.getsize(REFERENCE_EMBEDDINGS_PATH) / 1e6
    print(f"\nSaved: {REFERENCE_EMBEDDINGS_PATH} ({pkl_mb:.1f} MB)")

    # Also save mean embeddings as numpy for fast loading
    audio_ids = sorted(store.keys())
    embeddings_matrix = np.stack([store[aid]["embedding"] for aid in audio_ids])
    npz_path = REFERENCE_EMBEDDINGS_PATH.replace(".pkl", ".npz")
    np.savez(npz_path, embeddings=embeddings_matrix, audio_ids=np.array(audio_ids))
    print(f"Saved: {npz_path} (shape: {embeddings_matrix.shape})")


if __name__ == "__main__":
    main()
