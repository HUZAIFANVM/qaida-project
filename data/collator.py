"""Data collator for Whisper fine-tuning on Qaida data."""

import librosa
import torch
import numpy as np


class QaidaDataCollator:
    """
    Collator that:
    1. Loads WAV files on-the-fly (no caching all audio in RAM)
    2. Resamples 44100Hz -> 16000Hz
    3. Extracts mel spectrogram via WhisperProcessor
    4. Tokenizes Arabic text as target labels
    """

    def __init__(self, processor, target_sr=16000):
        self.processor = processor
        self.target_sr = target_sr

    def __call__(self, features):
        audio_arrays = []
        labels_text = []

        for f in features:
            # Load and resample audio
            audio, _ = librosa.load(f["audio_path"], sr=self.target_sr, mono=True)
            audio_arrays.append(audio)
            # Use Arabic text as the transcription target
            labels_text.append(f["arabic_text"])

        # Process audio -> mel spectrogram features
        # Whisper requires fixed 3000-frame (30s) mel input - use max_length padding
        input_features = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,  # 30 seconds * 16000 Hz
        ).input_features

        # Tokenize Arabic text labels
        label_ids = self.processor.tokenizer(
            labels_text,
            padding=True,
            return_tensors="pt",
        ).input_ids

        # Replace padding token id with -100 so loss ignores padding
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "labels": label_ids,
        }
