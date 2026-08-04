"""Audio embedding extractor using wav2vec2/XLSR-53 for acoustic similarity.

Supports:
- Pretrained wav2vec2-base (768-dim) or XLSR-53 (1024-dim)
- Contrastive-fine-tuned models with projection head (128 or 256-dim)
"""

import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


def resolve_encoder_config(model_name, finetuned_dir=None):
    """Build the encoder config without downloading the 1.2GB pretrained checkpoint.

    A fine-tuned model.pt covers every encoder parameter, so from_pretrained()
    would fetch 1.2GB only to have it immediately overwritten. We need the
    architecture, not the weights. Prefers a local encoder_config.json so a cold
    container never touches the network; falls back to the hub (~2KB of JSON).
    """
    if finetuned_dir:
        local_cfg = os.path.join(finetuned_dir, "encoder_config.json")
        if os.path.exists(local_cfg):
            return Wav2Vec2Config.from_json_file(local_cfg)
    return Wav2Vec2Config.from_pretrained(model_name)


class ContrastiveWav2Vec2(nn.Module):
    """wav2vec2/XLSR-53 encoder + projection head (must match training architecture)."""

    def __init__(self, model_name, proj_dim=128, hidden_size=1024, encoder_config=None):
        super().__init__()
        # Weights come from model.pt, so build the architecture from config only.
        self.encoder = Wav2Vec2Model(encoder_config or resolve_encoder_config(model_name))
        # v2 projection: hidden → 512 → proj_dim with BatchNorm
        # v1 projection: hidden → hidden → proj_dim (no BatchNorm)
        # Detect from proj_dim and hidden_size which version
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def _reduced_mask(self, attention_mask, hidden):
        """Reduce attention mask to match hidden states after CNN downsampling."""
        output_lengths = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1).long()
        )
        reduced_mask = torch.zeros(
            hidden.shape[:2], dtype=hidden.dtype, device=hidden.device
        )
        for i, length in enumerate(output_lengths):
            reduced_mask[i, :length] = 1.0
        return reduced_mask.unsqueeze(-1)

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            mask = self._reduced_mask(attention_mask, hidden)
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            hidden = hidden.mean(dim=1)
        projected = self.projection(hidden)
        return F.normalize(projected, p=2, dim=1)


class ContrastiveWav2Vec2V1(nn.Module):
    """v1 architecture (wav2vec2-base, hidden→hidden→proj_dim, no BatchNorm)."""

    def __init__(self, model_name, proj_dim=256, hidden_size=768, encoder_config=None):
        super().__init__()
        self.encoder = Wav2Vec2Model(encoder_config or resolve_encoder_config(model_name))
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            output_lengths = self.encoder._get_feat_extract_output_lengths(
                attention_mask.sum(-1).long()
            )
            reduced_mask = torch.zeros(
                hidden.shape[:2], dtype=hidden.dtype, device=hidden.device
            )
            for i, length in enumerate(output_lengths):
                reduced_mask[i, :length] = 1.0
            mask = reduced_mask.unsqueeze(-1)
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            hidden = hidden.mean(dim=1)
        projected = self.projection(hidden)
        return F.normalize(projected, p=2, dim=1)


class ReferenceEmbeddingExtractor:
    """Extract fixed-size audio embeddings using wav2vec2 or XLSR-53.

    If finetuned_dir is provided and contains model.pt, loads the
    contrastive-fine-tuned model. Otherwise falls back to pretrained model.
    """

    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", device=None, finetuned_dir=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = 16000
        self.use_finetuned = False
        # fp16 is a GPU optimisation. x86 CPUs have no native fp16 compute path,
        # so torch emulates it via fp32 up/downconversion on every op — measured
        # ~16x slower than plain fp32 on CPU. Only cast when we're on CUDA.
        self.use_fp16 = str(self.device).startswith("cuda")

        # Try loading fine-tuned contrastive model
        if finetuned_dir and os.path.exists(os.path.join(finetuned_dir, "model.pt")):
            print(f"Loading fine-tuned contrastive model from {finetuned_dir}")
            config_path = os.path.join(finetuned_dir, "config.json")
            with open(config_path, "r") as f:
                cfg = json.load(f)

            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(finetuned_dir)

            # Detect architecture version from config
            version = cfg.get("version", 1)
            encoder_config = resolve_encoder_config(cfg["base_model"], finetuned_dir)
            if version >= 2:
                self.model = ContrastiveWav2Vec2(
                    model_name=cfg["base_model"],
                    proj_dim=cfg["proj_dim"],
                    hidden_size=cfg["hidden_size"],
                    encoder_config=encoder_config,
                )
            else:
                self.model = ContrastiveWav2Vec2V1(
                    model_name=cfg["base_model"],
                    proj_dim=cfg["proj_dim"],
                    hidden_size=cfg["hidden_size"],
                    encoder_config=encoder_config,
                )

            state_dict = torch.load(
                os.path.join(finetuned_dir, "model.pt"),
                map_location=self.device,
                weights_only=False,
            )
            # v3 weights include ctc_head.* (training-only). Inference strips it.
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            extras = [k for k in unexpected if not k.startswith(("ctc_head", "ctc_dropout"))]
            if missing or extras:
                print(f"  load_state_dict missing={missing} unexpected={extras}")
            self.model.eval()
            if self.use_fp16:
                self.model.half()
            self.model.to(self.device)
            self.use_finetuned = True
            self.embed_dim = cfg["proj_dim"]
            precision = "fp16" if self.use_fp16 else "fp32"
            print(
                f"Contrastive model loaded ({self.embed_dim}-dim, v{version}, "
                f"{self.device}/{precision})"
            )
        else:
            # Fallback: pretrained model (no fine-tuning)
            print(f"Loading pretrained {model_name}")
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.model.eval()
            if self.use_fp16:
                self.model.half()
            self.model.to(self.device)
            self.embed_dim = self.model.config.hidden_size  # 768 or 1024

    @staticmethod
    def _normalize_audio(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Trim silence aggressively (top_db=30 vs old 25) and peak-normalize.

        Tier 1: stronger silence trim — half-second of leading room tone was
        dominating the mean pool on advanced lessons.
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        if len(trimmed) < 400:
            trimmed = audio
        peak = np.max(np.abs(trimmed))
        if peak > 0:
            trimmed = trimmed * (0.95 / peak)
        return trimmed

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load + silence-trim + peak-normalize."""
        audio, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return self._normalize_audio(audio)

    def _audio_to_inputs(self, audio: np.ndarray) -> dict:
        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        # Input dtype must match the weight dtype (see use_fp16 in __init__).
        return {k: v.to(self.device).half()
                   if (self.use_fp16 and v.is_floating_point())
                   else v.to(self.device)
                for k, v in inputs.items()}

    @torch.no_grad()
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract one mean-pooled L2-normalized embedding (proj_dim) from a WAV."""
        audio = self._load_audio(audio_path)
        inputs = self._audio_to_inputs(audio)

        if self.use_finetuned:
            embedding = self.model(**inputs).squeeze().cpu().float().numpy()
        else:
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().float().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    @torch.no_grad()
    def extract_both(self, audio_path: str) -> tuple:
        """Single forward pass → (mean_pool_embedding, frame_embeddings).

        Runs XLSR-53 ONCE instead of twice, halving scoring latency.
        Returns:
            mean_emb  : (proj_dim,) float32 — L2-normalised projected embedding
            frame_emb : (T, hidden_size) float16 — per-frame encoder embeddings
        """
        audio = self._load_audio(audio_path)
        inputs = self._audio_to_inputs(audio)

        if self.use_finetuned:
            # Run encoder once
            enc_out = self.model.encoder(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask"),
            )
            hidden = enc_out.last_hidden_state  # (1, T, H)

            # Mean-pool → projection → mean_emb
            attn = inputs.get("attention_mask")
            if attn is not None:
                lengths = self.model.encoder._get_feat_extract_output_lengths(
                    attn.sum(-1).long()
                )
                mask = torch.zeros(hidden.shape[:2], dtype=hidden.dtype, device=hidden.device)
                for i, l in enumerate(lengths):
                    mask[i, :l] = 1.0
                pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)
            projected = self.model.projection(pooled)
            mean_emb = F.normalize(projected, p=2, dim=1).squeeze(0).cpu().float().numpy()

            # Per-frame L2-normalised embeddings
            frames = F.normalize(hidden.squeeze(0), p=2, dim=-1).cpu().to(torch.float16).numpy()
        else:
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            # mean_emb
            pooled = hidden.mean(dim=1).squeeze(0)
            n = torch.norm(pooled)
            mean_emb = (pooled / n.clamp(min=1e-9)).cpu().float().numpy()
            # frames
            frames = F.normalize(hidden.squeeze(0), p=2, dim=-1).cpu().to(torch.float16).numpy()

        return mean_emb, frames

    @torch.no_grad()
    def extract_frames(self, audio_path_or_array, return_fp16: bool = True) -> np.ndarray:
        """Extract L2-normalized per-frame encoder embeddings.

        Tier 2: frame-level features for DTW-based scoring. Bypasses the mean
        pool that dilutes harakat / madd / leen distinctions. Uses the encoder's
        last_hidden_state (1024-d for XLSR-53, 768-d for wav2vec2-base) rather
        than the projection head, since the projection's BatchNorm running
        stats were computed on mean-pooled vectors and would be miscalibrated
        per-frame.

        Returns shape (T_frames, hidden_size).
        """
        if isinstance(audio_path_or_array, str):
            audio = self._load_audio(audio_path_or_array)
        else:
            audio = self._normalize_audio(np.asarray(audio_path_or_array, dtype=np.float32))

        inputs = self._audio_to_inputs(audio)

        if self.use_finetuned:
            outputs = self.model.encoder(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask"),
            )
        else:
            outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state.squeeze(0)  # (T, H)
        hidden = F.normalize(hidden, p=2, dim=-1)
        arr = hidden.cpu().float().numpy()
        if return_fp16:
            arr = arr.astype(np.float16)
        return arr
