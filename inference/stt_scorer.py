"""STT-based scoring using fine-tuned Whisper + LoRA."""

import librosa
import torch
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


class STTScorer:
    """Score pronunciation by comparing Whisper transcription against expected Arabic text."""

    def __init__(self, base_model_name, lora_adapter_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = 16000

        # Load processor
        self.processor = WhisperProcessor.from_pretrained(lora_adapter_path)

        # Load base model in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        self.model.eval()

        # Set Arabic language and transcribe task
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="ar", task="transcribe"
        )

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to Arabic text."""
        audio, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
        ).input_features.to(self.device, dtype=torch.float16)

        predicted_ids = self.model.generate(
            input_features=input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=32,
            no_repeat_ngram_size=3,
        )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return transcription

    def compute_score(self, audio_path: str, expected_arabic: str) -> dict:
        """Compute STT score by comparing transcription with expected Arabic text."""
        predicted = self.transcribe(audio_path)

        predicted_norm = predicted.strip()
        expected_norm = expected_arabic.strip()

        # Character Error Rate
        cer = jiwer.cer(expected_norm, predicted_norm)
        stt_score = max(0.0, 1.0 - cer)

        return {
            "stt_score": round(stt_score, 4),
            "predicted_text": predicted,
            "expected_text": expected_arabic,
            "cer": round(cer, 4),
        }
