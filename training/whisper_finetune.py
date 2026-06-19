"""
Whisper-large-v3 fine-tuning with QLoRA for Qaida pronunciation.

Uses 4-bit quantization to fit whisper-large in 4GB VRAM,
with LoRA adapters for efficient fine-tuning on 692 samples.
Target: Arabic text transcription from the xlsx files.

Usage:
    python -m qaida_project.training.whisper_finetune
    python -m qaida_project.training.whisper_finetune --resume   # resume from last checkpoint
"""

import os
import sys
import glob
import torch
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qaida_project.config.settings import (
    WHISPER_MODEL_NAME, WHISPER_OUTPUT_DIR, WHISPER_FINAL_DIR,
    UNIFIED_DATASET_JSON, SPLIT_INDICES_JSON, TARGET_SAMPLE_RATE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, USE_4BIT,
    TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE,
    MAX_STEPS, WARMUP_STEPS, EVAL_STEPS, SAVE_STEPS, WEIGHT_DECAY,
)
from qaida_project.data.dataset import load_dataset_splits
from qaida_project.data.collator import QaidaDataCollator


def setup_model_and_processor():
    """Load whisper-large-v3 with 4-bit quantization + LoRA adapters."""
    print(f"Loading model: {WHISPER_MODEL_NAME}")
    print(f"4-bit quantization: {USE_4BIT}")

    # Load processor (tokenizer + feature extractor)
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)

    # 4-bit quantization config for QLoRA
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Prepare model for k-bit training (handles gradient checkpointing etc.)
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA - apply to attention projection layers in both encoder and decoder
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Configure for Arabic transcription via generation_config (not model.config)
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ar", task="transcribe"
    )
    model.generation_config.suppress_tokens = []

    return model, processor


def compute_metrics(pred, processor):
    """Compute CER and WER for evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id for decoding
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize: strip whitespace
    pred_str = [p.strip() for p in pred_str]
    label_str = [l.strip() for l in label_str]

    # Character Error Rate (primary metric for short Arabic text)
    cer = jiwer.cer(label_str, pred_str)

    # Word Error Rate
    wer = jiwer.wer(label_str, pred_str)

    return {"cer": cer, "wer": wer}


def main():
    print("=" * 60)
    print("QAIDA WHISPER FINE-TUNING (QLoRA)")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu} ({vram:.1f}GB VRAM)")
    else:
        print("WARNING: No GPU detected, training will be very slow!")

    # Load model and processor
    model, processor = setup_model_and_processor()

    # Load datasets
    print(f"\nLoading dataset from {UNIFIED_DATASET_JSON}")
    train_dataset, val_dataset = load_dataset_splits(
        UNIFIED_DATASET_JSON, SPLIT_INDICES_JSON
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Create data collator
    data_collator = QaidaDataCollator(processor=processor, target_sr=TARGET_SAMPLE_RATE)

    # Create output dirs
    os.makedirs(WHISPER_OUTPUT_DIR, exist_ok=True)
    os.makedirs(WHISPER_FINAL_DIR, exist_ok=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=WHISPER_OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=32,
        dataloader_num_workers=0,  # Windows: 0 is safest
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",  # 8-bit optimizer to save more VRAM
    )

    # Trainer with custom compute_metrics
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor.feature_extractor,
    )

    # Check for resume from checkpoint
    resume_from = None
    if "--resume" in sys.argv:
        checkpoints = sorted(glob.glob(os.path.join(WHISPER_OUTPUT_DIR, "checkpoint-*")))
        if checkpoints:
            resume_from = checkpoints[-1]
            print(f"\nResuming from checkpoint: {resume_from}")
        else:
            print("\nNo checkpoint found, starting fresh.")

    # Train
    print("\nStarting training...")
    print(f"  Batch size: {TRAIN_BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print()

    trainer.train(resume_from_checkpoint=resume_from)

    # Save the final LoRA adapter + processor
    print(f"\nSaving final model to {WHISPER_FINAL_DIR}")
    model.save_pretrained(WHISPER_FINAL_DIR)
    processor.save_pretrained(WHISPER_FINAL_DIR)

    # Run final evaluation
    print("\nFinal evaluation on validation set...")
    metrics = trainer.evaluate()
    print(f"  CER: {metrics.get('eval_cer', 'N/A')}")
    print(f"  WER: {metrics.get('eval_wer', 'N/A')}")

    print("\nTraining complete!")
    print(f"Model saved to: {WHISPER_FINAL_DIR}")


if __name__ == "__main__":
    main()
