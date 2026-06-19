"""Run Whisper-large-v3 QLoRA fine-tuning."""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "qaida_project.training.whisper_finetune"],
    check=True,
)
