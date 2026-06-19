"""Run the full data preparation pipeline."""
import subprocess
import sys

base_python = sys.executable

print("Step 1: Preparing unified dataset...")
subprocess.run([base_python, "-m", "qaida_project.data.prepare_dataset"], check=True)

print("\nStep 2: Creating train/val split...")
subprocess.run([base_python, "-m", "qaida_project.data.split_dataset"], check=True)

print("\nData preparation complete!")
