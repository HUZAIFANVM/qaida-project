"""Extract reference embeddings for all 692 audio samples."""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "qaida_project.embeddings.extract_embeddings"],
    check=True,
)
