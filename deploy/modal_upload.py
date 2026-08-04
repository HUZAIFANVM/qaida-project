"""
Upload model files and audio to Modal volumes using Python SDK.

Run once before deploying (from anywhere):
    py -3.12 qaida_project/deploy/modal_upload.py
"""

import sys
from pathlib import Path
import modal

# This file lives at <repo>/deploy/modal_upload.py.
PROJECT = Path(__file__).resolve().parents[1]  # .../qaida_project  (repo root)
BASE = PROJECT.parent                          # .../Qaida_audios   (holds AUDIO_DIRS)

# Only these files are read at inference time. The model dir also holds ~12GB of
# stale training checkpoints (best/, checkpoint-200/ ... checkpoint-1800/) that the
# runtime never touches — put_directory would upload all of them.
#   config.json             — proj_dim / hidden_size / version, read by embedding_model
#   encoder_config.json     — XLSR-53 architecture; avoids a 1.2GB fetch from HF
#   preprocessor_config.json— Wav2Vec2FeatureExtractor settings
#   model.pt                — the fine-tuned weights
MODEL_FILES = [
    "config.json",
    "encoder_config.json",
    "preprocessor_config.json",
    "model.pt",
]

AUDIO_DIRS = [
    "1_Finalized quaida alphabet",
    "2_alphabet_shapes_final.xlsx",
    "3_harakat finalized",
    "4_finalized Long Voices",
    "5_finalized tanween",
    "6_Finalized Sukoon",
    "7_finalized shaddah",
    "8_Finalized qalqalah",
    "9_finalized Madd",
    "10_Finalized leen letters",
    "11_finalized Heavy_Letters",
    "12_finalized Heavy and Light Alif",
    "13_Finalized Heavy and Light Laam",
    "14_Finalized_Heavy_and_Light_Raa",
    "15_finalized_Gunnah",
    "16_finalized_hurof mukatat",
]


def upload_models():
    print("\n=== Uploading to qaida-models volume ===")
    vol = modal.Volume.from_name("qaida-models", create_if_missing=True)

    model_dir = PROJECT / "models" / "xlsr53-qaida-contrastive"
    missing = [f for f in MODEL_FILES if not (model_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"ERROR: missing required model files in {model_dir}: {missing}\n"
            "The container cannot load the scorer without these. Aborting upload."
        )

    total = 0
    with vol.batch_upload(force=True) as batch:
        # Fine-tuned XLSR-53 model — inference files only, not the training checkpoints
        for fname in MODEL_FILES:
            fpath = model_dir / fname
            size = fpath.stat().st_size
            total += size
            print(f"  Queuing: xlsr53-qaida-contrastive/{fname}  ({size / 1e6:.1f} MB)")
            batch.put_file(str(fpath), f"xlsr53-qaida-contrastive/{fname}")

        # Reference embeddings
        emb_path = PROJECT / "models" / "reference_embeddings.pkl"
        total += emb_path.stat().st_size
        print(f"  Queuing: reference_embeddings.pkl  ({emb_path.stat().st_size / 1e6:.1f} MB)")
        batch.put_file(str(emb_path), "reference_embeddings.pkl")

        # Dataset JSON
        ds_path = PROJECT / "data" / "unified_dataset.json"
        total += ds_path.stat().st_size
        print(f"  Queuing: unified_dataset.json  ({ds_path.stat().st_size / 1e6:.1f} MB)")
        batch.put_file(str(ds_path), "unified_dataset.json")

    print(f"Models volume upload complete ({total / 1e6:.1f} MB).")


def upload_audio():
    print("\n=== Uploading to qaida-audio volume ===")
    vol = modal.Volume.from_name("qaida-audio", create_if_missing=True)

    with vol.batch_upload(force=True) as batch:
        for d in AUDIO_DIRS:
            local_path = BASE / d
            if not local_path.exists():
                print(f"  SKIP (not found): {d}")
                continue
            wav_count = len(list(local_path.glob("*.wav")))
            print(f"  Queuing: {d}  ({wav_count} wav files)")
            batch.put_directory(str(local_path), d)

    print("Audio volume upload complete.")


def main():
    print("=" * 60)
    print("MODAL VOLUME UPLOAD")
    print("=" * 60)

    upload_models()
    upload_audio()

    print("\n" + "=" * 60)
    print("All uploads done! Now run:")
    print("    py -3.12 -m modal deploy modal_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
