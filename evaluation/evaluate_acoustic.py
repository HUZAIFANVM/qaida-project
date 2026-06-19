"""
Evaluate acoustic embedding model for pronunciation detection accuracy.

For each validation sample: extract embedding, find rank of correct item
within same lesson. Reports Top-1, Top-3 accuracy, MRR, per-lesson breakdown.

Supports testing multiple models:
  --model finetuned   (current contrastive wav2vec2-base, default)
  --model pretrained  (pretrained wav2vec2-base, no fine-tuning)
  --model xlsr        (pretrained XLSR-53, no fine-tuning)

Usage:
    python -m qaida_project.evaluation.evaluate_acoustic
    python -m qaida_project.evaluation.evaluate_acoustic --model xlsr
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["HF_HUB_OFFLINE"] = "1"

from qaida_project.config.settings import (
    UNIFIED_DATASET_JSON, SPLIT_INDICES_JSON,
    WAV2VEC2_MODEL_NAME, WAV2VEC2_FINETUNED_DIR,
    XLSR53_MODEL_NAME, XLSR53_FINETUNED_DIR,
    EMBEDDING_MODEL_NAME, EMBEDDING_FINETUNED_DIR,
    REFERENCE_EMBEDDINGS_PATH,
)
from qaida_project.embeddings.embedding_model import ReferenceEmbeddingExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xlsr-finetuned",
                        choices=["finetuned", "pretrained", "xlsr", "xlsr-finetuned"],
                        help="Which model to evaluate")
    args = parser.parse_args()

    print("=" * 60)
    print("ACOUSTIC EMBEDDING EVALUATION")
    print(f"Model: {args.model}")
    print("=" * 60)

    # Load dataset and split
    with open(UNIFIED_DATASET_JSON, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    with open(SPLIT_INDICES_JSON, "r") as f:
        split = json.load(f)

    val_entries = [all_entries[i] for i in split["val_indices"]]
    print(f"Validation set: {len(val_entries)} entries")

    # Build reference embeddings from ALL entries (train+val)
    # Group by lesson for within-lesson evaluation
    by_lesson = defaultdict(list)
    for e in all_entries:
        by_lesson[e["lesson_num"]].append(e)

    # Load model
    if args.model == "finetuned":
        print(f"Loading fine-tuned model from {WAV2VEC2_FINETUNED_DIR}")
        extractor = ReferenceEmbeddingExtractor(
            model_name=WAV2VEC2_MODEL_NAME,
            finetuned_dir=WAV2VEC2_FINETUNED_DIR,
        )
    elif args.model == "pretrained":
        print(f"Loading pretrained {WAV2VEC2_MODEL_NAME}")
        extractor = ReferenceEmbeddingExtractor(
            model_name=WAV2VEC2_MODEL_NAME,
            finetuned_dir=None,
        )
    elif args.model == "xlsr":
        os.environ.pop("HF_HUB_OFFLINE", None)  # need to download
        print(f"Loading pretrained {XLSR53_MODEL_NAME}")
        extractor = ReferenceEmbeddingExtractor(
            model_name=XLSR53_MODEL_NAME,
            finetuned_dir=None,
        )
    elif args.model == "xlsr-finetuned":
        print(f"Loading fine-tuned XLSR-53 from {XLSR53_FINETUNED_DIR}")
        extractor = ReferenceEmbeddingExtractor(
            model_name=XLSR53_MODEL_NAME,
            finetuned_dir=XLSR53_FINETUNED_DIR,
        )

    # Extract embeddings for ALL items (fresh, not from stored pkl)
    print("\nExtracting embeddings for all items...")
    all_embeddings = {}
    for i, e in enumerate(all_entries):
        emb = extractor.extract(e["audio_path"])
        all_embeddings[e["audio_id"]] = {
            "embedding": emb,
            "arabic_text": e["arabic_text"],
            "lesson_num": e["lesson_num"],
        }
        if (i + 1) % 50 == 0:
            print(f"  Extracted {i+1}/{len(all_entries)}")
    print(f"  Extracted {len(all_entries)}/{len(all_entries)}")

    # Evaluate each validation sample
    print("\nEvaluating validation samples...")
    results = []
    for entry in val_entries:
        audio_id = entry["audio_id"]
        lesson_num = entry["lesson_num"]
        student_emb = all_embeddings[audio_id]["embedding"]

        # Get all items in same lesson
        lesson_items = [e for e in all_entries if e["lesson_num"] == lesson_num]
        lesson_ids = [e["audio_id"] for e in lesson_items]

        # Compute similarity to all items in lesson
        sims = []
        for lid in lesson_ids:
            ref_emb = all_embeddings[lid]["embedding"]
            sim = float(np.dot(student_emb, ref_emb))
            sims.append((lid, sim))

        # Sort by similarity (descending)
        sims.sort(key=lambda x: -x[1])

        # Find rank of correct item
        ranked_ids = [s[0] for s in sims]
        rank = ranked_ids.index(audio_id) + 1  # 1-based

        # Get top match info
        top1_id = ranked_ids[0]
        top1_arabic = all_embeddings[top1_id]["arabic_text"]
        correct_sim = next(s[1] for s in sims if s[0] == audio_id)
        top1_sim = sims[0][1]
        margin = correct_sim - (sims[1][1] if len(sims) > 1 and ranked_ids[0] == audio_id else top1_sim)

        results.append({
            "audio_id": audio_id,
            "lesson_num": lesson_num,
            "arabic_text": entry["arabic_text"],
            "rank": rank,
            "correct_sim": round(correct_sim, 4),
            "top1_id": top1_id,
            "top1_arabic": top1_arabic,
            "top1_sim": round(top1_sim, 4),
            "margin": round(margin, 4),
            "lesson_size": len(lesson_ids),
            "correct": rank == 1,
        })

    # Aggregate metrics
    ranks = [r["rank"] for r in results]
    top1_acc = sum(1 for r in ranks if r == 1) / len(ranks)
    top3_acc = sum(1 for r in ranks if r <= 3) / len(ranks)
    mrr = np.mean([1.0 / r for r in ranks])
    margins = [r["margin"] for r in results]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy:  {top1_acc:.4f} ({sum(1 for r in ranks if r == 1)}/{len(ranks)})")
    print(f"  Top-3 Accuracy:  {top3_acc:.4f} ({sum(1 for r in ranks if r <= 3)}/{len(ranks)})")
    print(f"  MRR:             {mrr:.4f}")
    print(f"  Avg Margin:      {np.mean(margins):.4f}")
    print(f"  Avg Correct Sim: {np.mean([r['correct_sim'] for r in results]):.4f}")

    # Per-lesson breakdown
    print(f"\nPer-lesson Top-1 Accuracy:")
    lesson_results = defaultdict(list)
    for r in results:
        lesson_results[r["lesson_num"]].append(r)

    for lnum in sorted(lesson_results):
        lr = lesson_results[lnum]
        l_top1 = sum(1 for r in lr if r["correct"]) / len(lr)
        l_size = lr[0]["lesson_size"]
        lesson_type = next(e["type"] for e in all_entries if e["lesson_num"] == lnum)
        print(f"  L{lnum:2d} ({lesson_type:>20s}): Top1={l_top1:.2f} ({sum(1 for r in lr if r['correct'])}/{len(lr)}) "
              f"| items={l_size} | avg_sim={np.mean([r['correct_sim'] for r in lr]):.4f}")

    # Show worst misdetections
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        wrong.sort(key=lambda r: r["rank"], reverse=True)
        print(f"\nWorst misdetections ({len(wrong)} total):")
        for r in wrong[:15]:
            print(f"  {r['audio_id']} ({r['arabic_text']}) → detected: {r['top1_id']} ({r['top1_arabic']}) "
                  f"| rank={r['rank']}/{r['lesson_size']} | sim={r['correct_sim']:.4f} vs {r['top1_sim']:.4f}")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), f"eval_acoustic_{args.model}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "top1_accuracy": round(top1_acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "mrr": round(mrr, 4),
            "avg_margin": round(float(np.mean(margins)), 4),
            "per_sample": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
