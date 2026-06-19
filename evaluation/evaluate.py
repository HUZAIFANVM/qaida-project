"""
Evaluate the fine-tuned Whisper model on the validation set.

Tests the model by transcribing reference audio and comparing
with expected Arabic text. All reference audio should score high
since they are correct pronunciations.

Usage:
    python -m qaida_project.evaluation.evaluate
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qaida_project.config.settings import (
    WHISPER_MODEL_NAME, WHISPER_FINAL_DIR,
    UNIFIED_DATASET_JSON, SPLIT_INDICES_JSON,
)
from qaida_project.inference.stt_scorer import STTScorer


def main():
    print("=" * 60)
    print("WHISPER MODEL EVALUATION")
    print("=" * 60)

    # Load dataset and split
    with open(UNIFIED_DATASET_JSON, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    with open(SPLIT_INDICES_JSON, "r") as f:
        split = json.load(f)

    val_entries = [all_entries[i] for i in split["val_indices"]]
    print(f"Validation set: {len(val_entries)} entries")

    # Load scorer
    print(f"\nLoading model from {WHISPER_FINAL_DIR}...")
    scorer = STTScorer(
        base_model_name=WHISPER_MODEL_NAME,
        lora_adapter_path=WHISPER_FINAL_DIR,
    )

    # Evaluate
    results = []
    for i, entry in enumerate(val_entries):
        result = scorer.compute_score(entry["audio_path"], entry["arabic_text"])
        result["audio_id"] = entry["audio_id"]
        result["lesson_num"] = entry["lesson_num"]
        result["type"] = entry["type"]
        results.append(result)

        if (i + 1) % 10 == 0:
            avg_cer = np.mean([r["cer"] for r in results])
            print(f"  Processed {i+1}/{len(val_entries)} | Running CER: {avg_cer:.4f}")

    # Aggregate metrics
    cers = [r["cer"] for r in results]
    scores = [r["stt_score"] for r in results]
    perfect = sum(1 for r in results if r["cer"] == 0.0)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Average CER:      {np.mean(cers):.4f}")
    print(f"  Median CER:       {np.median(cers):.4f}")
    print(f"  Average Score:    {np.mean(scores):.4f}")
    print(f"  Perfect matches:  {perfect}/{len(results)} ({100*perfect/len(results):.1f}%)")

    # Per-lesson breakdown
    print(f"\nPer-lesson CER:")
    from collections import defaultdict
    by_lesson = defaultdict(list)
    for r in results:
        by_lesson[r["lesson_num"]].append(r["cer"])

    for lnum in sorted(by_lesson):
        lesson_cers = by_lesson[lnum]
        lesson_type = next(r["type"] for r in results if r["lesson_num"] == lnum)
        print(f"  L{lnum:2d} ({lesson_type:>25s}): CER={np.mean(lesson_cers):.4f} (n={len(lesson_cers)})")

    # Show worst predictions
    results_sorted = sorted(results, key=lambda r: r["cer"], reverse=True)
    print(f"\nWorst 10 predictions:")
    for r in results_sorted[:10]:
        print(f"  {r['audio_id']} | expected: {r['expected_text']:<15s} | predicted: {r['predicted_text']:<15s} | CER: {r['cer']:.4f}")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
