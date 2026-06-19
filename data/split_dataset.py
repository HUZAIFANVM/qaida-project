"""
Create stratified train/validation split.

85% train / 15% validation, stratified by lesson_num to ensure
each lesson is proportionally represented in both sets.

Usage:
    python -m qaida_project.data.split_dataset
"""

import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qaida_project.config.lesson_config import BASE_DIR


def stratified_split(entries, test_ratio=0.15, random_seed=42):
    """Stratified split by lesson_num."""
    import random
    rng = random.Random(random_seed)

    # Group by lesson_num
    by_lesson = {}
    for i, entry in enumerate(entries):
        lnum = entry["lesson_num"]
        if lnum not in by_lesson:
            by_lesson[lnum] = []
        by_lesson[lnum].append(i)

    train_indices = []
    val_indices = []

    for lnum in sorted(by_lesson.keys()):
        indices = by_lesson[lnum][:]
        rng.shuffle(indices)

        n_val = max(1, round(len(indices) * test_ratio))  # at least 1 per lesson
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return sorted(train_indices), sorted(val_indices)


def main():
    # Load unified dataset
    json_path = os.path.join(BASE_DIR, "qaida_project", "data", "unified_dataset.json")
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Total entries: {len(entries)}")

    # Split
    train_idx, val_idx = stratified_split(entries, test_ratio=0.15, random_seed=42)

    print(f"Train: {len(train_idx)} ({len(train_idx)/len(entries)*100:.1f}%)")
    print(f"Val:   {len(val_idx)} ({len(val_idx)/len(entries)*100:.1f}%)")

    # Verify stratification
    train_lessons = Counter(entries[i]["lesson_num"] for i in train_idx)
    val_lessons = Counter(entries[i]["lesson_num"] for i in val_idx)

    print(f"\nPer-lesson breakdown:")
    print(f"  {'Lesson':>8s}  {'Train':>6s}  {'Val':>5s}  {'Total':>6s}  {'Val%':>6s}")
    for lnum in sorted(set(list(train_lessons.keys()) + list(val_lessons.keys()))):
        tr = train_lessons.get(lnum, 0)
        va = val_lessons.get(lnum, 0)
        total = tr + va
        pct = va / total * 100 if total > 0 else 0
        print(f"  L{lnum:>6d}  {tr:>6d}  {va:>5d}  {total:>6d}  {pct:>5.1f}%")

    # Save split indices
    split_data = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "total": len(entries),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "random_seed": 42,
        "test_ratio": 0.15,
    }

    output_path = os.path.join(BASE_DIR, "qaida_project", "data", "split_indices.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)
    print(f"\nSaved split indices: {output_path}")


if __name__ == "__main__":
    main()
