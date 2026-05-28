"""
PaddleOCR baseline evaluation.

Evaluate on 2 datasets:
1. Synthetic test set: data/synthetic/ (images) + data/splits/test.csv (labels)
2. Real-world dataset: data/real_dataset/images/ + data/real_dataset/labels/

Metrics: full string accuracy and CER (Character Error Rate)
After removing all special characters (keep only alphanumeric).

Results saved in result/
"""

import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from paddleocr import PaddleOCR


# Config path
ROOT_DIR = Path(__file__).resolve().parent.parent
SYNTHETIC_IMG_DIR = ROOT_DIR / "data" / "synthetic"
TEST_CSV = ROOT_DIR / "data" / "splits" / "test.csv"
REAL_IMG_DIR = ROOT_DIR / "data" / "real_dataset" / "images"
REAL_LABEL_DIR = ROOT_DIR / "data" / "real_dataset" / "labels"
RESULT_DIR = ROOT_DIR / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Init Paddle OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)


def clean_text(text: str) -> str:
    """Remove all special characters, keep only alphanumeric."""
    return re.sub(r"[^a-zA-Z0-9]", "", text)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])


def compute_cer(ground_truth: str, prediction: str) -> float:
    """Compute Character Error Rate."""
    gt_clean = clean_text(ground_truth)
    pred_clean = clean_text(prediction)
    if len(gt_clean) == 0:
        return 0.0 if len(pred_clean) == 0 else 1.0
    dist = levenshtein_distance(gt_clean, pred_clean)
    return dist / len(gt_clean)


def predict_text(image_path: str) -> str:
    """Run PaddleOCR on an image and return concatenated recognized text."""
    try:
        result = ocr.predict(image_path)
        return " ".join(result[0]["rec_texts"]).strip()
    except Exception as e:
        print(f"    [ERROR] OCR failed for {image_path}: {e}")
        return ""


# Evaluation function for synthetic dataset
def evaluate_synthetic() -> dict:
    print("\n" + "=" * 60)
    print("EVALUATION 1: SYNTHETIC TEST SET")
    print("=" * 60)

    # Read test.csv
    samples = []
    with open(TEST_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    print(f"  Total test samples: {len(samples)}")

    results = []
    correct = 0
    total_cer = 0.0

    for i, sample in enumerate(samples):
        img_name = sample["img"]
        label = sample["plate"]
        img_path = str(SYNTHETIC_IMG_DIR / img_name)

        if not os.path.exists(img_path):
            print(f"    [WARN] Image not found: {img_path}")
            continue

        pred = predict_text(img_path)
        is_correct = clean_text(pred) == clean_text(label)
        cer = compute_cer(label, pred)

        if is_correct:
            correct += 1
        total_cer += cer

        results.append({
            "image": img_name,
            "label": label,
            "prediction": pred,
            "exact_match": is_correct,
            "cer": round(cer, 4),
        })

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(samples)}...")

    n = len(results)
    accuracy = correct / n * 100 if n > 0 else 0
    avg_cer = total_cer / n if n > 0 else 0

    summary = {
        "dataset": "synthetic",
        "total_samples": n,
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "avg_cer": round(avg_cer, 4),
    }

    print(f"\n  Results:")
    print(f"    Total: {n}")
    print(f"    Exact matches: {correct}")
    print(f"    Accuracy: {accuracy:.2f}%")
    print(f"    Avg CER: {avg_cer:.4f}")

    # Save detailed results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULT_DIR / f"synthetic_results_{ts}.json"
    with open(detail_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print(f"  Detailed results saved to: {detail_path}")

    return summary


# Evaluation function for real-world dataset
def evaluate_realworld() -> dict:
    print("\n" + "=" * 60)
    print("EVALUATION 2: REAL-WORLD DATASET")
    print("=" * 60)

    # Collect all label files
    label_files = sorted(REAL_LABEL_DIR.glob("*.txt"))
    if not label_files:
        print("  [WARN] No label files found!")
        return {}

    # Read all samples (image_path, label)
    samples = []
    for lf in label_files:
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    img_name, label = parts
                    img_path = str(REAL_IMG_DIR / img_name)
                    samples.append((img_name, label, img_path))

    print(f"  Total real-world labels: {len(samples)}")
    print(f"  Label files: {len(label_files)}")

    results = []
    correct = 0
    total_cer = 0.0

    for i, (img_name, label, img_path) in enumerate(samples):
        if not os.path.exists(img_path):
            continue

        pred = predict_text(img_path)
        is_correct = clean_text(pred) == clean_text(label)
        cer = compute_cer(label, pred)

        if is_correct:
            correct += 1
        total_cer += cer

        results.append({
            "image": img_name,
            "label": label,
            "prediction": pred,
            "exact_match": is_correct,
            "cer": round(cer, 4),
        })

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(samples)}...")

    n = len(results)
    accuracy = correct / n * 100 if n > 0 else 0
    avg_cer = total_cer / n if n > 0 else 0

    summary = {
        "dataset": "realworld",
        "total_samples": n,
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "avg_cer": round(avg_cer, 4),
    }

    print(f"\n  Results:")
    print(f"    Total: {n}")
    print(f"    Exact matches: {correct}")
    print(f"    Accuracy: {accuracy:.2f}%")
    print(f"    Avg CER: {avg_cer:.4f}")

    # Save detailed results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULT_DIR / f"realworld_results_{ts}.json"
    with open(detail_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print(f"  Detailed results saved to: {detail_path}")

    return summary


def main():
    print("=" * 60)
    print("PaddleOCR BASELINE EVALUATION")
    print("=" * 60)
    print(f"  Model: PaddleOCR (lang=en)")
    print(f"  Metric: Exact match (after removing special chars) + CER")

    # Evaluate synthetic
    syn_summary = evaluate_synthetic()

    # Evaluate real-world
    real_summary = evaluate_realworld()

    # Final comparison 
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Dataset':<20} {'Samples':<10} {'Accuracy%':<12} {'Avg CER':<12}")
    print("-" * 54)
    if syn_summary:
        print(f"{'Synthetic':<20} {syn_summary['total_samples']:<10} "
              f"{syn_summary['accuracy_pct']:<12} {syn_summary['avg_cer']:<12}")
    if real_summary:
        print(f"{'Real-world':<20} {real_summary['total_samples']:<10} "
              f"{real_summary['accuracy_pct']:<12} {real_summary['avg_cer']:<12}")

    # Save final summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = {
        "synthetic": syn_summary,
        "realworld": real_summary,
    }
    final_path = RESULT_DIR / f"final_baseline_summary_{ts}.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  Final summary saved to: {final_path}")
    print("Done!")


if __name__ == "__main__":
    main()