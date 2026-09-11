"""
Generate submission.csv for CropGuard.

Usage:
    python3 predict.py --test_dir /path/to/hidden_test --out submission.csv

Expects the test directory to contain image files directly (sample_id is the
filename without extension), OR class-labeled subfolders (for local
evaluation against the validation set). Output format matches
SUBMISSION_GUIDE.md exactly: sample_id,predicted_class,confidence
"""
import argparse
import glob
import os
import csv
import joblib
import numpy as np

from features import load_and_extract

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cropguard_model.joblib")


def find_images(test_dir):
    """Return list of (sample_id, path). Handles both a flat folder of images
    and a folder of class subfolders (labels are ignored at predict time;
    only used if you want to also compute accuracy locally)."""
    exts = ("*.png", "*.jpg", "*.jpeg")
    flat = []
    for e in exts:
        flat.extend(glob.glob(os.path.join(test_dir, e)))
    if flat:
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in sorted(flat)]

    # fall back: class subfolders
    nested = []
    for e in exts:
        nested.extend(glob.glob(os.path.join(test_dir, "*", e)))
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in sorted(nested)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--unknown_threshold", type=float, default=None,
                     help="Optional: if max class probability is below this, "
                          "you may want to flag it manually as an ambiguous case. "
                          "Confidence is still reported as the model's max probability.")
    args = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    model, scaler, classes = bundle["model"], bundle["scaler"], bundle["classes"]
    print(f"Loaded model={bundle['model_name']} (held-out val_accuracy={bundle['val_accuracy']:.4f})")

    items = find_images(args.test_dir)
    print(f"Found {len(items)} images in {args.test_dir}")

    rows = []
    for sample_id, path in items:
        feat = load_and_extract(path).reshape(1, -1)
        feat_s = scaler.transform(feat)
        proba = model.predict_proba(feat_s)[0]
        idx = int(np.argmax(proba))
        pred_class = model.classes_[idx] if hasattr(model, "classes_") else classes[idx]
        confidence = float(proba[idx])
        rows.append((sample_id, pred_class, round(confidence, 4)))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "predicted_class", "confidence"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
