import glob
import os
import time
import numpy as np
import cv2
import joblib

from features import extract_features
from augment import random_augment

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

DATA_ROOT = "/home/claude/cropguard/CropGuard"
CLASSES = ["blight", "healthy", "leaf_spot", "mildew", "mosaic", "rust"]
AUG_PER_IMAGE = 2  # extra augmented copies per training image
SEED = 42

rng = np.random.default_rng(SEED)


def load_split(split, augment=False):
    X, y = [], []
    for c in CLASSES:
        paths = sorted(glob.glob(os.path.join(DATA_ROOT, split, c, "*.png")))
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            X.append(extract_features(img))
            y.append(c)
            if augment:
                for _ in range(AUG_PER_IMAGE):
                    aug = random_augment(img, rng)
                    X.append(extract_features(aug))
                    y.append(c)
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    t0 = time.time()
    print("Loading + extracting features (train, with augmentation)...")
    X_train, y_train = load_split("train", augment=True)
    print(f"  train features: {X_train.shape}, time={time.time()-t0:.1f}s")

    print("Loading + extracting features (validation, no augmentation)...")
    X_val, y_val = load_split("validation", augment=False)
    print(f"  val features: {X_val.shape}, time={time.time()-t0:.1f}s")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    candidates = {
        "svm_rbf": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=SEED),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6, random_state=SEED),
    }

    fitted = {}
    print("\n--- Individual model validation accuracy ---")
    for name, clf in candidates.items():
        t1 = time.time()
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_val_s)
        acc = accuracy_score(y_val, pred)
        print(f"{name:15s} val_acc={acc:.4f}  ({time.time()-t1:.1f}s)")
        fitted[name] = (clf, acc)

    # Soft-voting ensemble of the three (all support predict_proba)
    ensemble = VotingClassifier(
        estimators=[(n, c) for n, (c, _) in fitted.items()],
        voting="soft",
    )
    # VotingClassifier refits internally; reuse already-fitted estimators via voting='soft' + fit
    ensemble.fit(X_train_s, y_train)
    ens_pred = ensemble.predict(X_val_s)
    ens_acc = accuracy_score(y_val, ens_pred)
    print(f"{'ensemble':15s} val_acc={ens_acc:.4f}")

    print("\n--- Classification report (ensemble) ---")
    print(classification_report(y_val, ens_pred, digits=3))

    # Pick best of {individual models, ensemble} by validation accuracy
    best_name, best_acc, best_model = "ensemble", ens_acc, ensemble
    for name, (clf, acc) in fitted.items():
        if acc > best_acc:
            best_name, best_acc, best_model = name, acc, clf

    print(f"\nSelected best model: {best_name} (val_acc={best_acc:.4f})")

    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "classes": CLASSES,
        "model_name": best_name,
        "val_accuracy": best_acc,
    }, "/home/claude/build/cropguard_model.joblib")
    print(f"Saved model. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
