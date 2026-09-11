"""Lightweight augmentations to simulate the 'nuisance condition shift' the
hidden test set is expected to have (rules: robustness to unseen conditions
matters more than raw validation score)."""
import numpy as np
import cv2


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _brightness_contrast(img, alpha, beta):
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


def random_augment(img, rng):
    """Apply a random combination of mild augmentations."""
    out = img.copy()
    if rng.random() < 0.7:
        out = _rotate(out, rng.uniform(-25, 25))
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.7:
        alpha = rng.uniform(0.75, 1.25)
        beta = rng.uniform(-25, 25)
        out = _brightness_contrast(out, alpha, beta)
    if rng.random() < 0.4:
        out = _gaussian_noise(out, rng.uniform(3, 12))
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = _blur(out, k)
    return out
