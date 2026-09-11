"""
Feature extraction for CropGuard leaf-disease classification.

Design goals (per competition rules: hidden test has distribution shift +
ambiguous/unknown cases, so raw-pixel / overfit-prone features are avoided):
  - Color: HSV + Lab histograms and per-channel moments (mean/std/skew).
    Captures disease-associated discoloration; HSV/Lab are more lighting-robust
    than raw RGB.
  - Texture: Local Binary Pattern histogram (uniform, rotation-invariant) +
    Haralick/GLCM contrast-energy-homogeneity stats. Captures lesion texture
    (blight/rust/leaf_spot look different in texture, not just color).
  - Shape/edge: HOG on a small resized grayscale image. Captures spot shape
    and vein/lesion patterns without being pixel-exact (robust to jitter).
  - Everything is scale/orientation-normalized by resizing to a fixed size
    and using rotation-invariant texture descriptors, to help with the
    "nuisance condition" shift mentioned in the rules.
"""
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

IMG_SIZE = 128
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS


def _color_hist(img_channel, bins=16, rng=(0, 256)):
    hist, _ = np.histogram(img_channel, bins=bins, range=rng)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def _moments(channel):
    c = channel.astype(np.float32).ravel()
    mean = c.mean()
    std = c.std()
    skew = (((c - mean) ** 3).mean() / (std ** 3 + 1e-6))
    return np.array([mean, std, skew], dtype=np.float32)


def _glcm_features(gray):
    # Quantize to fewer gray levels for a stable, fast GLCM
    q = (gray / 32).astype(np.uint8)  # 0..7
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=8, symmetric=True, normed=True)
    feats = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        feats.append(graycoprops(glcm, prop).mean())
    return np.array(feats, dtype=np.float32)


def extract_features(bgr_img):
    """bgr_img: HxWx3 uint8 (OpenCV BGR). Returns a 1D float32 feature vector."""
    img = cv2.resize(bgr_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = []

    # Color histograms (HSV: H,S,V ; Lab: a,b channels which carry disease-color info)
    for i, rng in enumerate([(0, 180), (0, 256), (0, 256)]):
        feats.append(_color_hist(hsv[:, :, i], bins=16, rng=rng))
    for i in [1, 2]:  # a, b channels of Lab
        feats.append(_color_hist(lab[:, :, i], bins=16, rng=(0, 256)))

    # Color moments (mean/std/skew) per HSV + Lab channel
    for i in range(3):
        feats.append(_moments(hsv[:, :, i]))
    for i in range(3):
        feats.append(_moments(lab[:, :, i]))

    # LBP texture histogram (rotation-invariant "uniform" variant)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    feats.append(lbp_hist.astype(np.float32))

    # GLCM texture stats
    feats.append(_glcm_features(gray))

    # HOG shape descriptor on a smaller grayscale crop (coarse, robust cell size)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hog_feat = hog(small, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    feature_vector=True)
    feats.append(hog_feat.astype(np.float32))

    return np.concatenate(feats).astype(np.float32)


def load_and_extract(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return extract_features(img)
