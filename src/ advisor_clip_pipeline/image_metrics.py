"""OpenCV image-quality and composition metrics.

These metrics are not labels. They are descriptive columns that help diagnose
image quality and can be reused later as tabular visual features.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - message is for cluster users
        raise RuntimeError(
            "OpenCV is required for image metrics. Install opencv-python-headless "
            "or run in the prepared conda environment."
        ) from exc
    return cv2


def _safe_float(value: float) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def compute_image_metrics(image_path: str | Path) -> dict[str, float | int | str]:
    """Compute lightweight OpenCV metrics for one image.

    The columns intentionally mirror the spirit of the earlier implementation:
    brightness, saturation, clarity, texture, visual balance, color warmth, and
    simple composition/edge measurements.
    """

    cv2 = _require_cv2()
    path = Path(image_path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {
            "metric_error": "could_not_read_image",
            "width": 0,
            "height": 0,
        }

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))
    saturation_std = float(np.std(hsv[:, :, 1]))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    image_clarity = float(laplacian.var())
    texture_contrast = float(gray.std())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    diagonal_energy = np.mean(np.abs(sobel_x + sobel_y)) + np.mean(np.abs(sobel_x - sobel_y))
    straight_energy = np.mean(np.abs(sobel_x)) + np.mean(np.abs(sobel_y)) + 1e-9
    diagonal_dominance = float(diagonal_energy / straight_energy)

    # Rule-of-thirds score: how much edge content lies near thirds lines.
    thirds_mask = np.zeros_like(gray, dtype=np.uint8)
    band = max(2, min(width, height) // 80)
    for x in (width // 3, 2 * width // 3):
        thirds_mask[:, max(0, x - band) : min(width, x + band + 1)] = 1
    for y in (height // 3, 2 * height // 3):
        thirds_mask[max(0, y - band) : min(height, y + band + 1), :] = 1
    edge_pixels = edges > 0
    rule_of_thirds_score = float(np.mean(thirds_mask[edge_pixels] > 0)) if edge_pixels.any() else 0.0

    # Visual balance from brightness/edge mass center. 1.0 means centered.
    mass = gray.astype(np.float64) + (edges > 0).astype(np.float64) * 255.0
    total_mass = float(mass.sum()) + 1e-9
    yy, xx = np.indices(gray.shape)
    center_x = float((xx * mass).sum() / total_mass)
    center_y = float((yy * mass).sum() / total_mass)
    visual_balance_x = 1.0 - min(abs(center_x - width / 2.0) / max(width / 2.0, 1.0), 1.0)
    visual_balance_y = 1.0 - min(abs(center_y - height / 2.0) / max(height / 2.0, 1.0), 1.0)

    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32)
    warm_mask = ((hue <= 60.0) | (hue >= 330.0)) & (sat > 40)
    warm_hue_ratio = float(np.mean(warm_mask))

    channel_means = image_rgb.reshape(-1, 3).mean(axis=0)
    color_difference = float(np.mean(np.abs(channel_means - channel_means.mean())))

    h_mid = height // 2
    w_mid = width // 2
    quads = [
        gray[:h_mid, :w_mid],
        gray[:h_mid, w_mid:],
        gray[h_mid:, :w_mid],
        gray[h_mid:, w_mid:],
    ]
    area_difference = float(np.std([float(q.mean()) if q.size else 0.0 for q in quads]))
    texture_difference = float(np.std([float(q.std()) if q.size else 0.0 for q in quads]))

    # A simple bounded quality index. It is meant for ranking/filtering, not truth.
    clarity_score = min(image_clarity / 1200.0, 1.0)
    exposure_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
    saturation_score = min(saturation / 90.0, 1.0)
    image_quality_score = 100.0 * (
        0.45 * clarity_score + 0.35 * exposure_score + 0.20 * saturation_score
    )

    return {
        "metric_error": "",
        "width": int(width),
        "height": int(height),
        "aspect_ratio": _safe_float(width / max(height, 1)),
        "mean_saturation": _safe_float(saturation),
        "saturation_std": _safe_float(saturation_std),
        "mean_brightness": _safe_float(brightness),
        "brightness_std": _safe_float(brightness_std),
        "image_clarity_laplacian_var": _safe_float(image_clarity),
        "texture_contrast_gray_std": _safe_float(texture_contrast),
        "edge_density": _safe_float(edge_density),
        "diagonal_dominance": _safe_float(diagonal_dominance),
        "rule_of_thirds_score": _safe_float(rule_of_thirds_score),
        "visual_balance_x": _safe_float(visual_balance_x),
        "visual_balance_y": _safe_float(visual_balance_y),
        "warm_hue_ratio": _safe_float(warm_hue_ratio),
        "area_difference": _safe_float(area_difference),
        "color_difference": _safe_float(color_difference),
        "texture_difference": _safe_float(texture_difference),
        "image_quality_score": _safe_float(image_quality_score),
    }
