#!/usr/bin/env python3
"""Classify a large unlabeled image folder with a fine-tuned CLIP model.

Outputs:
  - CSV with predicted label, confidence, top-3 labels, and OpenCV metrics.
  - Predicted-label folders using symlink or copy mode.

For 300k images, symlink mode is recommended to avoid duplicating huge data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advisor_clip_pipeline.clip_utils import (  # noqa: E402
    aggregate_prompt_logits,
    compute_text_features,
    load_clip,
    load_labels_from_bundle,
    safe_folder_name,
)
from advisor_clip_pipeline.image_metrics import compute_image_metrics  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def place_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        try:
            destination.symlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
        return
    raise ValueError(f"Unknown copy mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict labels for a large unlabeled image folder.")
    parser.add_argument("--best-model-dir", required=True, help="Directory containing clip_model/ and labels.json.")
    parser.add_argument("--unlabeled-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--copy-mode", choices=["none", "copy", "symlink"], default="symlink")
    parser.add_argument(
        "--copy-max-per-label",
        type=int,
        default=2000,
        help="Maximum images placed in each predicted folder. Use 0 for all images.",
    )
    parser.add_argument("--no-metrics", action="store_true", help="Skip OpenCV metrics for faster inference.")
    args = parser.parse_args()

    best_model_dir = Path(args.best_model_dir)
    model_dir = best_model_dir / "clip_model"
    unlabeled_dir = Path(args.unlabeled_dir)
    output_dir = Path(args.output_dir)
    prediction_csv = output_dir / "predictions_with_metrics.csv"

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing fine-tuned model folder: {model_dir}")
    if not unlabeled_dir.exists():
        raise FileNotFoundError(f"Missing unlabeled image folder: {unlabeled_dir}")

    labels = load_labels_from_bundle(best_model_dir)
    pattern = "**/*" if args.recursive else "*"
    image_paths = sorted(path for path in unlabeled_dir.glob(pattern) if is_image(path))
    if args.limit_images:
        image_paths = image_paths[: args.limit_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_root = output_dir / "predicted_label_folders"
    device = choose_device(args.device)

    model, processor = load_clip(str(model_dir), device=device, local_files_only=True)
    model.eval()
    text_features, prompt_labels = compute_text_features(model, processor, labels, device)

    copied_per_label = {label: 0 for label in labels}
    predicted_counts = {label: 0 for label in labels}

    fieldnames = [
        "image_name",
        "image_path",
        "predicted_label",
        "confidence",
        "top2_label",
        "top2_confidence",
        "top3_label",
        "top3_confidence",
    ]
    metric_fieldnames = []

    with prediction_csv.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for start in tqdm(range(0, len(image_paths), args.batch_size), desc="predict unlabeled"):
            batch_paths = image_paths[start : start + args.batch_size]
            images = []
            kept_paths = []
            for path in batch_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                    kept_paths.append(path)
                except Exception:
                    continue
            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                prompt_logits = model.logit_scale.exp() * image_features @ text_features.T
                class_logits = aggregate_prompt_logits(prompt_logits, prompt_labels, labels)
                probs = class_logits.softmax(dim=1).cpu()

            rows = []
            for path, row_probs in zip(kept_paths, probs):
                top = torch.topk(row_probs, k=min(3, len(labels)))
                top_indices = top.indices.tolist()
                top_values = top.values.tolist()
                pred_label = labels[int(top_indices[0])]
                predicted_counts[pred_label] += 1

                row = {
                    "image_name": path.name,
                    "image_path": str(path),
                    "predicted_label": pred_label,
                    "confidence": float(top_values[0]),
                    "top2_label": labels[int(top_indices[1])] if len(top_indices) > 1 else "",
                    "top2_confidence": float(top_values[1]) if len(top_values) > 1 else "",
                    "top3_label": labels[int(top_indices[2])] if len(top_indices) > 2 else "",
                    "top3_confidence": float(top_values[2]) if len(top_values) > 2 else "",
                }
                if not args.no_metrics:
                    metrics = compute_image_metrics(path)
                    row.update(metrics)
                    if not metric_fieldnames:
                        metric_fieldnames = list(metrics.keys())

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=fieldnames + metric_fieldnames)
                    writer.writeheader()

                rows.append(row)

                can_place = args.copy_mode != "none" and (
                    args.copy_max_per_label == 0 or copied_per_label[pred_label] < args.copy_max_per_label
                )
                if can_place:
                    destination = predicted_root / safe_folder_name(pred_label) / path.name
                    place_image(path, destination, args.copy_mode)
                    copied_per_label[pred_label] += 1

            if writer is not None:
                writer.writerows(rows)
                f.flush()

    summary = {
        "best_model_dir": str(best_model_dir),
        "unlabeled_dir": str(unlabeled_dir),
        "output_dir": str(output_dir),
        "n_images_found": len(image_paths),
        "predicted_counts": dict(sorted(predicted_counts.items())),
        "copied_per_label": dict(sorted(copied_per_label.items())),
        "copy_mode": args.copy_mode,
        "copy_max_per_label": args.copy_max_per_label,
        "metrics_included": not args.no_metrics,
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
