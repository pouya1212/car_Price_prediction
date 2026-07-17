#!/usr/bin/env python3
"""Fine-tune CLIP on labeled car images using train/validation split only.

This is for the current advisor workflow:
  - Train on 85-90% of the 8,820 labeled images.
  - Use the remaining labeled images only for validation/checkpoint selection.
  - Save the best model based on validation balanced accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advisor_clip_pipeline.clip_utils import (  # noqa: E402
    aggregate_prompt_logits,
    compute_text_features,
    freeze_for_finetuning,
    load_clip,
    safe_folder_name,
    save_model_bundle,
)


class ImageLabelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: list[str], image_column: str, label_column: str):
        self.frame = frame.reset_index(drop=True)
        self.labels = labels
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row[self.image_column]).convert("RGB")
        label = str(row[self.label_column])
        return {
            "image": image,
            "label_id": self.label_to_id[label],
            "label": label,
            "image_path": str(row[self.image_column]),
            "image_name": str(row.get("image_name", Path(row[self.image_column]).name)),
        }


def collate_batch(batch, processor):
    images = [item["image"] for item in batch]
    encoded = processor(images=images, return_tensors="pt", padding=True)
    encoded["labels"] = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
    encoded["image_path"] = [item["image_path"] for item in batch]
    encoded["image_name"] = [item["image_name"] for item in batch]
    encoded["true_label"] = [item["label"] for item in batch]
    return encoded


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def copy_review_images(predictions: list[dict], output_dir: Path, max_per_label: int) -> None:
    predicted_root = output_dir / "predicted_validation_images"
    missed_root = output_dir / "misclassified_validation_images"
    pred_counts: dict[str, int] = {}
    miss_counts: dict[str, int] = {}
    for row in predictions:
        pred = str(row["predicted_label"])
        true = str(row["true_label"])
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue

        pred_counts.setdefault(pred, 0)
        if pred_counts[pred] < max_per_label:
            folder = predicted_root / safe_folder_name(pred)
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, folder / image_path.name)
            pred_counts[pred] += 1

        if pred != true:
            key = f"{safe_folder_name(true)}__pred_{safe_folder_name(pred)}"
            miss_counts.setdefault(key, 0)
            if miss_counts[key] < max_per_label:
                folder = missed_root / key
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, folder / image_path.name)
                miss_counts[key] += 1


@torch.no_grad()
def predict_frame(model, processor, frame, labels, image_column, label_column, device, batch_size, num_workers):
    dataset = ImageLabelDataset(frame, labels, image_column, label_column)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, processor),
    )
    text_features, prompt_labels = compute_text_features(model, processor, labels, device)
    rows = []
    model.eval()
    for batch in tqdm(loader, desc="predict"):
        labels_tensor = batch["labels"].to(device)
        model_inputs = {k: v.to(device) for k, v in batch.items() if k in {"pixel_values", "attention_mask"}}
        image_features = model.get_image_features(**model_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        prompt_logits = model.logit_scale.exp() * image_features @ text_features.T
        class_logits = aggregate_prompt_logits(prompt_logits, prompt_labels, labels)
        probs = class_logits.softmax(dim=1)
        pred_ids = probs.argmax(dim=1).cpu().numpy()
        true_ids = labels_tensor.cpu().numpy()
        conf = probs.max(dim=1).values.cpu().numpy()
        for i, pred_id in enumerate(pred_ids):
            rows.append(
                {
                    "image_name": batch["image_name"][i],
                    "image_path": batch["image_path"][i],
                    "true_label": labels[int(true_ids[i])],
                    "predicted_label": labels[int(pred_id)],
                    "confidence": float(conf[i]),
                }
            )
    return rows


def metrics_from_predictions(rows: list[dict], labels: list[str]) -> dict:
    y_true = [row["true_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "image_path", "true_label", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP using train/validation holdout.")
    parser.add_argument("--labels-csv", required=True, help="CSV with image paths and manual labels.")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--label-column", default="manual_label")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="HF model id or local snapshot path.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction held out for validation.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face download. Default is offline/local only.")
    parser.add_argument("--unfreeze-vision-layers", type=int, default=2)
    parser.add_argument("--copy-validation-images", action="store_true")
    parser.add_argument("--copy-max-per-label", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    df = pd.read_csv(args.labels_csv)
    df = df.dropna(subset=[args.image_column, args.label_column]).copy()
    df[args.label_column] = df[args.label_column].astype(str)
    df[args.image_column] = df[args.image_column].astype(str)
    df = df[df[args.image_column].map(lambda p: Path(p).exists())].copy()

    labels = sorted(df[args.label_column].unique().tolist())
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=df[args.label_column],
    )

    model, processor = load_clip(args.model, device=device, local_files_only=not args.allow_download)

    # Baseline zero-shot evaluation before training.
    baseline_rows = predict_frame(
        model, processor, val_df, labels, args.image_column, args.label_column, device, args.batch_size, args.num_workers
    )
    baseline_metrics = metrics_from_predictions(baseline_rows, labels)
    write_predictions(output_dir / "baseline_zero_shot_validation_predictions.csv", baseline_rows)
    (output_dir / "baseline_zero_shot_metrics.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")

    freeze_for_finetuning(model, unfreeze_vision_layers=args.unfreeze_vision_layers)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = ImageLabelDataset(train_df, labels, args.image_column, args.label_column)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_batch(batch, processor),
    )

    best_balanced_accuracy = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        text_features, prompt_labels = compute_text_features(model, processor, labels, device)
        total_loss = 0.0
        n_seen = 0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}"):
            label_ids = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items() if k in {"pixel_values", "attention_mask"}}
            image_features = model.get_image_features(**model_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            prompt_logits = model.logit_scale.exp() * image_features @ text_features.T
            class_logits = aggregate_prompt_logits(prompt_logits, prompt_labels, labels)
            loss = torch.nn.functional.cross_entropy(class_logits, label_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(label_ids)
            n_seen += len(label_ids)

        val_rows = predict_frame(
            model, processor, val_df, labels, args.image_column, args.label_column, device, args.batch_size, args.num_workers
        )
        val_metrics = metrics_from_predictions(val_rows, labels)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_seen, 1),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        print(json.dumps(epoch_summary, indent=2))
        history.append(epoch_summary)

        if val_metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = val_metrics["balanced_accuracy"]
            save_model_bundle(
                output_dir / "best_model",
                model,
                processor,
                labels,
                {
                    "best_epoch": epoch,
                    "best_validation_metrics": val_metrics,
                    "model": args.model,
                    "val_size": args.val_size,
                },
            )
            write_predictions(output_dir / "finetuned_validation_predictions.csv", val_rows)
            (output_dir / "finetuned_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
            if args.copy_validation_images:
                copy_review_images(val_rows, output_dir, args.copy_max_per_label)

    final_rows = pd.read_csv(output_dir / "finetuned_validation_predictions.csv").to_dict("records")
    y_true = [row["true_label"] for row in final_rows]
    y_pred = [row["predicted_label"] for row in final_rows]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "finetuned_confusion_matrix.csv")

    experiment_summary = {
        "labels": labels,
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "model": args.model,
        "epochs": args.epochs,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "baseline_zero_shot": baseline_metrics,
        "finetuned": json.loads((output_dir / "finetuned_metrics.json").read_text(encoding="utf-8")),
        "history": history,
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")
    print(json.dumps(experiment_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
