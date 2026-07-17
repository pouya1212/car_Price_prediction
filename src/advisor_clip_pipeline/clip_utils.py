"""CLIP loading, prompts, training-free scoring, and saving helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor


PROMPT_TEMPLATES = [
    "a dealership vehicle listing photo showing {label}",
    "a car sales image of {label}",
    "a vehicle photo whose main subject is {label}",
    "a used car listing image of {label}",
]


def label_to_text(label: str) -> str:
    """Turn folder-style labels into text CLIP can understand."""

    label = label.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def safe_folder_name(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def load_clip(model_name_or_path: str, device: str, local_files_only: bool = True):
    processor = CLIPProcessor.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model = CLIPModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model.to(device)
    return model, processor


def build_prompts(labels: list[str]) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    prompt_labels: list[str] = []
    for label in labels:
        label_text = label_to_text(label)
        for template in PROMPT_TEMPLATES:
            prompts.append(template.format(label=label_text))
            prompt_labels.append(label)
    return prompts, prompt_labels


@torch.no_grad()
def compute_text_features(model: CLIPModel, processor: CLIPProcessor, labels: list[str], device: str):
    prompts, prompt_labels = build_prompts(labels)
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return text_features, prompt_labels


def aggregate_prompt_logits(logits: torch.Tensor, prompt_labels: list[str], labels: list[str]) -> torch.Tensor:
    """Average prompt logits back to one score per class label."""

    class_logits = []
    for label in labels:
        idxs = [i for i, prompt_label in enumerate(prompt_labels) if prompt_label == label]
        class_logits.append(logits[:, idxs].mean(dim=1))
    return torch.stack(class_logits, dim=1)


def freeze_for_finetuning(
    model: CLIPModel,
    unfreeze_vision_layers: int = 2,
    train_visual_projection: bool = True,
    train_logit_scale: bool = True,
) -> None:
    """Freeze most CLIP weights and unfreeze a small vision tail.

    This is conservative fine-tuning: it adapts image understanding to the car
    taxonomy without retraining the whole model from scratch.
    """

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_vision_layers > 0:
        layers = list(model.vision_model.encoder.layers)
        for layer in layers[-unfreeze_vision_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    if train_visual_projection and hasattr(model, "visual_projection"):
        for param in model.visual_projection.parameters():
            param.requires_grad = True

    if train_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True


def make_tensors_contiguous(model: CLIPModel) -> None:
    """Avoid safetensors save errors from non-contiguous fine-tuned tensors."""

    state = model.state_dict()
    for name, tensor in state.items():
        if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
            tensor.data = tensor.contiguous()


def save_model_bundle(
    output_dir: str | Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    labels: list[str],
    extra_summary: dict,
) -> None:
    output_dir = Path(output_dir)
    clip_dir = output_dir / "clip_model"
    clip_dir.mkdir(parents=True, exist_ok=True)
    make_tensors_contiguous(model)
    model.save_pretrained(clip_dir, safe_serialization=True)
    processor.save_pretrained(clip_dir)
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (output_dir / "training_summary.json").write_text(json.dumps(extra_summary, indent=2), encoding="utf-8")


def load_labels_from_bundle(best_model_dir: str | Path) -> list[str]:
    best_model_dir = Path(best_model_dir)
    label_path = best_model_dir / "labels.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels.json in model bundle: {label_path}")
    return list(json.loads(label_path.read_text(encoding="utf-8")))
