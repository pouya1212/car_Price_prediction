#!/usr/bin/env python3
"""Build a labeled CSV from an Organized/<label>/ image folder tree.

This keeps every label and every image. It does not discard rare classes.
Optionally it also computes OpenCV image metrics for every labeled image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advisor_clip_pipeline.image_metrics import compute_image_metrics  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a manual-label CSV from folder names and optionally add OpenCV metrics."
    )
    parser.add_argument("--organized-dir", required=True, help="Folder containing one subfolder per label.")
    parser.add_argument("--output-csv", required=True, help="CSV to write.")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path.")
    parser.add_argument("--compute-metrics", action="store_true", help="Add OpenCV metrics columns.")
    args = parser.parse_args()

    organized_dir = Path(args.organized_dir)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")

    if not organized_dir.exists():
        raise FileNotFoundError(f"Organized folder does not exist: {organized_dir}")

    label_dirs = sorted(path for path in organized_dir.iterdir() if path.is_dir())
    if not label_dirs:
        raise RuntimeError(f"No label folders found in: {organized_dir}")

    rows: list[dict] = []
    duplicate_names: list[str] = []
    seen_names: set[str] = set()
    label_counts: Counter[str] = Counter()

    for label_dir in label_dirs:
        label = label_dir.name
        images = sorted(path for path in label_dir.rglob("*") if is_image(path))
        for image_path in images:
            if image_path.name in seen_names:
                duplicate_names.append(image_path.name)
            seen_names.add(image_path.name)
            label_counts[label] += 1
            row = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "manual_label": label,
                "label_folder": str(label_dir),
            }
            rows.append(row)

    if args.compute_metrics:
        enriched_rows = []
        for row in tqdm(rows, desc="opencv metrics"):
            metrics = compute_image_metrics(row["image_path"])
            enriched_rows.append({**row, **metrics})
        rows = enriched_rows

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["image_name", "image_path", "manual_label", "label_folder"]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "organized_dir": str(organized_dir),
        "output_csv": str(output_csv),
        "n_rows": len(rows),
        "n_unique_image_names": len(seen_names),
        "n_labels": len(label_counts),
        "label_counts": dict(sorted(label_counts.items())),
        "n_duplicate_image_names": len(duplicate_names),
        "duplicate_image_names_preview": sorted(set(duplicate_names))[:50],
        "metrics_included": bool(args.compute_metrics),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
