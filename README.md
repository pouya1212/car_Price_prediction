# CLIP + OpenCV Holdout Pipeline for Dealership Images

This project fine-tunes CLIP on manually annotated car-image labels and then
uses the saved model to classify a large unlabeled image folder.

It is designed for a fully offline/free university-cluster workflow once the
Hugging Face model files are cached locally.

## Labels Used

The code keeps every folder label. It does not drop rare labels.

Expected labels include:

- `Car Symbol_Brand`
- `dealership`
- `document`
- `doors`
- `engine_bay`
- `front_left_quarter`
- `front_right_quarter`
- `front_view`
- `graphic_advertisement`
- `interior_cargo`
- `interior_dashboard`
- `interior_seats`
- `left_side`
- `lights`
- `odometer_speedometer_detail`
- `rear_left_quarter`
- `rear_right_quarter`
- `rear_view`
- `right_side`
- `roof_detail`
- `wheel_closeup`

## Recommended Model Choices

### Default: `openai/clip-vit-base-patch32`

Best first choice. It is fast, fits easily on one GPU, and has already worked
well in the earlier experiments.

### Larger: `openai/clip-vit-large-patch14`

Potentially stronger because it has a larger vision encoder and smaller patch
size, but it needs more GPU memory and must be cached first. Use it after the
base model is working.

### Why CLIP?

CLIP maps images and text labels into the same feature space. We fine-tune it
using your manual folder labels so the model learns your advisor's exact
taxonomy instead of generic internet labels.

## What the Pipeline Produces

### Training Step

The training script:

1. Reads all organized labeled images.
2. Builds a CSV with:
   - image name
   - image path
   - manual label
   - OpenCV image metrics
3. Splits the labeled data into train and validation sets.
4. Fine-tunes CLIP on the train split.
5. Selects the best checkpoint using validation balanced accuracy.
6. Saves:
   - best model
   - baseline zero-shot metrics
   - fine-tuned metrics
   - validation predictions
   - confusion matrix
   - predicted validation image folders
   - misclassified validation image folders

### Large Unlabeled Prediction Step

The prediction script:

1. Loads the saved fine-tuned CLIP model.
2. Reads a large unlabeled image folder, such as 300,000 images.
3. Predicts one of the same manual labels for each image.
4. Computes OpenCV image metrics for each image.
5. Writes a complete CSV with prediction + metrics.
6. Creates predicted-label folders using symlinks by default.

Symlink mode is recommended for 300k images because it avoids copying hundreds
of thousands of files.

## Main Scripts

### 1. Build labeled CSV only

```bash
python scripts/build_labels_csv_from_folders.py \
  --organized-dir "/path/to/Annotated Data_complete/Organized" \
  --output-csv data/manual_labels_complete_with_opencv_metrics.csv \
  --compute-metrics
```

### 2. Train/fine-tune CLIP

```bash
sbatch scripts/run_slurm_train_clip_holdout.sh
```

Important environment variables:

```bash
MODEL_ID="openai/clip-vit-base-patch32"
EPOCHS=35
VAL_SIZE=0.15
BATCH_SIZE=128
LR=2e-5
UNFREEZE_LAYERS=2
```

Example with CLIP large:

```bash
MODEL_ID="openai/clip-vit-large-patch14" \
RUN_NAME="clip_large_holdout_25_epochs" \
EPOCHS=25 \
BATCH_SIZE=64 \
LR=1e-5 \
sbatch scripts/run_slurm_train_clip_holdout.sh
```

### 3. Predict/cluster large unlabeled folder

```bash
sbatch scripts/run_slurm_predict_large_unlabeled.sh
```

For a 300k-image folder:

```bash
UNLABELED_DIR="/path/to/300k_unlabeled_images" \
OUTPUT_DIR="$PWD/Output/advisor_300k_predictions" \
COPY_MODE="symlink" \
COPY_MAX_PER_LABEL=2000 \
sbatch scripts/run_slurm_predict_large_unlabeled.sh
```

To create folders for all images, set:

```bash
COPY_MAX_PER_LABEL=0
```

Be careful: copying or symlinking all 300k images creates many filesystem
entries. The CSV always contains every prediction regardless of this limit.

## Important Outputs

Training:

```text
Output/clip_base_holdout_35_epochs/experiment_summary.json
Output/clip_base_holdout_35_epochs/baseline_zero_shot_metrics.json
Output/clip_base_holdout_35_epochs/finetuned_metrics.json
Output/clip_base_holdout_35_epochs/finetuned_confusion_matrix.csv
Output/clip_base_holdout_35_epochs/best_model/
Output/clip_base_holdout_35_epochs/predicted_validation_images/
Output/clip_base_holdout_35_epochs/misclassified_validation_images/
```

Large unlabeled prediction:

```text
Output/large_unlabeled_predictions/predictions_with_metrics.csv
Output/large_unlabeled_predictions/prediction_summary.json
Output/large_unlabeled_predictions/predicted_label_folders/
```

## OpenCV Metrics in the CSV

The CSV includes descriptive image metrics such as:

- width, height, aspect ratio
- brightness and brightness standard deviation
- saturation and saturation standard deviation
- Laplacian clarity/sharpness
- gray texture contrast
- edge density
- diagonal dominance
- rule-of-thirds score
- visual balance x/y
- warm hue ratio
- area/color/texture differences
- image quality score

These metrics are not the classification target. They help describe image
quality and can be used later for filtering, diagnostics, or tabular modeling.

## Offline Use

The scripts default to offline mode. The model must already exist under:

```text
.hf_cache
```

If a model is not cached, download it once with internet enabled, then rerun in
offline mode.
