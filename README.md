# CLIP + OpenCV Car-Image View Classification

This project classifies dealership vehicle images into manually defined image-view and content categories such as `front_view`, `rear_view`, `interior_dashboard`, `engine_bay`, and `wheel_closeup`.

It fine-tunes the Hugging Face implementation of OpenAI CLIP on annotated folders, evaluates the model on a stratified holdout set, and uses the saved model to classify new unlabeled images. OpenCV metrics are also calculated for image-quality analysis.

The core pipeline is written in Python and runs on Windows, macOS, and Linux. Slurm is optional: the included `.sh` files only submit the Python scripts to a university cluster.

## Example prediction

The following unlabeled image was assigned to `interior_dashboard` by the trained model:

![Example prediction classified as interior dashboard](docs/images/example_interior_dashboard.jpg)

## What the project does

The pipeline has three main stages:

1. `scripts/build_labels_csv_from_folders.py`
   - Reads an annotated `Organized/<label>/<image>` directory.
   - Uses each subfolder name as the manual class label.
   - Optionally calculates OpenCV image metrics.
   - Writes the complete labeled dataset to CSV.

2. `scripts/train_clip_holdout.py`
   - Creates a stratified training/validation split.
   - Measures zero-shot CLIP performance before training.
   - Fine-tunes the last CLIP vision layers.
   - Selects the best checkpoint using validation balanced accuracy.
   - Saves metrics, predictions, confusion matrices, and review folders.

3. `scripts/predict_large_unlabeled.py`
   - Loads a saved `best_model` bundle.
   - Classifies new images into the same label set.
   - Writes predicted labels, confidence scores, top-three predictions, and OpenCV metrics to CSV.
   - Optionally creates one review folder per predicted class.

## Choose a workflow

### Workflow A: Train a new model

Use this workflow when annotated images are available in one subfolder per label. Training produces validation results and a portable `best_model` directory.

### Workflow B: Predict new images

Use this workflow when a trained `best_model` already exists. Prediction only requires:

- the `best_model` directory;
- a folder containing new images;
- the Python environment.

The original annotated dataset and the original Hugging Face cache are not required when predicting with a complete saved `best_model` bundle.

## Supported labels

The current dataset contains 21 labels:

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

The code keeps all folder labels, including rare classes.

## Annotated dataset format

Training images must be organized with one subfolder per label:

```text
Organized/
|-- front_view/
|   |-- image001.jpg
|   `-- image002.png
|-- rear_view/
|-- interior_dashboard/
|-- wheel_closeup/
`-- ...
```

Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, and `.tiff`.

## Platform support

| Platform | Recommended device | Notes |
|---|---|---|
| Windows with NVIDIA GPU | `cuda` or `auto` | Install a CUDA-enabled PyTorch build. |
| Windows without NVIDIA GPU | `cpu` | Works, but training is slower. |
| Apple Silicon Mac | `cpu` with the current scripts | Optional MPS support is described below. |
| Intel Mac | `cpu` | Apple MPS is generally unavailable. |
| Linux workstation | `cuda`, `auto`, or `cpu` | Run the Python scripts directly. |
| Slurm cluster | Usually `cuda` | Submit the optional shell wrappers with `sbatch`. |

No different classification algorithm is needed for Windows or macOS. Only the PyTorch compute device and installation command differ.

## Installation

Python 3.12 is the tested and recommended version. The commands below use [uv](https://docs.astral.sh/uv/), which supports Windows, macOS, and Linux.

### 1. Install uv

#### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell or VS Code after installation.

#### macOS or Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal after installation.

### 2. Create the virtual environment

Run these commands from the project directory on any operating system:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --torch-backend auto -r requirements.txt
```

The Transformers version is intentionally pinned to `4.57.6`. Transformers 5 changes the CLIP feature-return API and is not compatible with the current pipeline.

### 3. Activate the environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, enable scripts only for the current terminal and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
source .venv/bin/activate
```

After activation, the command `python` refers to the project environment on all platforms.

### 4. Verify the environment

```bash
python -c "import torch, transformers, cv2, pandas, sklearn, PIL; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('CUDA:', torch.cuda.is_available()); print('Apple MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

Expected acceleration:

- NVIDIA Windows/Linux computer: `CUDA: True`
- Apple Silicon Mac with an MPS-enabled PyTorch build: `Apple MPS: True`
- Other computers: CPU execution

## Build the annotated CSV

Run this stage only when training a new model. Replace `PATH_TO_ORGANIZED` with the annotated dataset path.

```bash
python scripts/build_labels_csv_from_folders.py --organized-dir "PATH_TO_ORGANIZED" --output-csv "data/manual_labels_complete_with_opencv_metrics.csv" --summary-json "data/manual_labels_complete_summary.json" --compute-metrics
```

Examples of valid dataset paths:

```text
Windows: C:\Users\name\Dataset\Organized
macOS:   /Users/name/Dataset/Organized
Linux:   /home/name/Dataset/Organized
```

The command creates:

```text
data/manual_labels_complete_with_opencv_metrics.csv
data/manual_labels_complete_summary.json
```

OpenCV metrics are descriptive features for image quality and diagnostics. They are not the CLIP classification target.

## Download and cache CLIP once

The training script defaults to local/offline model loading. The first run must be allowed to download `openai/clip-vit-base-patch32`.

Set a project-local Hugging Face cache before the first training run.

### Windows PowerShell

```powershell
$env:HF_HOME = "$PWD\.hf_cache"
```

### macOS or Linux

```bash
export HF_HOME="$PWD/.hf_cache"
```

Keep the same `HF_HOME` value in later terminals so the script finds the cached model.

## One-epoch smoke test

Always run a one-epoch test before a long experiment. This checks the dataset paths, model cache, dependencies, and available memory.

```bash
python scripts/train_clip_holdout.py --labels-csv "data/manual_labels_complete_with_opencv_metrics.csv" --image-column image_path --label-column manual_label --output-dir "Output/clip_smoke_test" --model "openai/clip-vit-base-patch32" --val-size 0.15 --epochs 1 --batch-size 16 --num-workers 0 --lr 2e-5 --unfreeze-vision-layers 2 --device auto --copy-validation-images --copy-max-per-label 100 --allow-download
```

Use `--allow-download` only when the pretrained model is not already cached.

## Full training

After the smoke test succeeds, run the complete experiment:

```bash
python scripts/train_clip_holdout.py --labels-csv "data/manual_labels_complete_with_opencv_metrics.csv" --image-column image_path --label-column manual_label --output-dir "Output/clip_base_holdout_35_epochs" --model "openai/clip-vit-base-patch32" --val-size 0.15 --epochs 35 --batch-size 16 --num-workers 0 --lr 2e-5 --unfreeze-vision-layers 2 --device auto --copy-validation-images --copy-max-per-label 400
```

Recommended starting batch sizes:

| Hardware | Starting batch size |
|---|---:|
| Windows RTX 3060, 12 GB | 16 |
| Apple Silicon Mac | 8 |
| CPU-only computer | 4-8 |
| Large cluster GPU | 64-128 |

If an out-of-memory error occurs, reduce `--batch-size`. The Slurm wrapper's batch size of 128 is a cluster setting, not a normal-PC default.

Use `--num-workers 0` on Windows and macOS. The current data-loader callback is not multiprocessing-safe under their spawn-based worker model.

## Training outputs

The training output directory contains:

```text
experiment_summary.json
baseline_zero_shot_metrics.json
baseline_zero_shot_validation_predictions.csv
finetuned_metrics.json
finetuned_validation_predictions.csv
finetuned_confusion_matrix.csv
best_model/
predicted_validation_images/
misclassified_validation_images/
```

The `best_model` directory is the portable trained model bundle used for later prediction.

## Predict unlabeled images

Prediction does not require the annotated dataset. Replace the three path placeholders below:

```bash
python scripts/predict_large_unlabeled.py --best-model-dir "PATH_TO_BEST_MODEL" --unlabeled-dir "PATH_TO_UNLABELED_IMAGES" --output-dir "PATH_TO_PREDICTION_OUTPUT" --batch-size 16 --device auto --copy-mode copy --copy-max-per-label 0
```

Use `--recursive` if images are located inside nested input subfolders:

```bash
python scripts/predict_large_unlabeled.py --best-model-dir "PATH_TO_BEST_MODEL" --unlabeled-dir "PATH_TO_UNLABELED_IMAGES" --output-dir "PATH_TO_PREDICTION_OUTPUT" --recursive --batch-size 16 --device auto --copy-mode copy --copy-max-per-label 0
```

Prediction produces:

```text
predictions_with_metrics.csv
prediction_summary.json
predicted_label_folders/
```

The CSV includes:

- predicted label and confidence;
- second- and third-ranked labels and confidence scores;
- source image name and path;
- OpenCV image-quality and composition metrics.

### Choosing a copy mode

- `--copy-mode copy`: easiest for a small review set; duplicates the images.
- `--copy-mode symlink`: avoids duplication, but Windows may require Developer Mode or administrator rights.
- `--copy-mode none`: recommended for very large datasets when only the CSV is needed.

`--copy-max-per-label 0` means place every prediction in a review folder. Use a positive limit for very large datasets.

## Optional Apple Silicon GPU support

The current scripts accept `auto`, `cpu`, and `cuda`. They run on macOS using CPU, but they do not yet automatically select Apple's MPS device.

To enable Apple Silicon GPU execution, update `choose_device` in both `scripts/train_clip_holdout.py` and `scripts/predict_large_unlabeled.py`:

```python
def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg
```

In both scripts, also change the device choices from:

```python
choices=["auto", "cpu", "cuda"]
```

to:

```python
choices=["auto", "cpu", "cuda", "mps"]
```

After that change, `--device auto` selects CUDA first, Apple MPS second, and CPU otherwise. PyTorch documents MPS support at <https://docs.pytorch.org/docs/stable/notes/mps>.

If a particular operation is not implemented for MPS, macOS users can try:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

or run with `--device cpu`.

## Optional Slurm cluster execution

The shell scripts are convenience wrappers around the same Python programs. Use them only on a system with Slurm:

```bash
sbatch scripts/run_slurm_train_clip_holdout.sh
sbatch scripts/run_slurm_predict_large_unlabeled.sh
```

Do not use `sbatch` on Windows or macOS. Run the corresponding Python scripts directly.

The Slurm scripts support environment variables such as:

```bash
MODEL_ID="openai/clip-vit-base-patch32"
EPOCHS=35
VAL_SIZE=0.15
BATCH_SIZE=128
LR=2e-5
UNFREEZE_LAYERS=2
```

These defaults assume cluster hardware and should not be copied directly to a normal PC.

## OpenCV metrics

The CSV outputs may include:

- width, height, and aspect ratio;
- brightness and brightness standard deviation;
- saturation and saturation standard deviation;
- Laplacian clarity/sharpness;
- grayscale texture contrast;
- edge density;
- diagonal dominance;
- rule-of-thirds score;
- horizontal and vertical visual balance;
- warm-hue ratio;
- area, color, and texture differences;
- image-quality score.

These metrics support filtering and diagnostics. CLIP classification is trained from the images and manual folder labels.

## Tested reference run

One completed reference experiment used:

- 8,820 labeled images;
- 7,497 training images;
- 1,323 validation images;
- 21 labels;
- 50 training epochs;
- `openai/clip-vit-base-patch32`;
- validation accuracy: approximately 93.20%;
- validation balanced accuracy: approximately 91.53%.

These are holdout-validation results from this dataset, not performance on an independent external test set.

## Troubleshooting

### `BaseModelOutputWithPooling` has no attribute `norm`

Transformers 5 was installed. Reinstall the pinned compatible version:

```bash
uv pip install "transformers==4.57.6"
```

### Cannot load the CLIP image processor

The Hugging Face cache is missing files or the current terminal is using a different `HF_HOME`. Set `HF_HOME` consistently and run the first training command with `--allow-download`.

### `.venv` Python is not found

Confirm that the terminal is in the project directory and that `.venv` exists. Activate the environment before using `python`.

### CUDA out of memory

Reduce `--batch-size`, for example from 16 to 8 or 4.

### Slow image processor warning

The message about a slow image processor is informational and does not stop training or prediction.

### Windows symlinks become copies

Windows may prevent symlink creation without Developer Mode or elevated rights. The prediction script falls back to copying. Use `--copy-mode none` when processing a very large folder and only the CSV is required.

## Offline use

Internet access is required for the initial Python package installation and the first pretrained CLIP download.

After the model is cached, training can use local files only. After fine-tuning, prediction loads the saved `best_model` bundle and can run offline without contacting Hugging Face.

