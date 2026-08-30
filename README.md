# DINOv3 + Mask2Former for Mapillary Instance Segmentation

Research code for the SSY340 project at Chalmers University of Technology. The
project replaces the standard Mask2Former backbone with a DINOv3 visual
backbone and explores semantic and instance segmentation on Mapillary Vistas.

The current `main` branch contains the work from both the exploratory
`create_pipeline` branch and the later `Instance_Segmentation` branch. The
instance-segmentation pipeline is the main path; the earlier scripts and
notebooks are retained as experiments and diagnostics.

## Project overview

The main model is a Mask2Former instance-segmentation head connected to a
DINOv3 backbone:

1. DINOv3 produces patch features from several transformer depths.
2. Adapter layers convert those features to the multi-scale channel sizes
   expected by Mask2Former.
3. Mask2Former predicts class logits and an instance mask for each query.
4. The training loop evaluates segmentation mAP and saves Hugging Face-style
   checkpoints.

The Small+ implementation uses an enhanced adapter with convolutional
projections, squeeze-and-excitation channel attention, and optional feature
fusion. The Large implementation provides the corresponding DINOv3-Large
backbone adapter. By default, the DINOv3 backbone is frozen while the adapter,
decoder, and prediction heads are trained with differential learning rates.

This is experimental coursework code. Paths, class mappings, hardware
settings, and model checkpoints must be adapted to the local environment.

## Repository layout

| Path | Purpose |
| --- | --- |
| `mask2former_dinov3_no_trainer_coco.py` | Main training and checkpointing script; supports local Mapillary data and COCO-style datasets. |
| `models/mask2former_dinov3_vitsmallplus.py` | DINOv3 Small+ + Mask2Former model and enhanced adapter. |
| `models/mask2former_dinov3_vitlarge.py` | DINOv3 Large + Mask2Former model and adapter. |
| `Mapillary_loading_Instance_Segmentation.py` | Standalone Mapillary instance-mask loader and visualization utilities. |
| `evaluation.py` | Validation/evaluation utilities: instance mAP, mIoU, pixel accuracy, and PR curves at IoU 0.50/0.75. |
| `simple_inference.py` | Command-line visualization for single images or directories. |
| `gradio_dinov3_segmentation.py` | Experimental enhanced-demo generator retained from the latest branch. |
| `enhanced_gradio_segmentation.py` | Interactive Gradio demo for a trained Small+ checkpoint. |
| `Segmentation_pipeline.py` | Earlier DINOv3 patch-feature semantic-segmentation prototype. |
| `Instance_segmentation_pipeline_v0.py` | Early DINOv3/Mask2Former exploration notebook-style script. |
| `dataset_exploration.py` | Initial dataset inspection experiments. |
| `*.ipynb` | Exploratory notebooks and visualization experiments. |

## Requirements

- Python 3.11 or newer
- PyTorch with a working CPU, CUDA, or Apple Silicon/MPS installation
- Hugging Face access to the DINOv3 checkpoints and the base Mask2Former
  checkpoints
- Mapillary Vistas v2.0 for training/evaluation

Install the Python dependencies with `uv`:

```bash
uv sync
```

If `uv` is not installed, install it using the method appropriate for your
platform, then run the command above. The committed `uv.lock` records the
resolved environment used by the project.

The first model load downloads pretrained weights from Hugging Face. DINOv3
may require accepting the model terms and authenticating with the Hugging Face
CLI before it can be downloaded.

## Dataset preparation

The training script expects a local Mapillary Vistas directory with this
structure:

```text
mapillary_dataset/
├── config_v2.0.json
├── training/
│   ├── images/*.jpg
│   └── v2.0/instances/*.png
└── validation/
    ├── images/*.jpg
    └── v2.0/instances/*.png
```

The dataset is not included in this repository. Instance PNG values encode a
semantic label and an instance identifier; the loader converts these values
to contiguous instance masks and semantic class IDs for Mask2Former.

The current training implementation intentionally limits experiments to the
first 3,000 training images and first 600 validation images. Change the
`N_train` and `N_val` limits in `mask2former_dinov3_no_trainer_coco.py` when a
different split size is required.

## Training

The training script reads a JSON configuration. Create a local configuration
file, for example `train_config.json`:

```json
{
  "model": "models/mask2former_dinov3_vitsmallplus.py",
  "dataset_name": "./mapillary_dataset",
  "output_dir": "./output/dinov3-smallplus-mapillary",
  "image_height": 512,
  "image_width": 512,
  "num_labels": 12,
  "num_train_epochs": 25,
  "per_device_train_batch_size": 4,
  "per_device_eval_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 0.0001,
  "checkpointing_steps": "200",
  "seed": 42,
  "push_to_hub": false
}
```

Launch training with:

```bash
uv run accelerate launch mask2former_dinov3_no_trainer_coco.py \
  --config train_config.json
```

The configuration can be overridden from the command line:

```bash
uv run accelerate launch mask2former_dinov3_no_trainer_coco.py \
  --config train_config.json \
  --model models/mask2former_dinov3_vitlarge.py \
  --image_height 1024 \
  --image_width 1024 \
  --output_dir ./output/dinov3-large-mapillary
```

Important training options include:

- `--checkpointing_steps 200` saves state and model checkpoints every 200
  update steps; use `epoch` to save at epoch boundaries.
- `--resume_from_checkpoint PATH` resumes an interrupted run.
- `--gradient_accumulation_steps N` increases the effective batch size.
- `--freeze_backbone` freezes DINOv3, which is the default experiment setup.
- `--use_gradient_checkpointing` reduces memory use at the cost of speed.
- `--report_to wandb` enables Weights & Biases logging when configured.

Each saved model directory contains the model configuration, weights, image
processor configuration, and metrics needed by the inference scripts. When
training finishes, the best checkpoint is written below
`<output_dir>/best_model/`.

## Inference

Use a completed `best_model` directory (or another `save_pretrained` model
directory) for inference. The directory must contain both model weights and a
compatible processor/configuration.

For one image:

```bash
uv run python simple_inference.py \
  --model_path ./output/dinov3-smallplus-mapillary/best_model \
  --image_path ./example.jpg \
  --output ./example_result.png \
  --threshold 0.5
```

For a directory:

```bash
uv run python simple_inference.py \
  --model_path ./output/dinov3-smallplus-mapillary/best_model \
  --input_dir ./images \
  --output_dir ./results \
  --batch \
  --threshold 0.5
```

Add `--recursive` to process nested directories. The visualization overlays
predicted masks and class names on the input image.

## Gradio demo

Launch the enhanced interactive demo with the trained checkpoint path
explicitly set:

```bash
uv run python enhanced_gradio_segmentation.py \
  --model_path ./output/dinov3-smallplus-mapillary/best_model \
  --gradio \
  --port 7860
```

Use `--share` only when a public Gradio link is wanted. The interface supports
an uploaded image, a confidence threshold, and automatic visualization of the
detected instances.

The latest branch also keeps `gradio_dinov3_segmentation.py` as a generator
for the enhanced script; run it without arguments only if that generated file
needs to be recreated.

## Evaluation

`evaluation.py` is configured as a research utility rather than a fully
parameterized CLI. Before running it, update the `model_path` and dataset
variables in `main()`:

```bash
uv run python evaluation.py
```

The evaluation code reports:

- segmentation mAP, including mAP@0.50 and mAP@0.75;
- mean IoU and per-class IoU in the baseline evaluator;
- mean foreground pixel accuracy; and
- per-class precision/recall curves at IoU 0.50 and 0.75.

The latest evaluation experiment also writes JSON-compatible results to
`metrics_results.json`. PR plots are written to `pr_curves/`. Evaluation and
training outputs are ignored by Git so that checkpoints and generated figures
do not accidentally enter the source repository.

## Reproducibility notes

- Set `seed` in the training configuration when comparing runs.
- DINOv3 and Mask2Former weights are downloaded and cached by Hugging Face.
- Select image dimensions that are compatible with the DINOv3 patch size;
  the supplied configurations use 512×512 or 1024×1024 inputs.
- Batch size and worker count may need to be reduced on laptops or MPS.
- The notebooks and prototype scripts contain assumptions from the original
  experiments; pass paths explicitly when running them.

## Credits and licenses

The training loop is adapted from the Hugging Face Transformers
instance-segmentation example. The project also uses DINOv3 and Mask2Former
pretrained components. Respect the licenses and model terms of each upstream
component when downloading, modifying, or redistributing weights.
