# u-mamba-nnunet-benchmark

This folder is a repackaged, runnable code bundle with a cleaner structure. The goal is to make it easy to locate:
- U-Mamba training / inference / evaluation entrypoints
- nnU-Net training / inference / evaluation entrypoints
- Extra, paper/task-specific evaluation scripts (NSD/SurfaceDice/cell instance F1, etc.)

## Entrypoints (Most Important)

### U-Mamba
U-Mamba and nnU-Net share the same `nnunetv2/` framework codebase here. The only difference is the trainer name (`-tr`):

- Training: `nnUNetv2_train ... -tr nnUNetTrainerUMambaBot` or `-tr nnUNetTrainerUMambaEnc`
  - Training CLI implementation: `nnunetv2/run/run_training.py`
  - U-Mamba trainers:
    - `nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaBot.py`
    - `nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEnc.py`
  - U-Mamba networks:
    - `nnunetv2/nets/UMambaBot.py`
    - `nnunetv2/nets/UMambaEnc.py`

- Inference: `nnUNetv2_predict ... -tr nnUNetTrainerUMambaBot` or `-tr nnUNetTrainerUMambaEnc`
  - Inference CLI implementation: `nnunetv2/inference/predict_from_raw_data.py`

- Evaluation (nnUNetv2 built-in metrics / summary.json): `nnUNetv2_evaluate_folder ...`
  - Evaluation CLI implementation: `nnunetv2/evaluation/evaluate_predictions.py`

### nnU-Net
- Training: `nnUNetv2_train ... -tr nnUNetTrainer`
  - nnU-Net trainer: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

- Inference: `nnUNetv2_predict ... -tr nnUNetTrainer`
  - Inference CLI implementation: `nnunetv2/inference/predict_from_raw_data.py`

- Evaluation: `nnUNetv2_evaluate_folder ...`
  - Evaluation CLI implementation: `nnunetv2/evaluation/evaluate_predictions.py`

## Where the CLI Comes From
`setup.py` registers the command-line entry points (via `entry_points`). After running `pip install -e .` in this folder (typically on Linux), you get:
  - `nnUNetv2_train`
  - `nnUNetv2_predict`
  - `nnUNetv2_evaluate_folder`

## What `evaluation/` Is (Extra Scripts)
This `evaluation/` directory is not the nnUNetv2 built-in evaluation module. It contains additional evaluation scripts for specific tasks/datasets (often used for paper reproduction).

- `SurfaceDice.py`: Surface distance / surface dice metrics implementation
- `abdomen_DSC_Eval.py`: Multi-organ abdomen DSC evaluation for `.nii.gz`
- `abdomen_NSD_Eval.py`: Multi-organ abdomen NSD evaluation for `.nii.gz` (per-organ tolerance)
- `endoscopy_DSC_Eval.py`: Endoscopy `.png` multi-class DSC evaluation
- `endoscopy_NSD_Eval.py`: Endoscopy `.png` NSD evaluation
- `compute_cell_metric.py`: Cell instance segmentation evaluation (Hungarian matching; TP/FP/FN, Precision/Recall/F1)

These scripts are typically meant to be executed as standalone scripts (they parse CLI args and read files at import time), so run them from the command line instead of importing them as modules.

## Minimal Command Templates

### Training
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainer
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc
```

### Inference
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 2d -tr nnUNetTrainer
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 2d -tr nnUNetTrainerUMambaBot
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 2d -tr nnUNetTrainerUMambaEnc
```

### Evaluation (nnUNetv2 built-in)
```bash
nnUNetv2_evaluate_folder GT_FOLDER PRED_FOLDER -djfile DATASET_JSON -pfile PLANS_JSON
```
