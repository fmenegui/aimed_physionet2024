# Training tutorial

## Before starting

Complete the [installation](../README.md#1-install) and [data preparation](data.md).
Run all commands from the repository root. Use a new output folder for each run.
The CLI refuses to overwrite an existing `run_config.json`.

Choose the starting weights:

| Option | Starting point | Download needed | Relation to the original method |
| --- | --- | --- | --- |
| `--init ecg` | Released 13-class ECG-pretrained checkpoint | `download_weights.py --kind pretrained` | Starting point for final Challenge fine-tuning |
| `--init imagenet` | timm ImageNet pretrained ConvNeXt-Tiny | Automatic timm download | Public-data alternative, skips ECG pretraining |
| `--init none` | Random weights | None | Software checks or a separate experiment |

The default is `ecg`. The two other settings are not equivalent to the winning
method. Private pretraining datasets are not included.

## 1. Check the pipeline

```bash
python examples/make_demo_data.py --output data/demo --records 20
python train.py -d data/demo -m runs/software-check --init none --fold 0 --epochs 1 --batch-size 2 --size 64 --num-workers 0 --device cpu --fast-dev-run --quiet
```

This runs one training batch and one validation batch using the actual ConvNeXt
architecture. The artificial labels deliberately have no diagnostic meaning.
Do not report the resulting F1 as model performance.

## 2. Fine-tune one fold on real training data

```bash
python download_weights.py --kind pretrained
python train.py -d data/training -m runs/ptbxl-fold0 --fold 0
```

`--fold 0` selects the first fold, not the first training example. Lower
`--batch-size` if GPU memory is insufficient. The paper reports an RTX 3080 with
10 GB VRAM. GPU requirements depend on image size, batch size and precision.
CPU is supported for checks and inference but is slower for training.

## 3. Train the five-fold ensemble

```bash
python train.py -d data/training -m runs/ptbxl-fivefold
```

All folds train sequentially on the same selected device. Every fold starts from
the same pretrained checkpoint with a newly initialized 11-class head. The
original unused `classifier` layer is retained for checkpoint compatibility.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--epochs` | 30 | Maximum epochs per fold |
| `--batch-size` | 16 | Images per batch |
| `--lr` | 0.0001 | AdamW learning rate |
| `--lr-find` | Off | Explicitly run and apply Lightning's LR suggestion |
| `--patience` | 5 | Early stopping patience on validation loss |
| `--folds` | 5 | Number of grouped folds |
| `--fold` | -1 | Train all folds, or select a zero-based index |
| `--size` | 384 | Square training image size |
| `--num-workers` | 2 | Data loader workers. Zero runs in the main process |
| `--device` | auto | CUDA when available, otherwise CPU |
| `--seed` | 42 | Software random seed |
| `--pretrained-checkpoint` | `pretrained_model/model_fold0.ckpt` | ECG starting checkpoint |

Inference with the published weights always uses 384 × 384 pixels. A different
training size is a separate experiment. Mixed precision is used on GPU and
32-bit precision on CPU. AdamW retains PyTorch's default weight decay of 0.01.
The class weights match the submitted configuration and multiply BCE terms for
both positive and negative examples. They are not `pos_weight` values.

For an LR-finder experiment:

```bash
python train.py -d data/training -m runs/ptbxl-lrfind --lr-find
```

Do this on a sufficiently large dataset. The original configuration enabled the
finder. This release makes it opt-in and records the applied learning rate per fold.
The finder needs enough optimizer steps to suggest a rate. A one-epoch check on
a tiny dataset may collect too few points and retain `--lr` instead. The audit
also exercises a full 100-step search and verifies that its suggested rate is used.

## 4. Inspect the outputs

| Path inside the run folder | Contents |
| --- | --- |
| `run_config.json` | CLI settings, class order, threshold and key dependency versions |
| `df.csv` | Image paths, labels and train/validation membership for each fold |
| `model/model_fold0.ckpt` through `model_fold4.ckpt` | Final saved model for each trained fold |
| `predictions/validation_fold0.csv` and analogous files | Validation probabilities and reference labels |
| `metrics.json` | Per-class and macro F1 at threshold 0.5 for each fold |
| `logs/fold*/version_*/metrics.csv` | Epoch losses and macro F1 |
| `output.log` | Training console output |
| `resized_images/` | Run-specific image cache |
| `timers.json` | Elapsed training time |

As in the inspected source configuration, the saved checkpoint is from the last
epoch reached by early stopping. The source uses `best_weights=False`. This
differs from the paper's description of selecting the best model in each fold.
See the [discrepancy record](reproducibility.md).

## 5. Use the new weights

```bash
python predict_image.py -i /path/to/new_ecg.png -m runs/ptbxl-fivefold -o outputs/new_prediction.json
```

Both `run_folder/model/` and a folder containing `model_fold*.ckpt` directly are
accepted. Every discovered model receives equal weight. A folder containing
only one fold produces a single-model prediction.

For the standard Challenge interface:

```bash
python train_model.py -d data/training -m runs/challenge-style -v
python run_model.py -d data/heldout_hidden -m runs/ptbxl-fivefold -o outputs/heldout -v
python evaluate_model.py -d data/heldout_reference -o outputs/heldout -s outputs/scores.txt
```

The standard training wrapper uses the defaults. Use `train.py` to change them.
On systems that cannot create multiprocessing workers, set
`AIMED_NUM_WORKERS=0` before either training command. Explicit `--num-workers`
still takes precedence in `train.py`.

```bash
AIMED_NUM_WORKERS=0 python train_model.py -d data/training -m runs/challenge-no-workers -v
```

## Evaluate without leakage

The inherited split groups all images of one ECG record together. This prevents
different renderings of that record from entering different folds. It does not
group different records from the same patient. Use PTB-XL patient identifiers
and its recommended partitions when building a separate patient-level benchmark.

Reserve held-out patients before calling the training CLI. Do not evaluate the
five-model ensemble on the same data that trained its folds and present that as
an independent result. Use each fold's own validation predictions for internal
cross-validation. A five-fold mean and SD are not a confidence interval.

The official hidden test is not included. Training this code does not guarantee
reproduction of the published 0.817 score.
