# Inference tutorial

## One image

After installation:

```bash
python download_weights.py --kind ensemble
python predict_image.py -i /path/to/ecg.png -m model -o outputs/prediction.json --device cpu
```

Omit `--device cpu` to use an available NVIDIA GPU automatically. Image inference
does not need a `.hea` header or an ECG-pretraining checkpoint. The JSON contains
the ordered class names, probabilities, threshold, labels and checkpoint paths.

## Python

Run from the repository root, with the environment activated:

```python
from PIL import Image
from aimed.model import CLASSES, load_ensemble, predict_probabilities, predicted_labels

models = load_ensemble("model", device="auto")
with Image.open("/path/to/ecg.png") as image:
    probabilities = predict_probabilities(models, image)

print(dict(zip(CLASSES, probabilities.tolist())))
print(predicted_labels(probabilities))
```

Load the ensemble once and reuse it across images. Checkpoint loading is strict.
Missing or incompatible weights raise an error instead of silently using random
parameters. Checkpoints use the legacy PyTorch format. Load files from a trusted
source. The bundled downloader verifies the published artifacts.

## Preprocessing and aggregation

1. Convert to RGB and resize the entire page to 384 × 384 with Pillow LANCZOS.
2. Convert to grayscale, represented as three channels.
3. Normalize by ImageNet mean `[0.485, 0.456, 0.406]` and SD `[0.229, 0.224, 0.225]`.
4. Compute 11 logits per model and apply a sigmoid to each logit.
5. Average probabilities across models and select classes with a score strictly
   greater than 0.5.

Resizing stretches the page to a square, as in the submission. There is no
rectification, lead detection, signal extraction or clinical calibration step.
This release uses deterministic grayscale preprocessing. The inspected source
left `A.ToGray()` at its default probability, making preprocessing stochastic.
This change prevents claiming bitwise reproduction of the original submission.

## Challenge-format folders

Use a folder of `.hea` files and the images named in their `# Image:` comments:

```bash
python run_model.py -d data/heldout_hidden -m model -o outputs/predictions -v
```

No reference labels are required for inference. When multiple images are listed,
the first listed image is used, matching the effective behavior of `cinc_of5`.
This release returns `None` for digitization and writes classification labels
only. The original scaffold's random waveform generator is not used.

To create a copy that hides labels and waveforms but keeps images:

```bash
python remove_hidden_data.py -i data/heldout_reference -o data/heldout_hidden -m
python run_model.py -d data/heldout_hidden -m model -o outputs/predictions -v
python evaluate_model.py -d data/heldout_reference -o outputs/predictions -s outputs/scores.txt
```

Keep input and output paths distinct. The evaluator is retained from the source
snapshot. Its output reports SNR as unavailable for classification-only outputs.
The score file is plain text, even if a `.csv` extension is chosen.

## Troubleshooting

| Error or symptom | Action |
| --- | --- |
| No classification checkpoints found | Run `download_weights.py --kind ensemble` or point `-m` to a trained run |
| State dictionary size mismatch | Check that inference uses final 11-class weights, not the 13-class pretraining checkpoint |
| Missing ECG pretraining checkpoint | Needed for `--init ecg` training only. Run `download_weights.py --kind pretrained` |
| NumPy ABI error | Install the pinned NumPy 1.26.4 and OpenCV versions |
| `pkg_resources` missing | Install `requirements.txt`, including pinned setuptools |
| CPU is slow | Try `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` and reuse the loaded models |
| Several labels, or no labels | This is multilabel classification with independent thresholds |

## Docker

```bash
docker build -t aimed-physionet2024 .
docker run --rm -v "$PWD/model:/challenge/model" -v "$PWD/data:/challenge/data:ro" -v "$PWD/outputs:/challenge/outputs" aimed-physionet2024 python predict_image.py -i data/ecg.png -m model -o outputs/prediction.json
```

Download weights on the host first. Add `--gpus all` to `docker run` on an NVIDIA
Container Toolkit host. The image build does not download weights. Docker/GPU
validation status is listed separately in [validation](validation.md).
