# AIMED: ECG image classification

**1st place in the classification task of the George B. Moody PhysioNet Challenge 2024.**

[Paper](https://doi.org/10.22489/CinC.2024.398) | [Challenge](https://moody-challenge.physionet.org/2024/) | [Model weights](https://huggingface.co/fmenegui/cinc_of5)

This repository makes the AIMED method easier to use and train. A ConvNeXt-Tiny
classifies an ECG page directly into 11 possible findings. The five fold models
are combined by averaging their sigmoid probabilities. No signal digitization
is required.

The code is adapted from `fmenegui/cinc_of5`, commit
`00da6ef97f52f475da86895c286538d3fa8f7c4b`. This is a maintained public release,
with documented changes to the submitted implementation. The score below is a
historical Challenge result, not a new evaluation of this release.

| Published evaluation | Macro F1 |
| --- | ---: |
| Challenge hidden test, classification | 0.817 |
| Internal five-fold cross-validation, mean ± SD | 0.698 ± 0.022 |

The original paper reports these results on different evaluation sets. They are
not interchangeable. See [reproducibility](docs/reproducibility.md).

## 1. Install

Use Python 3.10 to 3.12. Run commands from this repository's root.

```bash
git clone https://github.com/fmenegui/aimed_physionet2024.git
cd aimed_physionet2024
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **one** PyTorch build:

```bash
# CPU
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

```bash
# NVIDIA GPU, CUDA 11.8 wheels
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e media
```

On Windows, activate with `.venv\Scripts\activate` and use `--num-workers 0`
for initial training checks. The automated workflow targets Linux.

## 2. Classify an image

```bash
python download_weights.py --kind ensemble
python predict_image.py -i /path/to/ecg.png -m model -o outputs/prediction.json
```

The download is about 557 MB. Five checkpoint files are downloaded from a pinned
Hugging Face revision and verified with SHA-256. Git LFS and a Hugging Face token
are not required. Inference does not need the pretraining checkpoint.

The JSON includes one probability per class, the selected labels, threshold
`0.5`, and the checkpoint paths. Findings are not mutually exclusive. These
scores have not been calibrated as clinical probabilities.

For an installation check without patient data:

```bash
python examples/make_demo_data.py --output data/demo --records 20
python predict_image.py -i data/demo/demo000-0.png -m model
```

The demo draws artificial traces and assigns arbitrary labels. It checks the
software pipeline only. See [inference](docs/inference.md) for Python and
Challenge-format examples.

## 3. Train

Prepare a folder of ECG images and WFDB `.hea` headers listing `# Image:` and
`# Labels:`. See the complete [data preparation tutorial](docs/data.md), including
PTB-XL, PTB-XL+ and synthetic image generation.

```bash
python download_weights.py --kind pretrained
python train.py -d data/training -m runs/experiment01
```

This fine-tunes five models from the released ECG-pretrained checkpoint. The
default configuration uses 384 × 384 images, weighted BCE, AdamW, batch size 16,
up to 30 epochs, and early stopping after five epochs without sufficient
validation-loss improvement.

Start with one fold if needed:

```bash
python train.py -d data/training -m runs/fold0 --fold 0 --batch-size 4
```

Check training on the artificial demo without downloading pretrained weights:

```bash
python train.py -d data/demo -m runs/demo-check --init none --fold 0 --epochs 1 --batch-size 2 --size 64 --num-workers 0 --device cpu --fast-dev-run --quiet
```

This is a one-batch software check. Its weights and metrics have no diagnostic
meaning. See [training](docs/training.md) for initialization options, outputs,
evaluation and hardware notes.

## Documentation

| Guide | Contents |
| --- | --- |
| [Data](docs/data.md) | Required files, labels, PTB-XL conversion, image generation |
| [Training](docs/training.md) | Five-fold training, CLI options, checkpoints, validation |
| [Inference](docs/inference.md) | One image, Python API, Challenge runner, preprocessing |
| [Reproducibility](docs/reproducibility.md) | Original method, pinned artifacts, release changes and limits |
| [Reference repositories](docs/references.md) | Other teams and documentation patterns used here |
| [Validation](docs/validation.md) | Checks actually executed for this release |
| [Licensing status](LICENSING.md) | Code, third-party notices, weights and data |

The private InCor training datasets are not distributed here. Full regeneration
of the original multi-stage pretraining is therefore unavailable from public
data alone. Fine-tuning from the released checkpoint and inference are provided.

## License

New AIMED-owned code is available under [PolyForm Noncommercial 1.0.0](LICENSE)
with the attribution notices in [NOTICE](NOTICE). Documentation and AIMED's rights
in the identified checkpoint artifacts are offered under CC BY-NC 4.0.
Commercial use requires separate authorization unless an existing permission
or an expressly permitted purpose applies.

Earlier BSD or other permissions are preserved. The historical 2024 submission
is not represented as becoming commercial-use prohibited. Read
[LICENSING.md](LICENSING.md) and [MODEL_LICENSE.md](MODEL_LICENSE.md) for the scope.

## Cite

Felipe M Dias, Estela Ribeiro, Quenaz B Soares, Jose E Krieger, Marco A Gutierrez.
**Image-Based Electrocardiogram Classification Using Pre-trained ConvNext.**
Computing in Cardiology, 2024, volume 51.
[DOI: 10.22489/CinC.2024.398](https://doi.org/10.22489/CinC.2024.398).

Citation metadata is available in [CITATION.cff](CITATION.cff). Please also cite
the Challenge, the datasets you use and ECG-Image-Kit when generating images.
