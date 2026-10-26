# AIMED: ECG image classification

**1st place in the classification task of the George B. Moody PhysioNet Challenge 2024.**

[Paper](https://doi.org/10.22489/CinC.2024.398) | [Challenge](https://moody-challenge.physionet.org/2024/) | [Model weights](https://huggingface.co/fmenegui/cinc_of5)

New AIMED-owned code is offered under [PolyForm Noncommercial 1.0.0](LICENSE).
AIMED's covered checkpoint rights and the documentation are offered under
[CC BY-NC 4.0](LICENSES/CC-BY-NC-4.0.txt). Commercial use outside the licenses' expressly
permitted purposes requires separate authorization. The
[licensing terms](LICENSING.md) define the scope and preserve valid earlier
permissions.

This repository makes the AIMED method easier to use and train. A ConvNeXt-Tiny
classifies an ECG page directly into 11 possible findings. The five fold models
are combined by averaging their sigmoid probabilities. No signal digitization
is required.

The code is adapted from `fmenegui/cinc_of5`, commit
`00da6ef97f52f475da86895c286538d3fa8f7c4b`. This is a maintained public release,
with documented changes to the submitted implementation. The
[inference tutorial](docs/inference.md) provides single-image, Python and
Challenge-format examples using public or artificial images. Private InCor
patient data are not needed to run the classifier.

| Published evaluation | Macro F1 |
| --- | ---: |
| Challenge hidden test, classification | 0.817 |
| Internal five-fold cross-validation, mean ± SD | 0.698 ± 0.022 |

The original paper reports these results on different evaluation sets. The
0.817 score belongs to the 2024 entry. This maintained release has not been
evaluated on that hidden set. The archived source did not pin weight revisions,
so exact identity between the downloadable files and the officially scored files
is unverified. The [reproducibility record](docs/reproducibility.md) explains
the artifact provenance and implementation changes.

## Classes

Each model produces **11 independent scores**, in the following fixed order.
This is multilabel classification: an ECG can receive several labels or none.

| Output index | Exact label | Meaning |
| ---: | --- | --- |
| 0 | `NORM` | Normal ECG |
| 1 | `Acute MI` | Acute myocardial infarction |
| 2 | `Old MI` | Old myocardial infarction |
| 3 | `STTC` | ST-segment and T-wave changes |
| 4 | `CD` | Conduction disturbances |
| 5 | `HYP` | Hypertrophy |
| 6 | `PAC` | Premature atrial complex |
| 7 | `PVC` | Premature ventricular complex |
| 8 | `AFIB/AFL` | Atrial fibrillation or atrial flutter |
| 9 | `TACHY` | Tachycardia |
| 10 | `BRADY` | Bradycardia |

The five fold models all predict these same 11 classes. Their probabilities are
averaged, and labels are selected when their score is strictly greater than 0.5.
See [classes and model weights](docs/models.md) for the exact training-label
rules, including the Acute MI / Old MI distinction and grouped rhythm labels.

## Model weights

Model weights are learned neural-network parameters saved in a `.ckpt` file.
The released ensemble provides inference after the final Challenge fine-tuning.
The ECG-pretrained checkpoint supplies the starting parameters for a new
training run.

| Artifact | Hugging Face repository and pinned revision | Files |
| --- | --- | --- |
| Final five-model ensemble | [`fmenegui/cinc_of5`](https://huggingface.co/fmenegui/cinc_of5), `1ab856738289e4c0b76c4df323ef33cfdc45f1f2` | `model_fold0.ckpt` through `model_fold4.ckpt`, each with 11 outputs |
| ECG-pretrained initialization | [`fmenegui/pretrained_cinc_of5`](https://huggingface.co/fmenegui/pretrained_cinc_of5), `831f9bdb728a502f214831a26d0457738645ffc3` | `model_fold0.ckpt`, with 13 original outputs |

[`weights_manifest.json`](weights_manifest.json) records every filename, size
and SHA-256 together with the inference preprocessing, class order and
threshold. The downloader uses these pinned revisions and verifies the files.

Inference loads the five ensemble files into `model/`. Training with
`--init ecg` loads `pretrained_model/model_fold0.ckpt`, retains its learned
feature extractor and replaces the 13-output head with a new 11-output head
before fine-tuning each fold. Keep the two weight folders separate.

The original 13 class names and output order have not been recovered from the
available files. The paper lists 12 InCor-EMG categories, which do not establish
a mapping for those 13 outputs. The [model guide](docs/models.md#ecg-pretraining-checkpoint)
documents the reported categories and missing metadata.

Use this maintained code and its manifest together with the pinned artifacts.
The older `fmenegui/cinc_of2` configuration uses different training settings and
transforms, as described in the [source comparison](docs/reproducibility.md#legacy-code).

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
for initial training checks. The automated workflow covers clean virtual
environments on Linux and Windows.

## 2. Classify an image

```bash
python download_weights.py --kind ensemble
python predict_image.py -i /path/to/ecg.png -m model -o outputs/prediction.json
```

The download is about 557 MB. Five checkpoint files are downloaded from a pinned
Hugging Face revision and verified with SHA-256. Git LFS and a Hugging Face token
are not required. Inference does not need the pretraining checkpoint.

Loading and preprocessing are implemented in [`aimed/model.py`](aimed/model.py)
and described in the [inference tutorial](docs/inference.md). The JSON includes
one probability per class in the order above, the selected labels, threshold
`0.5`, and the checkpoint paths. Findings are not mutually exclusive. These
scores have not been calibrated as clinical probabilities. Preserve the code
commit and weight manifest with your evaluation results.

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
| [Classes and model weights](docs/models.md) | All 11 final outputs, pretraining stages, checkpoint roles and the missing 13-class mapping |
| [Data](docs/data.md) | Required files, labels, PTB-XL conversion, image generation |
| [Training](docs/training.md) | Five-fold training, CLI options, checkpoints, validation |
| [Inference](docs/inference.md) | One image, Python API, Challenge runner, preprocessing |
| [Reproducibility](docs/reproducibility.md) | Original method, pinned artifacts, release changes and limits |
| [Reference repositories](docs/references.md) | Other teams and documentation patterns used here |
| [Validation](docs/validation.md) | Checks actually executed for this release |
| [License and attribution](LICENSING.md) | Code, third-party notices, weights and data |

The private InCor training datasets are not distributed here. Full regeneration
of the original multi-stage pretraining is therefore unavailable from public
data alone. Fine-tuning from the released checkpoint and inference are provided.

## License

New AIMED-owned software contributions are offered under
[PolyForm Noncommercial 1.0.0](LICENSE), with the required notices in
[NOTICE](NOTICE). Documentation and AIMED's copyright interests in the six
identified checkpoints are offered under
[CC BY-NC 4.0](LICENSES/CC-BY-NC-4.0.txt). Noncommercial evaluation is permitted
within the respective grants, subject to their conditions.

The checkpoint CC BY-NC offer does not authorize evaluations primarily directed
toward commercial product development, commercial inference services or
commercial product integration. Obtain separate authorization for those uses
unless an applicable existing permission already covers them. A local,
nonclinical test using public or synthetic images can still have a commercial
purpose. The [licensing guide](LICENSING.md#local-evaluation-and-product-development)
explains the separate code and checkpoint terms.

Earlier BSD, MIT and other valid permissions remain in force. These new offers
do not withdraw those rights or establish the terms of earlier distributions.
Read [LICENSING.md](LICENSING.md) and [MODEL_LICENSE.md](MODEL_LICENSE.md) for
scope, attribution and commercial authorization.

## Cite

Felipe M Dias, Estela Ribeiro, Quenaz B Soares, Jose E Krieger, Marco A Gutierrez.
**Image-Based Electrocardiogram Classification Using Pre-trained ConvNext.**
Computing in Cardiology, 2024, volume 51.
[DOI: 10.22489/CinC.2024.398](https://doi.org/10.22489/CinC.2024.398).

Citation metadata is available in [CITATION.cff](CITATION.cff). Please also cite
the Challenge, the datasets you use and ECG-Image-Kit when generating images.
