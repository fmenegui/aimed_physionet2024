# Prepare training and evaluation data

## Required format

Training reads `.hea` headers recursively. Each header lists its image filenames
and the Challenge labels. Images must be relative to the header's directory.

For example, a folder may contain `00001.hea`, `00001-0.png` and `00001-1.png`.
The header's comments include:

```text
# Labels: NORM
# Image: 00001-0.png, 00001-1.png
```

Preserve the WFDB record and signal description lines before these comments.
Classification does not read signal samples. `.dat` files are useful for image
generation but are not required to train on existing images. The loader also
accepts common `#Dx:` variants. Prefer the official `# Labels:` field.

| Output index | Label |
| ---: | --- |
| 0 | NORM |
| 1 | Acute MI |
| 2 | Old MI |
| 3 | STTC |
| 4 | CD |
| 5 | HYP |
| 6 | PAC |
| 7 | PVC |
| 8 | AFIB/AFL |
| 9 | TACHY |
| 10 | BRADY |

Keep spelling and capitalization unchanged. Several labels can be comma-separated
in one header. Labels outside this list are not training outputs.

See [classes and model weights](models.md#final-challenge-classes) for every
abbreviation, the exact PTB-XL/12SL labeling rules and how these 11 final outputs
differ from the earlier pretraining tasks.

## PTB-XL route

The preparation follows the organizers' [Python example](https://github.com/physionetchallenges/python-example-2024)
and [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit).
The original paper reports default image-generator parameters for CODE15 and
PTB-XL. The example below starts with one subfolder to keep the first run small.

### 1. Download the source data

Download and extract:

- [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL+ 1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/)

In the following commands, the extracted folders have been placed at
`data/ptb-xl` and `data/ptb-xl-plus`. PTB-XL contains `records500/`,
`ptbxl_database.csv` and `scp_statements.csv`. PTB-XL+ contains
`labels/12sl_statements.csv` and `labels/mapping/12slv23ToSNOMED.csv`.

These downloads are separate from the model weights. Check available storage
before generating large image collections. Keep source data and outputs outside
version control.

For bounded software validation, the following command fetches 20 public records
and 100 annotation rows instead of the full datasets:

```bash
python scripts/fetch_tutorial_fixture.py --output data/tutorial
```

Its manifest records separate patient groups and input
hashes. Replace the data path prefixes in the following commands with
`data/tutorial` when using that fixture. It is deliberately not a clinical
benchmark.

### 2. Prepare labels and headers

```bash
python prepare_ptbxl_data.py \
  -i data/ptb-xl/records500/00000 \
  -pd data/ptb-xl/ptbxl_database.csv \
  -pm data/ptb-xl/scp_statements.csv \
  -sd data/ptb-xl-plus/labels/12sl_statements.csv \
  -sm data/ptb-xl-plus/labels/mapping/12slv23ToSNOMED.csv \
  -o data/prepared/00000
```

The script combines the PTB-XL and 12SL labels, including the Challenge's Acute
MI / Old MI distinction. Do not replace it with a generic five-class PTB-XL
mapping.

### 3. Generate ECG images

Use a separate environment. The audited Linux/Python 3.12 dependency set is
provided here because the upstream generator requirements include older
TensorFlow and spaCy versions. The following commands pin the tested generator
revision:

```bash
python -m venv .venv-generator
.venv-generator/bin/python -m pip install -r requirements-generator.txt
git clone --filter=blob:none --sparse https://github.com/alphanumericslab/ecg-image-kit.git third_party/ecg-image-kit
git -C third_party/ecg-image-kit sparse-checkout set codes/ecg-image-generator
git -C third_party/ecg-image-kit checkout 27b90f56896c9fc78b05a83ca14844ea2637aa0b
```

From this repository's root, run:

```bash
.venv-generator/bin/python third_party/ecg-image-kit/codes/ecg-image-generator/gen_ecg_images_from_data_batch.py \
  -i "$PWD/data/prepared/00000" \
  -o "$PWD/data/images/00000" \
  --print_header
```

Keep the generated headers and their images together. This revision is pinned
for the present tutorial audit. Its identity with the generator used during
the 2024 experiment is not established. Full handwriting synthesis and its
external pretrained assets are outside this default generation recipe.

### 4. Register the images in the headers

Return to this repository and reactivate the classifier environment:

```bash
source .venv/bin/activate
python prepare_image_data.py -i data/images/00000 -o data/training/00000
```

Expected image names are `record-0.png`, `record-1.png`, etc. The preparation script
lists them in `# Image:`. Check at least one header and confirm that each image
exists before training.

Repeat for the other PTB-XL subfolders to prepare a larger dataset. A single
subfolder is only a first run, not the complete public training set.

### 5. Train

```bash
python download_weights.py --kind pretrained
python train.py -d data/training -m runs/ptbxl-first-run --fold 0
```

See [training](training.md) for all folds, outputs and evaluation design.

## Your existing ECG images

For a single image, use `predict_image.py` without any header. For training, create
valid headers containing the image filename and the appropriate reference labels.
Use the generated demo as a file-format example:

```bash
python examples/make_demo_data.py --output data/demo --records 20
```

Its labels are arbitrary placeholders and must never be used as diagnostic
ground truth. The training loader groups images by their full relative record
path. It does not infer patient identity from image filenames.

## Data boundaries

The original method used CODE15, InCor-AMB and InCor-EMG before the final PTB-XL
stage. The private InCor datasets are not included or fetched. This repository
provides the resulting pretraining checkpoint, not a substitute dataset.

For a separate benchmark, reserve held-out patients before generating multiple
images per record. Follow each dataset's access and licensing terms. Keep
patient identifiers and private data out of commits and examples.
