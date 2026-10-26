# Executed validation

Release preparation date: 2026-10-26.

Local environment: Linux, Python 3.12.14, CPU, PyTorch 2.2.2+cpu,
torchvision 0.17.2+cpu, timm 0.9.12, Lightning 2.1.3, NumPy 1.26.4.
Dependencies are installed from the pinned requirement files.

## Checks performed

| Check | Result |
| --- | --- |
| Software test suite | 8 tests passed |
| Final weight downloads | All five files downloaded and verified against pinned SHA-256 values |
| Original checkpoint compatibility | Strict loading succeeded with all expected model keys |
| ECG-pretraining checkpoint compatibility | 13-class checkpoint loaded strictly and classifier reset to 11 classes |
| Published ensemble inference | All five models ran on an artificial demo image and produced 11 finite scores |
| Challenge-format inference | Completed on 20 artificial records with hidden labels and no digitization output |
| Evaluator interface | Completed on those software-check outputs |
| Training pipeline | ConvNeXt training step, validation, checkpoint save and reload passed |
| Dependency consistency | `pip check` passed after installation of the pinned dependencies |

The test suite covers deterministic grayscale preprocessing, averaging
probabilities, strict thresholding, missing/duplicate/incompatible checkpoints,
record grouping across image variants, zero data loader workers, correct
multilabel macro F1, and train/save/reload/Challenge-interface integration.

Run the software checks with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The GitHub Actions workflow runs these checks on Linux with Python 3.11 and 3.12.
Current hosted run status is available on the repository's Actions tab. Local
results above are independent of the hosted workflow status.

## Limits

The artificial image generator assigns placeholder labels. Its outputs are used
only to verify software execution. No demo F1 is reported as model performance.

This validation does not include a complete 30-epoch five-fold PTB-XL training
run, reconstruction of private InCor pretraining, the original hidden-test
benchmark, or Docker/NVIDIA GPU execution. The external ECG-Image-Kit data
generation tutorial follows the primary documentation and was not executed
against a full downloaded PTB-XL collection during this release preparation.

The published 0.817 score belongs to the 2024 evaluation, not to these tests.
