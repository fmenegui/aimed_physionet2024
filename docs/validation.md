# Executed tutorial validation

Audit date: 2026-10-26. Tested application commit:
`9ab1fb4bfb34f087cbe920fd4f9dbaca15850e01`.

This audit executes the documented software paths on bounded data. It does not
claim a new clinical score or reproduce the official hidden-test evaluation.

## Evidence

- [Clean-install and regression test workflow](https://github.com/fmenegui/aimed_physionet2024/actions/runs/40382740649)
- [Docker, CUDA-wheel and real-data generator workflow](https://github.com/fmenegui/aimed_physionet2024/actions/runs/40382740437)
- [Machine-readable audit results](../reports/tutorial_validation_2026-10-26.json)
- [Reference training configuration, epochs and checkpoint hashes](../reports/reference_training_2026-10-26.json)

Both hosted workflows completed successfully. Their downloadable artifacts
include inference outputs, generator manifests, record headers and run settings.

## Results

| Tutorial path | Executed check | Result |
| --- | --- | --- |
| Public clone | The documented public Git clone command on a hosted runner | Passed |
| CPU installation | New virtual environments and `pip check`, Linux Python 3.10/3.11/3.12 and Windows Python 3.11 | Passed |
| Software tests | 20 tests on each of those four platform/version combinations | Passed |
| CUDA installation | Exact cu118 wheel command, dependency check, CUDA 11.8 build inspection and CPU inference | Passed |
| Published weights | All six files, pinned revisions, sizes, SHA-256 and cached verification | Passed |
| Attribution delivery | Weight downloads include the license, scope and required notices | Passed |
| Single-image inference | Published five-model ensemble and 11 output scores | Passed |
| Python inference API | Repeated predictions are identical and match CLI JSON | Passed |
| PTB-XL label preparation | 20 actual records with PTB-XL 1.0.3 and PTB-XL+ 1.0.1 annotations | Passed |
| Image generation | Pinned ECG-Image-Kit, separate Python 3.12 environment, `--print_header`, 20 records | Passed |
| Header/image registration | Generated image files appear in valid headers | Passed |
| Random initialization | Actual ConvNeXt train/validation step, save and reload | Passed |
| ImageNet initialization | Actual pretrained weights download and train/validation step | Passed |
| ECG-pretrained initialization | Five folds trained for one epoch each at 64 pixels | Passed |
| Batch-size option | One complete epoch with batch size 4 | Passed |
| LR finder | 100 optimizer steps and subsequent use of the suggested learning rate | Passed |
| Standard training wrapper | Five folds, 384 pixels, batch size 16, maximum 30 epochs on a small real-data cohort | Passed |
| New-model use | All five default-recipe checkpoints strictly reloaded and used on held-out images | Passed |
| Challenge-format inference | Hidden-label inputs, 20 artificial records and 10 actual held-out records | Passed |
| Evaluation | Classification-only outputs scored against real reference records containing waveforms | Passed |
| Docker | Image build, dependency check, published inference and checkpoint training/save check | Passed |
| Documentation and metadata | Local links, YAML parsing, CFF 1.2.0 schema validation and Python compilation | Passed |

## Default training recipe

The standard `train_model.py` wrapper ran on 10 real ECG records with distinct
patients. Ten separate patients were reserved for held-out interface checks.
The selected training examples are normal ECGs. This intentionally small,
unrepresentative fixture is for software validation, not diagnostic evaluation.

```bash
AIMED_NUM_WORKERS=0 python train_model.py -d data/training -m runs/audit-default-workers0 -v
python predict_image.py -i data/heldout_reference/00000/00008_hr-0.png -m runs/audit-default-workers0 -o outputs/reference-trained-prediction.json
python run_model.py -d data/heldout_hidden -m runs/audit-default-workers0 -o outputs/reference-trained-heldout -v
python evaluate_model.py -d data/heldout_reference -o outputs/reference-trained-heldout -s outputs/reference-trained-scores.txt
```

The numerical training defaults were retained: five folds, 384 × 384 images,
batch size 16, AdamW at 0.0001, maximum 30 epochs and early stopping patience 5.
Worker processes were disabled because the local execution host restricts
multiprocessing sockets. Hosted Linux/Windows tests independently cover the
portable training path.

The completed epoch counts were **30, 30, 30, 10 and 30**. The fourth fold
stopped early. All five checkpoints were saved and reloaded. Epoch counts were
read from the epoch logs, rather than adding one to Lightning's post-fit
checkpoint counter.

The separate LR-finder test completed 100 search steps and applied
`4.4668359215096314e-05`. A one-epoch tiny-data test also confirmed the documented
fallback to the specified fixed rate when too few search points are available.

## Regressions found and corrected

- Pin Hugging Face Hub 0.23.5 for timm 0.9.12 ImageNet loading.
- Use native relative paths for resized-image caches on Windows.
- Handle missing 12SL annotations and match exact acronyms.
- Preserve existing header comments on their own lines during preparation.
- Reject headers without corresponding generated images.
- Skip all-NaN waveform alignment for classification-only predictions.
- Support `AIMED_NUM_WORKERS=0` in the standard training wrapper.
- Provide a tested, separate dependency set for ECG-Image-Kit on Python 3.12.

The expanded tests also check corrupted downloads, failed integrity checks,
checkpoint compatibility, record grouping, deterministic preprocessing,
multilabel macro F1 and attribution files.

## Explicit limits

The CUDA package installation and CUDA-runtime container were validated, with
inference executed on CPU. **Training or inference on a physical NVIDIA GPU was
not run**, because no such device was available. CUDA mixed-precision execution
is therefore not claimed as tested.

This audit does not train the full PTB-XL cohort, reconstruct private InCor
pretraining, evaluate the official hidden set, or test optional handwriting
synthesis outside the documented default generator command. Its fixture F1
values are not reported as clinical model performance. The historical 0.817
score belongs to the original 2024 evaluation.

Run the repeatable software suite with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Class and checkpoint documentation follow-up

On 2026-10-26, the class tables and [model guide](models.md) were checked against
the actual source and downloaded checkpoint tensors. The 20-test suite passed
again. Additional direct checks confirmed:

- Both final-class tables and manifest metadata match the API's 11-label order.
- All six published checkpoint sizes and SHA-256 values remain unchanged.
- The downloader copies the expanded class and purpose metadata into both weight folders.
- The pretraining head has 13 outputs. Strict loading succeeds before it is replaced.
- Every feature-extractor tensor is retained exactly at initialization, while
  the new head has 11 outputs. Both the backbone and head remain trainable.
- A forward pass after initialization returns 11 logits.
- The edited relative documentation links and section anchors resolve.

The available checkpoint, original configurations and pinned model card contain
no original 13-class name/order mapping. The paper's 12 named InCor-EMG
categories are documented separately, without assigning them to unverified
checkpoint output indices. No new benchmark result is asserted.
