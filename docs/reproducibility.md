# Method, provenance and reproducibility

## Source

The code was inspected from `fmenegui/cinc_of5` at commit
`00da6ef97f52f475da86895c286538d3fa8f7c4b`. This release imports the relevant
training package and Challenge scripts, and adds portable interfaces and guides.
Server-copy scripts, obsolete experiment entry points, unrelated models,
cached bytecode and private datasets were not imported.

The paper is [Dias et al., CinC 2024](https://doi.org/10.22489/CinC.2024.398).
The public release is derived from the source snapshot. It is not represented
as an unmodified archive of the official evaluated entry.

If you found the older `fmenegui/cinc_of2` code, see the
[legacy-source comparison](evaluation-faq.md#legacy-code).
That inspected configuration and Dockerfile differ from the source and pinned
artifact combination used by this tutorial.

## Published pipeline

1. Start from an ImageNet-pretrained ConvNeXt.
2. Adapt to synthetic CODE15 images and real InCor-AMB images.
3. Fine-tune on the InCor-EMG emergency image dataset.
4. Fine-tune on PTB-XL with five folds and combine the fold models.

The provided ECG-pretraining checkpoint is the starting point for step 4.
The final checkpoints provide inference after that step. Earlier data and
training runs cannot be regenerated from the files in this repository alone.

## Pinned weights

| Artifact | Hugging Face repository | Revision | Contents |
| --- | --- | --- | --- |
| Final ensemble | `fmenegui/cinc_of5` | `1ab856738289e4c0b76c4df323ef33cfdc45f1f2` | Five 11-class checkpoints |
| ECG pretraining | `fmenegui/pretrained_cinc_of5` | `831f9bdb728a502f214831a26d0457738645ffc3` | One 13-output checkpoint, original class names and order unavailable |

Every checkpoint's exact filename, byte size and SHA-256 are in
[`weights_manifest.json`](../weights_manifest.json). The hashes come from the
Hub's LFS metadata at those revisions and are checked after download.

Both artifacts have a file named `model_fold0.ckpt` but serve different purposes.
Do not interchange them. The final ensemble is about 557 MB and the pretraining
checkpoint about 111 MB. These files are downloaded separately from GitHub.

The [model guide](models.md) documents all 11 final class names and labeling
rules. It also distinguishes the six initial ECG-adaptation categories, the
12 InCor-EMG categories named in the paper, and the 13 outputs observed in the
released pretraining checkpoint. The original 13-output class vocabulary and
order are absent from the available artifacts and have not been inferred from
the paper. Final fine-tuning replaces that head with an 11-output head.

The original Dockerfile downloaded these same repository names, without fixing
revisions. The inspected source does not record which Hub commit was present
during official scoring. These are pinned public artifacts available at release
preparation time. Their exact identity with the bytes scored in 2024 is not
independently established by the available source.

## Source and paper differences

| Item | Inspected source snapshot | Paper | Public-release choice |
| --- | --- | --- | --- |
| Early stopping patience | 5 | 7 | Default 5, configurable with `--patience` |
| Checkpoint selection | `best_weights=False`, saves final reached epoch | Describes best models from each fold | Preserve final reached epoch and document the difference |
| Learning rate | `lr=1e-4`, `find_lr=True` | LR-finder strategy | Fixed `1e-4` by default, explicit `--lr-find` applies a suggestion |
| Grayscale | `A.ToGray()` with library default probability | Describes grayscale images | Preserve default training probability, use deterministic grayscale for validation/inference |

No unsupported reconciliation of these differences is asserted. If an exact
historical experiment configuration is recovered, it should be versioned
separately and linked to the corresponding checkpoint hashes.

## Packaging and correctness changes

- Removed configuration import side effects and machine-specific paths.
- Split model construction for inference from ECG-pretraining initialization.
- Load checkpoint tensors onto CPU first and enforce strict model key matching.
- Keep the original extra `classifier` layer for binary checkpoint compatibility.
- Put inference models in evaluation mode and disable gradients.
- Make evaluation grayscale deterministic instead of applying a random transform.
- Preserve the effective first-image policy for multi-image records.
- Return no digitization output instead of random waveforms from the template.
- Put resized images and logs inside each run folder.
- Respect zero data loader workers and stop writing `dataframe.csv` on dataset initialization.
- Group records by full relative path, avoiding collisions between nested folders.
- Compute logged macro F1 on the multilabel matrix. The old callback flattened
  all class columns before computing its logged score.
- Use CSV epoch logs and export validation predictions, metrics and run settings.
- Pin compatible dependencies, including NumPy, OpenCV and setuptools.
- Add software tests and an artificial example dataset generator.
- Normalize whitespace in the imported source files.
- Pin Hugging Face Hub to preserve the timm ImageNet-loading API.
- Parse 12SL labels as exact acronyms and support missing annotations.
- Preserve existing header comments on a separate line during label preparation.
- Reject image-registration inputs whose headers have no corresponding images.
- Allow the standard training wrapper to disable multiprocessing through
  `AIMED_NUM_WORKERS=0` on restricted hosts.
- Provide a tested generator environment and bounded public-data fixture helper.
- Skip pointless all-NaN waveform alignment when evaluating classification-only
  output against reference records that include real waveforms.
- Use native relative paths for the resized-image cache on Windows and POSIX.

These changes improve usability and address observed implementation issues.
They do not establish a new Challenge score.

## What can be reproduced

| Goal | Supported by this release? |
| --- | --- |
| Download and integrity-check a fixed set of public weights | Yes |
| Run their classifier on an image or a Challenge-format folder | Yes |
| Fine-tune on public PTB-XL images using the released ECG checkpoint | Yes |
| Use ImageNet initialization or random initialization instead | Yes, separate experimental settings |
| Rebuild the entire historical pretraining from public data alone | No, private InCor data and earlier run configurations are absent |
| Recreate the exact 2024 generated image set | Not established, generator revision and historical outputs are absent |
| Independently reproduce the official hidden-test score | No, the hidden evaluation data are not included |

The [validation report](validation.md) separates executable checks from
unperformed training and benchmark evaluations.
