# Classes and model weights

## Final Challenge classes

The released ensemble contains **five ConvNeXt-Tiny models, each with the same
11 outputs**. Five is the number of folds, not the number of classes. The fixed
output order comes from [`aimed/model.py`](../aimed/model.py) and the submitted
configuration. It is also recorded in [`weights_manifest.json`](../weights_manifest.json).

| Output index | Exact label | Meaning | Rule in `prepare_ptbxl_data.py` |
| ---: | --- | --- | --- |
| 0 | `NORM` | Normal ECG | PTB-XL diagnostic superclass `NORM` |
| 1 | `Acute MI` | Acute myocardial infarction | At least one of the selected PTB-XL+ 12SL acute-MI statements |
| 2 | `Old MI` | Old myocardial infarction | PTB-XL diagnostic superclass `MI`, with none of those selected acute-MI statements |
| 3 | `STTC` | ST-segment and T-wave changes | PTB-XL diagnostic superclass `STTC` |
| 4 | `CD` | Conduction disturbances | PTB-XL diagnostic superclass `CD` |
| 5 | `HYP` | Hypertrophy | PTB-XL diagnostic superclass `HYP` |
| 6 | `PAC` | Premature atrial complex | SCP code `PAC` |
| 7 | `PVC` | Premature ventricular complex | SCP code `PVC` |
| 8 | `AFIB/AFL` | Atrial fibrillation or atrial flutter | SCP code `AFIB` or `AFLT` |
| 9 | `TACHY` | Tachycardia | SCP code `STACH`, `SVTAC` or `PSVT` |
| 10 | `BRADY` | Bradycardia | SCP code `SBRAD` |

Keep the exact label strings, capitalization and order when building headers or
interpreting arrays. A header can contain multiple comma-separated labels:

```text
# Labels: CD, AFIB/AFL
```

The selected 12SL statement numbers for `Acute MI` are `821, 822, 823, 827, 829,
902, 903, 904, 963, 964, 965, 966, 967, 968`. The preparation script looks up
their acronyms in the supplied 12SL mapping. `Old MI` is the preparation rule
above, including when acute-MI annotations are unavailable. These are dataset
label definitions, not an independent adjudication of infarct timing.

`CD`, `HYP` and `STTC` are grouped labels. The final model does not return a
separate result for each conduction disturbance or hypertrophy subtype.
Similarly, `AFIB/AFL` combines two rhythms and `TACHY` combines the three SCP
codes above. Do not interpret them as separate output probabilities.

## Reading predictions

Each fold produces 11 logits. Apply a sigmoid to each, then take the mean across
folds for each class. The returned probability vector follows the table above.
The selected labels have scores **strictly greater than 0.5**.

The scores are independent and do not have to sum to one. Several classes can
exceed the threshold, or none can. A score of exactly 0.5 is not selected.
No clinical probability calibration is performed by this release.

The [single-image CLI](inference.md#one-image) writes named probabilities and
selected labels to JSON. The Python API exposes the same class order through
`aimed.model.CLASSES`.

## Checkpoints

A checkpoint is a saved set of learned network parameters. It allows an already
trained model to be used, or its parameters to initialize another training run.
It does not contain the training images.

Both downloads below contain a file called `model_fold0.ckpt`. The directory,
Hugging Face repository and SHA-256 identify which artifact it is. Renaming a
file does not change its output classes or make the two checkpoints compatible.

| Artifact | Final ensemble | ECG-pretrained checkpoint |
| --- | --- | --- |
| Hugging Face repository | [`fmenegui/cinc_of5`](https://huggingface.co/fmenegui/cinc_of5) | [`fmenegui/pretrained_cinc_of5`](https://huggingface.co/fmenegui/pretrained_cinc_of5) |
| Download | `python download_weights.py --kind ensemble` | `python download_weights.py --kind pretrained` |
| Destination | `model/` | `pretrained_model/` |
| Files | Five fold checkpoints | One initialization checkpoint |
| Output head | 11 outputs with the verified order above | 13 outputs, original names and order unavailable |
| Intended use | Inference after final Challenge fine-tuning | Initialize `train.py --init ecg` |
| Training data needed to use it | No | Yes, labeled images for the new training run |

Pinned revisions, sizes and hashes are in the manifest. The inference loader
requires 11-output models and rejects the 13-output checkpoint.

## ECG pretraining checkpoint

### Role in training

The original Challenge configuration loads this checkpoint before final PTB-XL
training. It provides ECG features learned before that final stage. The
[2024 paper, Sections 2.1 and 2.2 and Table 1](https://cinc.org/archives/2024/pdf/CinC2024-398.pdf)
describes these stages:

| Stage | Data and role | Reported categories |
| --- | --- | --- |
| ImageNet initialization | Start from general image features | ImageNet vocabulary, not Challenge diagnoses |
| Initial ECG adaptation | Synthetic CODE15 images and real InCor-AMB images | Six shared categories: `1dAVb`, `RBBB`, `LBBB`, `SB`, `ST`, `AF` |
| Further ECG adaptation | Real InCor-EMG emergency ECG images | 12 named categories in the paper, listed below |
| Final Challenge fine-tuning | PTB-XL images, five folds | The 11 final Challenge classes |

The six initial ECG categories mean first-degree atrioventricular block, right
and left bundle branch block, sinus bradycardia, sinus tachycardia and atrial
fibrillation, respectively.

These stages describe the published method. The pinned pretraining
[model card](https://huggingface.co/fmenegui/pretrained_cinc_of5/blob/831f9bdb728a502f214831a26d0457738645ffc3/README.md)
only identifies CODE and InCor-EMG and gives an original checkpoint path. It
does not supply the complete run configuration or its class vocabulary.

### Categories reported for InCor-EMG

The paper's Table 1 names the following 12 categories. **This table is not a
verified output-index mapping for the released checkpoint.**

| Category reported in the paper | Meaning |
| --- | --- |
| `RBBB` | Right bundle branch block |
| `LBBB` | Left bundle branch block |
| `AF` | Atrial fibrillation |
| `NORM` | Normal ECG |
| `PVC` | Premature ventricular complex |
| `PSVT / ST` | Paroxysmal supraventricular tachycardia or sinus tachycardia |
| Pacemaker | Pacemaker category |
| ST-segment elevation | ST-segment elevation category |
| Current of Injury | Current of injury category |
| Atrial tachycardia / Atrial Flutter | Combined category |
| 2nd/3rd/Adv AV block | Second-degree, third-degree or advanced atrioventricular block |
| `VT / VF` | Ventricular tachycardia or ventricular fibrillation |

### Checkpoint output metadata

Direct inspection confirms `model.head.fc.weight` has shape **`(13, 768)`**
and `model.head.fc.bias` has shape **`(13,)`**. The original configuration also
constructs a model with `num_classes=13` before loading it.

However, the checkpoint contains no class-name list, and the available original
configuration and model card do not define that list. The paper names 12
InCor-EMG categories. **The exact 13 class names, their order and the reason for
the additional output cannot be established from these sources.** In particular,
do not assume the extra output means "Other" or "Normal", or that the paper's
table order matches the checkpoint.

Recovering the original pretraining run configuration, label encoder or training
CSV with its target-column order is necessary to document those 13 outputs.
Until then, the manifest records their class list as `null`, rather than
assigning unverified names. This checkpoint is provided for initialization,
not for interpreting its original 13 predictions through this release.

### Transfer to the Challenge task

For every fold, [`training_model()`](../aimed/model.py) performs these steps:

1. Build a 13-output ConvNeXt-Tiny and load the checkpoint strictly.
2. Retain its learned feature extractor.
3. Replace the 13-output prediction head with a newly initialized 11-output head.
4. Fine-tune the feature extractor and new head on the 11 Challenge labels.
5. Save an 11-output checkpoint that the inference loader can use.

The original 13 prediction outputs are discarded. They are not renamed, merged
or mapped into the final 11 classes. Their unknown order does not prevent this
feature-transfer step, because the old prediction head is replaced. The extra,
unused `classifier` layer in the original state dictionary is also recreated
with 11 outputs for checkpoint compatibility.

Private InCor data are not distributed. Downloading the checkpoint lets you
start final fine-tuning, but does not recreate the earlier training stages.

## Model weights versus loss weights

The downloadable `.ckpt` files contain learned **model weights**.
`CLASS_WEIGHTS` in `aimed/model.py` are different: they are 11 fixed numerical
**loss weights**, in the final class order, used to weight the BCE training
objective. They are not additional models or pretraining classes. See the
[training tutorial](training.md) for the loss configuration.
