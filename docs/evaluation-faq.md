# Local evaluation FAQ

This page answers the three questions that commonly arise when evaluating the
AIMED ECG image classifier with public or artificial images. Private InCor
patient data are not needed for inference or for the software check below.

## 1. Which weights should I use, and are they the exact files that scored 0.817?

Use the maintained code in
[`fmenegui/aimed_physionet2024`](https://github.com/fmenegui/aimed_physionet2024)
with the following pinned Hugging Face artifacts:

| Purpose | Hugging Face repository | Revision | Files |
| --- | --- | --- | --- |
| Final five-fold inference ensemble | [`fmenegui/cinc_of5`](https://huggingface.co/fmenegui/cinc_of5) | `1ab856738289e4c0b76c4df323ef33cfdc45f1f2` | `model_fold0.ckpt`, `model_fold1.ckpt`, `model_fold2.ckpt`, `model_fold3.ckpt`, `model_fold4.ckpt` |
| Initialize a new training run | [`fmenegui/pretrained_cinc_of5`](https://huggingface.co/fmenegui/pretrained_cinc_of5) | `831f9bdb728a502f214831a26d0457738645ffc3` | `model_fold0.ckpt` |

The final ensemble has five models, each predicting the same **11 Challenge
classes**. The separate pretraining checkpoint has 13 outputs and initializes
training before its prediction head is replaced. It is not an inference fold.

**The historical 0.817 score belongs to AIMED's official 2024 classification
entry.** The inspected submission source downloads weights from the same
Hugging Face repository names, but does not pin their revisions. Consequently,
the available records do not independently establish that the currently pinned
files are byte-for-byte identical to the files used during official scoring.
Nor has the maintained release been re-evaluated on the official hidden test.
The [reproducibility record](reproducibility.md) documents that distinction and
the release's preprocessing changes.

The supported statement is that these are the released five-fold artifacts
selected, integrity-checked and tested for this maintained code. Confirming
their exact historical identity requires the original scored submission's
checkpoint hashes or an equivalent archived artifact record.

### Legacy code

Treat it as a different source snapshot when following this tutorial. At
[`384e3bb3e82affeb167983b9219c6447b4548c3d`](https://github.com/fmenegui/cinc_of2/commit/384e3bb3e82affeb167983b9219c6447b4548c3d):

- Its [`config_dx.py`](https://github.com/fmenegui/cinc_of2/blob/384e3bb3e82affeb167983b9219c6447b4548c3d/config_dx.py)
  configures ten folds, batch size 32 and CLAHE preprocessing.
- Its [`Dockerfile`](https://github.com/fmenegui/cinc_of2/blob/384e3bb3e82affeb167983b9219c6447b4548c3d/Dockerfile)
  downloads `fmenegui/cinc_of1` and `fmenegui/cinc_of_pretrained`.

This tutorial uses a five-fold configuration, batch size 16 for training and
the explicitly documented inference pipeline. Use its loader and manifest
together with its pinned weights. Compatibility with an arbitrary combination
of legacy code, transforms and Hub files is not established here.

## 2. Is there an executable inference script and a matching manifest?

**Yes.** After the [installation](../README.md#1-install), run this complete
software check from the repository root:

```bash
python download_weights.py --kind ensemble
python examples/make_demo_data.py --output data/demo --records 20
python predict_image.py -i data/demo/demo000-0.png -m model -o outputs/prediction.json --device cpu
```

The demo draws artificial traces and assigns arbitrary labels. It checks that
the software executes, not diagnostic performance. Replace the image path to
classify your own public image. A single image needs no WFDB header or signal.

| Component | Where it is specified |
| --- | --- |
| Checkpoint revisions, filenames, sizes and SHA-256 values | [`weights_manifest.json`](../weights_manifest.json) |
| Strict loading and model construction | [`aimed/model.py`](../aimed/model.py) |
| Single-image CLI and named JSON output | [`predict_image.py`](../predict_image.py) |
| Python API and Challenge-format folder examples | [Inference tutorial](inference.md) |
| Class meanings, exact output order and label preparation rules | [Classes and model weights](models.md) |

The manifest and loader specify whole-page RGB conversion, a LANCZOS resize to
384 × 384, deterministic three-channel grayscale, ImageNet normalization,
sigmoid scores averaged across folds and a strict **`score > 0.5`** threshold
for every class. The 11 scores are independent and need not sum to one.

Preserve the code commit and `weights_manifest.json` with your evaluation
results. The downloader pins the weights and checks their hashes. The JSON
prediction includes the class names, scores, threshold, selected labels and
loaded checkpoint paths.

## 3. What terms apply to code, weights and later product development?

The terms apply separately:

| Material | Current offer | Scope and obligations |
| --- | --- | --- |
| New AIMED-owned software contributions | [PolyForm Noncommercial 1.0.0](../LICENSE) | Use within its permitted purposes, retain its terms and required notices when sharing |
| Documentation and AIMED's copyright interests in the six identified checkpoints | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Noncommercial purposes, with attribution, license information and modification notices when sharing |
| Historical code, third-party code, libraries, datasets and starting weights | Their applicable original terms | Earlier permissions and third-party rights are preserved |

See [LICENSING.md](../LICENSING.md), [MODEL_LICENSE.md](../MODEL_LICENSE.md) and
[NOTICE](../NOTICE) for the exact scope. The downloader places checkpoint
licensing and attribution files alongside the weights.

**A test for a noncommercial purpose is permitted under the relevant grants,
subject to their conditions.** Running locally, using public or synthetic
images and avoiding clinical use do not by themselves establish that purpose.
Under CC BY-NC, the intended commercial advantage matters, rather than merely
whether the user is a company or whether money changes hands.

An internal feasibility test directed toward a commercial product can therefore
fall outside the noncommercial checkpoint offer before any sale or clinical
deployment. If your evaluation is directed toward product development, contact
the maintainers to establish authorization for that scope unless an applicable
existing permission already covers it. No blanket commercial-evaluation grant
is made by this tutorial.

The new offers do not withdraw any earlier BSD, MIT or other valid permission.
The [2024 Challenge rules](https://moody-challenge.physionet.org/2024/#intellectual-property-and-open-source-licenses)
provide a BSD-3-Clause default when a submission specifies no license. The
exact terms accompanying the historical evaluated entry and earlier checkpoint
downloads are not independently established here. The new noncommercial
policy must not be read as retroactively prohibiting all commercial use of
previously released material.

The official [CC BY-NC legal text](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en),
[CC explanation of noncommercial use](https://creativecommons.org/faq/#does-my-use-violate-the-noncommercial-clause-of-the-licenses)
and [PolyForm license](https://polyformproject.org/licenses/noncommercial/1.0.0)
provide the underlying definitions. Dataset permissions remain separate.
