# Documentation references

The following primary repositories were inspected while organizing this release.
Their documentation structure informed the guides. Their methods and scores
remain separate from AIMED's.

| Repository | Useful documentation pattern | Used here |
| --- | --- | --- |
| [ECG-Digitiser, SignalSavants](https://github.com/felixkrones/ECG-Digitiser) | Quick start, pretrained use, image assumptions, separate training and data preparation | Separate inference and training paths, explicit data assumptions |
| [DSAIL](https://github.com/Antony-gitau/physionet-challenge-2024-dsail) | Installation, train/run/evaluate commands and arguments, PTB-XL preparation | Commands with input/output folder explanations |
| [wenh06/cinc2024](https://github.com/wenh06/cinc2024) | Architecture overview, file/module descriptions, experiment and submission records, disclosed implementation issue | Source provenance, reproducibility table, explicit discrepancies |
| [Official Python example](https://github.com/physionetchallenges/python-example-2024) | Standard Challenge entry points, data conversion and evaluation | Preserve compatible wrappers and original helper/evaluation scripts |

ECG-Digitiser addresses digitization and DSAIL uses a different classifier.
They are references for documentation, not components of the AIMED model.

## Scientific sources

- [AIMED paper, DOI 10.22489/CinC.2024.398](https://doi.org/10.22489/CinC.2024.398)
- [PhysioNet Challenge 2024](https://moody-challenge.physionet.org/2024/)
- [Official results](https://moody-challenge.physionet.org/2024/results/)
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit)

No other team's model or reported score is presented as an AIMED result.
