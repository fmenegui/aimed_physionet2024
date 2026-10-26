"""Classify a PNG/JPEG ECG image without requiring a WFDB header."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-m", "--model-folder", default="model")
    parser.add_argument("-o", "--output", help="Optional JSON output filename.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    from PIL import Image
    from aimed.model import CLASSES, THRESHOLD, checkpoint_files, load_ensemble, predict_probabilities, predicted_labels
    models = load_ensemble(args.model_folder, args.device)
    with Image.open(args.image) as image:
        probabilities = predict_probabilities(models, image)
    result = {"image": str(args.image), "classes": CLASSES,
              "probabilities": dict(zip(CLASSES, probabilities.tolist())),
              "threshold": THRESHOLD, "labels": predicted_labels(probabilities),
              "checkpoints": [str(path) for path in checkpoint_files(args.model_folder)]}
    text = json.dumps(result, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
