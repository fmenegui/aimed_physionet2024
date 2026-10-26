"""User-facing training CLI. Standard Challenge entry: train_model.py."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description="Train AIMED ConvNeXt on labeled ECG images.")
    parser.add_argument("-d", "--data-folder", "--data_folder", dest="data_folder", required=True)
    parser.add_argument("-m", "--model-folder", "--model_folder", dest="model_folder", required=True)
    parser.add_argument("--init", choices=["ecg", "imagenet", "none"], default="ecg")
    parser.add_argument("--pretrained-checkpoint", default="pretrained_model/model_fold0.ckpt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-find", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=-1, help="Zero-based fold, or -1 for all folds.")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true", help="One train/validation batch for installation checks.")
    return parser


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.folds < 2 or args.size < 32:
        raise ValueError("Use epochs >= 1, batch-size >= 1, folds >= 2 and size >= 32.")
    if args.fold < -1 or args.fold >= args.folds or args.num_workers < 0:
        raise ValueError("fold must be -1 or in [0, folds). num-workers must be nonnegative.")
    if not Path(args.data_folder).is_dir():
        raise FileNotFoundError(f"Data folder not found: {args.data_folder}")
    if args.init == "ecg" and not Path(args.pretrained_checkpoint).is_file():
        raise FileNotFoundError("Run python download_weights.py --kind pretrained first, "
                                "or select --init imagenet / --init none.")
    output = Path(args.model_folder).resolve()
    if (output / "run_config.json").exists():
        raise FileExistsError(f"{output} already contains a training run. Choose a new output folder.")
    import lightning as L
    from config_dx import make_config
    from media.train.train_dx import train_dx
    L.seed_everything(args.seed, workers=True)
    config = make_config(args)
    packages = ["torch", "torchvision", "timm", "lightning", "albumentations", "numpy", "scikit-learn"]
    metadata = {**vars(args), "classes": config.classes, "threshold": config.threshold,
                "versions": {name: importlib.metadata.version(name) for name in packages}}
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    train_dx(config)


if __name__ == "__main__":
    run(get_parser().parse_args())
