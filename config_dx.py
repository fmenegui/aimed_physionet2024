"""Portable configuration derived from the final cinc_of5 submission.

Importing this module never creates directories or downloads weights.
"""
from pathlib import Path
from types import SimpleNamespace
import torch
from aimed.model import CLASSES, CLASS_WEIGHTS, make_transforms, training_model

def make_config(args):
    output = Path(args.model_folder).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = "gpu" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu.")
    config = SimpleNamespace(
        data_img_dir=str(Path(args.data_folder).resolve()), resized_path=None,
        save_dir=str(output), save_model=True, log=True, timers={},
        size=args.size, data_perc=1.0, num_workers=args.num_workers,
        threshold=0.5, find_lr=args.lr_find,
        image_path_columns=["image_full_path"], group_column="header_base",
        classes=list(CLASSES), task="multilabel", f1_score_avg="macro",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        kfold_splits=args.folds, folder_index=args.fold,
        early={"use": True, "monitor": "val_loss", "verbose": True,
               "patience": args.patience, "mode": "min", "min_delta": 0.001,
               "best_weights": False},
        trainer_kwargs={"precision": "16-mixed" if device == "gpu" else "32-true",
                        "accelerator": device, "devices": 1,
                        "enable_progress_bar": not args.quiet},
        loss_fn=torch.nn.BCEWithLogitsLoss(weight=torch.tensor(CLASS_WEIGHTS)),
        scheduler=None, use_meta=False, metrics_log=None, metrics_nolog=None,
        transforms={"train": make_transforms(training=True),
                    "valid": make_transforms(), "test": make_transforms()},
        df=None,
    )
    config.optimizer = lambda parameters: torch.optim.AdamW(parameters, lr=config.lr)
    config.model = lambda: training_model(args.init, args.pretrained_checkpoint)
    if args.fast_dev_run:
        config.trainer_kwargs["fast_dev_run"] = True
    return config
