"""Training loop adapted from the submitted five-fold implementation."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import f1_score
import pandas as pd
from media.dataset.base.dataset import BaseDataset
from media.dataset.base.dataloader import BaseDataloader
from media.classification.base.classification import BaseClassification
from media.classification.callbacks import BatchAccumulatedMetricsCallback
from media.predict.predict_from_dl import predict_and_save
from media.train.prepare_from_config import prepare_df, prepare_resize
from media.helpers.log import Tee

def train_dx(config):
    output = Path(config.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    log = Tee(str(output / "output.log")) if config.log else None
    start = time.time()
    metrics = {}
    try:
        if config.resized_path is None:
            config.resized_path = prepare_resize(config)
        if config.df is None:
            config.df = prepare_df(config)
        config.df.to_csv(output / "df.csv", index=False)
        for fold in range(config.kfold_splits):
            if config.folder_index not in (-1, fold):
                continue
            train_idx = config.df.index[config.df[f"fold_{fold}_train"].astype(bool)].tolist()
            val_idx = config.df.index[config.df[f"fold_{fold}_valid"].astype(bool)].tolist()
            if not train_idx or not val_idx:
                raise ValueError(f"Fold {fold} is empty. Provide more labeled records.")
            dl = BaseDataloader(
                BaseDataset, dataframe=config.df, file_column=config.image_path_columns,
                label_columns=config.classes, batch_size=config.batch_size,
                train_indices=train_idx, val_indices=val_idx,
                transforms=config.transforms, num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available())
            model = BaseClassification(
                model=config.model(), loss_fn=config.loss_fn,
                task=config.task, optimizer_fn=config.optimizer, lr=config.lr)
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=config.early["patience"],
                              min_delta=config.early["min_delta"], mode="min"),
                BatchAccumulatedMetricsCallback(
                    {"f1": (f1_score, {"average": "macro", "zero_division": 0})})
            ]
            trainer = L.Trainer(
                max_epochs=config.epochs,
                default_root_dir=str(output),
                logger=CSVLogger(str(output / "logs"), name=f"fold{fold}"),
                callbacks=callbacks, enable_checkpointing=False,
                **config.trainer_kwargs)
            if config.find_lr:
                from lightning.pytorch.tuner import Tuner
                results = Tuner(trainer).lr_find(
                    model, datamodule=dl, min_lr=1e-5, max_lr=1.0,
                    early_stop_threshold=None, update_attr=False)
                suggestion = results.suggestion()
                if suggestion is not None:
                    config.lr = float(suggestion)
                    model.lr = config.lr
                    print(f"Fold {fold}: learning rate {config.lr}")
            trainer.fit(model, datamodule=dl)
            # Preserve the submitted code's last-epoch checkpoint policy.
            target = output / "model" / f"model_fold{fold}.ckpt"
            target.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(target), weights_only=True)
            predictions_file = output / "predictions" / f"validation_fold{fold}.csv"
            predict_and_save(model, dl.val_dataloader(), str(predictions_file),
                             task=config.task, classes=config.classes)
            table = pd.read_csv(predictions_file)
            y = table[[f"Label_{name}" for name in config.classes]].to_numpy()
            p = table[[f"Prediction_{name}" for name in config.classes]].to_numpy()
            per_class = f1_score(y, p > config.threshold, average=None, zero_division=0)
            metrics[str(fold)] = {
                "macro_f1": float(np.mean(per_class)),
                "per_class_f1": dict(zip(config.classes, per_class.tolist())),
                "validation_images": len(val_idx), "learning_rate": config.lr}
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            del model, trainer, dl
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        (output / "timers.json").write_text(json.dumps({"seconds": time.time() - start}, indent=2) + "\n")
    finally:
        if log is not None:
            log.close()
