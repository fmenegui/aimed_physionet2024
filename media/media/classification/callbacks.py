from sklearn.metrics import f1_score
from lightning.pytorch.callbacks import Callback
import numpy as np

class BatchAccumulatedMetricsCallback(Callback):
    def __init__(self, metric_to_function_dict=None):
        self.metric_to_function_dict = metric_to_function_dict or {"f1": (f1_score, {'average': 'binary'})}

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_actual_labels_npy = []
        self.train_pred_labels_npy = []

    def on_train_epoch_end(self, trainer, pl_module):
        for m, fn_args in self.metric_to_function_dict.items():
            fn, args = fn_args
            if self.train_actual_labels_npy:
                pl_module.log("train_" + m, fn(np.concatenate(self.train_actual_labels_npy), np.concatenate(self.train_pred_labels_npy), **args), on_step=False, on_epoch=True, prog_bar=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs and "actual_labels" in outputs and "pred_labels" in outputs:
            self.train_actual_labels_npy.append(outputs["actual_labels"].detach().cpu().numpy())
            self.train_pred_labels_npy.append(outputs["pred_labels"].detach().cpu().numpy())

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_actual_labels_npy = []
        self.val_pred_labels_npy = []

    def on_validation_epoch_end(self, trainer, pl_module):
        for m, fn_args in self.metric_to_function_dict.items():
            fn, args = fn_args
            if self.val_actual_labels_npy:
                pl_module.log("val_" + m, fn(np.concatenate(self.val_actual_labels_npy), np.concatenate(self.val_pred_labels_npy), **args), on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs and "actual_labels" in outputs and "pred_labels" in outputs:
            self.val_actual_labels_npy.append(outputs["actual_labels"].detach().cpu().numpy())
            self.val_pred_labels_npy.append(outputs["pred_labels"].detach().cpu().numpy())
