import lightning as L
import warnings
import torch
import torch.nn as nn

class BaseClassification(L.LightningModule):
    def __init__(self,
                 model,
                 loss_fn=None,
                 metrics_dict=None,
                 lr=None,
                 optimizer_fn=None,
                 scheduler_fn=None,
                 task='binary',
                 threshold=0.5,
                 save_model=False):
        super().__init__()
        # if save_model: self.save_hyperparameters('model') # Saving model hyperparameters is slow.
        self.lr = lr
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.task = task
        self.threshold = threshold

        if loss_fn is None:
            if self.task == 'binary' or self.task == 'multilabel':
                self.loss_fn = nn.BCEWithLogitsLoss()
            elif self.task == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError("Invalid task type. Supported types: 'binary', 'multilabel', 'multiclass'")
        else:
            self.loss_fn = loss_fn

        if (self.task in ['binary', 'multilabel'] and not isinstance(self.loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss))) or (self.task == 'multiclass' and not isinstance(self.loss_fn, nn.CrossEntropyLoss)):
            warnings.warn(f"Potential mismatch between task '{self.task}' and the provided loss function '{self.loss_fn.__class__.__name__}'.", UserWarning)

        # self.metrics = {k: v.to(self.device) for k, v in metrics_dict.items()}
        self.metrics = metrics_dict
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.optimizer = None
        self.scheduler = None

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage=None):
        images, labels = batch
        # print(images.dtype, labels.dtype, images.shape, labels.shape)
        outputs = self(images)
        outputs = outputs.float()

        if len(outputs.shape) == 1: outputs = outputs.unsqueeze(dim=1)
        if len(outputs.shape) == 0: outputs = outputs.unsqueeze(dim=0)
        if len(labels.shape) == 1: labels = labels.unsqueeze(dim=1)

        loss = self.loss_fn(outputs, labels)

        if self.task in ['binary', 'multilabel']:
            preds = torch.sigmoid(outputs) > self.threshold
        elif self.task == 'multiclass':
            preds = torch.softmax(outputs, dim=1).argmax(dim=1)
        else:
            raise ValueError(f"Unsupported task '{self.task}'. Valid options are 'binary', 'multilabel', 'multiclass'.")

        # self.metrics = {k: v.to(self.device) for k, v in self.metrics.items()}
        # print(self.metrics)
        # for k, v in self.metrics.items(): self.log(f"{stage}_{k}", v(preds.float(), labels), on_step=False, on_epoch=True, prog_bar=True)

        # metrics = {f"{stage}_{k}": v(preds.float(), labels) for k, v in self.metrics.items()} if stage else {}
        if self.scheduler is not None: current_lr = self.scheduler.get_last_lr()[0]
        else: current_lr = self.optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log_dict({f"{stage}_loss": loss, **metrics}, on_step=False, on_epoch=True, prog_bar=True)
        # return loss
        return {'loss':loss, 'pred_labels':preds,'actual_labels':labels}

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        self.optimizer = self.optimizer_fn(self.parameters()) if self.optimizer_fn else torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        if self.scheduler_fn is not None:
            self.scheduler = self.scheduler_fn(self.optimizer)
            return [self.optimizer], [self.scheduler]
        return self.optimizer
