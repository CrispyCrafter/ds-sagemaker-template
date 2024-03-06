import pytorch_lightning as pl
import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from src.loss import *


class BaseNet(pl.LightningModule):
    def __init__(self, config, embeddings):
        super().__init__()
        self.predict_step_outputs = []
        self.train_losses = []
        self.val_losses = []

        self.prediction = None
        self.config = config

        self.loss_func = getattr(config.hyperparameters.loss_func)

        self.automatic_optimization = False
        self.scaler = GradScaler()

        self.embeddings = embeddings
        self.input_size = len(config.features()) + sum(
            self.embeddings.output_size.values()
        )

        # Required to generate ONNX output
        self.example_input_array = (
            # Features
            torch.randn(self.config.hyperparameters.batch_size, len(config.features())),
            # Dx
            torch.randn(self.config.hyperparameters.batch_size),
            # Categories
            torch.randn(
                self.config.hyperparameters.batch_size, len(config.categories())
            ),
        )

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        opt = self.optimizers()
        clip_value = self.config.hyperparameters.gradient_clip
        features, cohort_date, dx, dnu, labels, categories = batch[0]
        with autocast():
            output = self(x=features, dx=dx, categories=categories)
            loss = self.loss_func(output, labels, dnu, self.config)

        self.scaler.scale(loss).backward()
        clip_grad_norm_(self.parameters(), clip_value)
        self.scaler.step(opt)

        self.scaler.update()
        opt.zero_grad()

        wandb.log({"train_loss": loss})
        self.log("train_loss", loss)
        self.train_losses.append(loss)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()

        wandb.log({"avg_train_loss": avg_loss})
        self.log("avg_train_loss", avg_loss)
        self.train_losses = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        features, cohort_date, dx, dnu, labels, categories = batch
        output = self(x=features, dx=dx, categories=categories)
        loss = self.loss_func(output, labels, dnu, self.config)

        wandb.log({"val_loss": loss})
        self.log("val_loss", loss)
        self.val_losses.append(loss)

        return loss

    def on_val_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()

        wandb.log({"avg_val_loss": avg_loss})
        self.log("avg_val_loss", avg_loss)
        self.val_losses = []

    def configure_optimizers(self):
        if self.config.hyperparameters.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.hyperparameters.learning_rate,
                momentum=0.9,
            )
        elif self.config.hyperparameters.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.hyperparameters.learning_rate,
            )
        else:
            raise Exception(
                "Unknown optimizer selected : "
                + self.config.hyperparameters.optimizer_name
            )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=self.config.hyperparameters.decayRate
        )
        return [optimizer], [lr_scheduler]

    def export_model(self, model_path):
        self.to_onnx(  # model being run
            model_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=[
                "features",
                "days_since_install",
                "categories",
            ],  # the model's input names
            output_names=[self.config.label_name],  # the model's output names
            dynamic_axes={
                "features": {0: "batch_size"},  # variable length axes
                "days_since_install": {0: "batch_size"},  # variable length axes
                "categories": {0: "batch_size"},  # variable length axes
                "retention": {0: "batch_size"},  # variable length axes
            },
        )
