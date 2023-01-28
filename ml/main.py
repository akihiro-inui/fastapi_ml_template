import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from ml.src.datasets.bsds300 import BSDS300
from ml.src.datasets.bsds100 import BSDS100
from ml.src.datasets.set14 import Set14
from ml.src.datasets.set5 import Set5
from ml.src.models.cnn import SRCNN
from common_tools.src import metrics
import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER, MLFLOW_SOURCE_NAME
from common_tools.src.ml_monitoring import MlflowWriter


class Module(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, experiment_name: str, use_mlflow: bool, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = model.get_model_name()
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        if self.use_mlflow:
            self._init_monitoring(**kwargs)

    def _init_monitoring(self,
                         run_name: str = "test_run",
                         user_name: str = "test_user",
                         source_name: str = "test_source"):
        self.writer = MlflowWriter(experiment_name=self.experiment_name)
        tags = {MLFLOW_RUN_NAME: run_name,
                MLFLOW_USER: user_name,
                MLFLOW_SOURCE_NAME: source_name
                }
        self.writer.create_new_run(tags)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")

        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        if self.use_mlflow:
            self.writer.log_metric("train_loss", float(loss))
            self.writer.log_metric("train_mae", float(mae))
            self.writer.log_metric("train_psnr", float(psnr))
        else:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_mae", mae)
            self.log("train_psnr", psnr)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")

        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        if self.use_mlflow:
            self.writer.log_metric("val_loss", float(loss))
            self.writer.log_metric("val_mae", float(mae))
            self.writer.log_metric("val_psnr", float(psnr))
        else:
            self.log("val_loss", loss)
            self.log("val_mae", mae)
            self.log("val_psnr", psnr)

        return loss

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")

        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        if self.use_mlflow:
            self.writer.log_metric("test_loss", float(loss))
            self.writer.log_metric("test_mae", float(mae))
            self.writer.log_metric("test_psnr", float(psnr))
        else:
            self.log("test_loss", loss)
            self.log("test_mae", mae)
            self.log("test_psnr", psnr)

        return loss

    def save_model(self):
        model_version = self.writer.create_new_model(model_name=module.model_name, model_source="")
        torch.save(self.state_dict(), f"models/{self.model_name}/{self.model_name}-{model_version}.pt")

        # Push to MLFlow
        self.writer.log_model(self.model, self.model_name)
        self.writer.log_artifact(f"models/{self.model_name}/{self.model_name}-{model_version}.pt")


if __name__ == '__main__':
    scale_factor = 2

    # Setup dataloaders
    train_dataset = BSDS100(scale_factor=scale_factor)
    val_dataset = Set14(scale_factor=scale_factor)
    test_dataset = Set5(scale_factor=scale_factor)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Define model
    channels = 3 if train_dataset.color_space == "RGB" else 1
    model = SRCNN(scale_factor, channels)
    module = Module(model,
                    experiment_name="Image Super Resolution",
                    use_mlflow=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        trainer = pl.Trainer(accelerator='gpu', devices=1)
    else:
        trainer = pl.Trainer(accelerator='cpu', max_epochs=2)
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, module.model_name)

    trainer.fit(
        module,
        train_dataloader,
        val_dataloader,
    )

    trainer.test(module, test_dataloader)
    module.save_model()

    if module.use_mlflow:
        module.writer.set_terminated()
