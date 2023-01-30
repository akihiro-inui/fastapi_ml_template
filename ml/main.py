import os
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
from common_tools.src.config_loader import load_config


class Module(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, experiment_name: str, use_mlflow: bool, config_file_path: str, **kwargs):
        super().__init__()
        load_config(config_file_path)
        self.save_hyperparameters()
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
        self.writer = MlflowWriter(tracking_uri=os.environ.get("MLFLOW_DB_URI"),
                                   registry_uri=os.environ.get("MLFLOW_ARTIFACT_URI"),
                                   experiment_name=self.experiment_name)
        tags = {MLFLOW_RUN_NAME: run_name,
                MLFLOW_USER: user_name,
                MLFLOW_SOURCE_NAME: source_name
                }
        self.writer.create_new_run(tags)
        self.model_version = self.writer.create_new_model(model_name=self.model_name, model_source="")
        self.model_file_path = f"models/{self.model_name}/{self.model_name}-{self.model_version}.ckpt"

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
            self.writer.log_metric_step("train_loss", float(loss), batch_idx)
            self.writer.log_metric_step("train_mae", float(mae), batch_idx)
            self.writer.log_metric_step("train_psnr", float(psnr), batch_idx)
        else:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_psnr", psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
            self.writer.log_metric_step("val_loss", float(loss), batch_idx)
            self.writer.log_metric_step("val_mae", float(mae), batch_idx)
            self.writer.log_metric_step("val_psnr", float(psnr), batch_idx)
        else:
            self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
            self.log("val_mae", mae, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
            self.log("val_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

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
            self.log("test_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
            self.log("test_mae", mae, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
            self.log("test_psnr", psnr, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        return loss

    def save_model(self):
        torch.save(self.state_dict(), self.model_file_path)

        # Push to MLFlow
        self.writer.log_model(self.model, self.model_name)
        self.writer.log_artifact(self.model_file_path)

    def load_model(self, model_path: str):
        self.load_state_dict(torch.load(model_path))


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
                    config_file_path=".env",
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

    # Train model
    x = torch.randn(1, 3, 128, 128)
    output_initial = module.model(x)
    trainer.fit(
        module,
        train_dataloader,
        val_dataloader,
    )

    # Save model
    module.save_model()

    # Test model
    trainer.test(module, test_dataloader)

    # Load model
    TestModule = Module(model,
                        config_file_path=".env",
                        experiment_name="Image Super Resolution",
                        use_mlflow=False)

    TestModule.load_model(module.model_file_path)

    if module.use_mlflow:
        module.writer.set_terminated()

    # Inference
    model = TestModule.model
    output_train = module.model(x)
    output_test = model(x)
