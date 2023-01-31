import os
import torch
import torch.nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Tuple, Union, Dict
from pytorch_lightning import LightningModule

from common_tools.src import metrics
import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER, MLFLOW_SOURCE_NAME
from common_tools.src.ml_monitoring import MlflowWriter


class ModelWrapper(LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 experiment_name: str,
                 use_mlflow: bool,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.model = model
        self.model_name = model.get_model_name()
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        if self.use_mlflow:
            self._init_monitoring(**kwargs)
            self.model_file_path = f"models/{self.model_name}/{self.model_name}-{self.model_version}.ckpt"
        else:
            self.model_file_path = f"models/{self.model_name}/{self.model_name}.ckpt"

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

    # def train_dataloader(self):
    #     return load_dataset_split(self.config["dataset"]["train_dataset_names"], "train")
    #
    # def val_dataloader(self):
    #     return load_dataset_split(self.config["dataset"]["train_dataset_names"], "train")
    #
    # def test_dataloader(self):
    #     return load_dataset_split(self.config["dataset"]["train_dataset_names"], "train")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        dateset_name_metrics = {}
        for dataset_name, dataset in batch.items():
            lr, hr = dataset
            sr = self(lr)

            # Loss and metrics
            loss = F.mse_loss(sr, hr, reduction="mean")
            mae = metrics.mae(sr, hr)
            psnr = metrics.psnr(sr, hr)
            dateset_name_metrics[dataset_name] = {"loss": loss, "mae": mae, "psnr": psnr}

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
        """
        Run validation step for each validation dataset
        :param batch: Dict of validation datasets
        :param batch_idx: Batch index
        :return:
        """
        dateset_name_metrics = {}
        for dataset_name, dataset in batch.items():
            lr, hr = dataset
            sr = self(lr)

            # Loss and metrics
            loss = F.mse_loss(sr, hr, reduction="mean")
            mae = metrics.mae(sr, hr)
            psnr = metrics.psnr(sr, hr)
            dateset_name_metrics[dataset_name] = {"loss": loss, "mae": mae, "psnr": psnr}

            # Log metrics
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
        dateset_name_metrics = {}
        for dataset_name, dataset in batch.items():
            lr, hr = dataset
            sr = self(lr)

            # Loss and metrics
            loss = F.mse_loss(sr, hr, reduction="mean")
            mae = metrics.mae(sr, hr)
            psnr = metrics.psnr(sr, hr)
            dateset_name_metrics[dataset_name] = {"loss": loss, "mae": mae, "psnr": psnr}

            # Log metrics
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
        if self.use_mlflow:
            self.writer.log_model(self.model, self.model_name)
            self.writer.log_artifact(self.model_file_path)

    def load_model(self, model_path: str):
        self.load_state_dict(torch.load(model_path))
