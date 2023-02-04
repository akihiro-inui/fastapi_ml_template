import mlflow
import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning import LightningModule
from common_tools.src import metrics


class ModelWrapper(LightningModule):
    def __init__(self,
                 model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x) -> Tensor:
        """
        Inference step
        :param x: input tensor
        :return: output tensor
        """
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer
        :return: Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Run training step for each training dataset
        :param batch: Batch of training datasets
        :param batch_idx: Index of batch
        :return: Loss
        """
        for dataset_name, dataset in batch.items():
            lr, hr = dataset
            sr = self(lr)

            # Loss and metrics
            loss = F.mse_loss(sr, hr, reduction="mean")
            mae = metrics.mae(sr, hr)
            psnr = metrics.psnr(sr, hr)

            metrics_dict = {"loss": float(loss), "mae": float(mae), "psnr": float(psnr)}
            mlflow.log_metrics(metrics_dict, step=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        """
        Run validation step for each validation dataset
        :param batch: Batch of validation datasets
        :param batch_idx: Batch index
        :return: Loss
        """
        for dataset_name, dataset in batch.items():
            lr, hr = dataset
            sr = self(lr)

            # Loss and metrics
            loss = F.mse_loss(sr, hr, reduction="mean")
            mae = metrics.mae(sr, hr)
            psnr = metrics.psnr(sr, hr)

            metrics_dict = {"loss": float(loss), "mae": float(mae), "psnr": float(psnr)}
            mlflow.log_metrics(metrics_dict, step=batch_idx)

        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        """
        Run test step for each test dataset
        :param batch: Batch of test datasets
        :param batch_idx: Batch index
        :return: Loss
        """
        for dataset_name, dataset in batch.items():
            lr, hr = dataset
            sr = self(lr)

            # Loss and metrics
            loss = F.mse_loss(sr, hr, reduction="mean")
            mae = metrics.mae(sr, hr)
            psnr = metrics.psnr(sr, hr)

            metrics_dict = {"loss": float(loss), "mae": float(mae), "psnr": float(psnr)}
            mlflow.log_metrics(metrics_dict, step=batch_idx)

        return loss

    def save_model(self):
        """
        Save model to disk and log it to mlflow
        """
        mlflow.pytorch.save_model(self.model, self.model_name)

        # if self.monitor:
        #     self.monitor.log_model(self.model, self.monitor.model_name)
        #     # self.monitor.log_artifact()
            # self.monitor.save_model(self.model, self.monitor.model_name)

    def load_model(self, model_uri: str):
        """
        Load model from disk
        :param model_uri: Path to model
        """
        return mlflow.pytorch.load_model(model_uri)
