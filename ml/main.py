from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import CombinedLoader

from ml.src.models.model_wrapper import ModelWrapper
from ml.src.models.model_selector import ModelSelector
from ml.src.datasets.dataset_selector import DatasetSelector
from common_tools.src.file_handler import load_json_file
from common_tools.src.config_loader import load_env_file
from common_tools.src.custom_logger import logger


class ExperimentRunner:
    def __init__(self, env_file_path: str, config_file_path: str):
        """
        Load config file.
        Initialize model wrapper and Data Loader.
        :param env_file_path: Path to env file.
        :param config_file_path: Path to config file.
        """
        # Load .env file and experiment config file
        load_env_file(env_file_path)
        self.config = load_json_file(config_file_path)

        # Initialize model
        self.model_selector = ModelSelector(self.config["dataset"]["scale_factor"])
        self.model = self.model_selector.select_model(self.config["model"]["name"])

        # Initialize experiment monitoring
        self.experiment_name = self.config["tracking"]["experiment_name"]
        # self.monitor = Monitor(use_mlflow=self.config["tracking"]["use_mlflow"])

        # Initialize model wrapper
        self.model_wrapper = ModelWrapper(model=self.model,
                                          experiment_name=self.experiment_name,
                                          use_mlflow=self.config["tracking"]["use_mlflow"])

        # Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

        # Define trainer
        if self.device.type == 'cuda':
            self.trainer = pl.Trainer(accelerator='gpu', devices=1)
        else:
            self.trainer = pl.Trainer(accelerator='cpu', max_epochs=self.config["model"]["max_epochs"])

    def load_dataset(self) -> Dict[str, Dict[str, DataLoader]]:
        # Place holder for dataset loader
        dataset_dict = {}

        # Initialize dataset
        dataset_selector = DatasetSelector(self.config["dataset"]["scale_factor"])

        def load_dataset_split(dataset_name_list: List[str], split_name: str):
            dataset_loader_dict = {}
            for dataset_name in dataset_name_list:
                dataset = dataset_selector.select_dataset(dataset_name)
                train_dataloader = DataLoader(dataset,
                                              batch_size=self.config["dataset"][f"{split_name}_batch_size"])
                dataset_loader_dict[dataset_name] = train_dataloader
            return CombinedLoader(dataset_loader_dict)

        # Load train, val and test dataset
        dataset_dict["train"] = load_dataset_split(self.config["dataset"]["train_dataset_names"], "train")
        dataset_dict["val"] = load_dataset_split(self.config["dataset"]["val_dataset_names"], "val")
        dataset_dict["test"] = load_dataset_split(self.config["dataset"]["test_dataset_names"], "test")

        return dataset_dict

    def train(self, train_dataset_loader_dict: dict, val_dataset_loader_dict: dict):
        """
        Run the training process and save the model.
        """
        # Train model
        self.trainer.fit(
            self.model_wrapper,
            train_dataloaders=train_dataset_loader_dict,
            val_dataloaders=val_dataset_loader_dict
        )

        # Save model
        self.model_wrapper.save_model()

    def test(self, test_dataset_loader_dict: dict):
        """
        Run the test process.
        :return:
        """
        # Test model
        self.trainer.test(self.model_wrapper, test_dataset_loader_dict)

        # Terminate MLFlow Process
        if self.model_wrapper.use_mlflow:
            self.model_wrapper.writer.set_terminated()

        # Load loading model
        test_model_wrapper = ModelWrapper(self.model,
                                          experiment_name=self.config["tracking"]["experiment_name"],
                                          use_mlflow=False)

        test_model_wrapper.load_model(self.model_wrapper.model_file_path)

        # Test inference
        x = torch.randn(1, 3, 128, 128)
        model = test_model_wrapper.model
        output_train = self.model_wrapper.model(x)
        output_test = model(x)


if __name__ == '__main__':
    experiment_runner = ExperimentRunner(env_file_path=".env",
                                         config_file_path="config/srcnn.json")

    dataset_loader = experiment_runner.load_dataset()

    experiment_runner.train(train_dataset_loader_dict=dataset_loader["train"],
                            val_dataset_loader_dict=dataset_loader["val"])

    experiment_runner.test(test_dataset_loader_dict=dataset_loader["test"])
