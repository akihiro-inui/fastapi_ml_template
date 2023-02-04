import os
from typing import List, Dict, Any

import mlflow
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
        1. Load config files.
        2. Initialize model
        3. Initialize model trainer
        :param env_file_path: Path to env file.
        :param config_file_path: Path to config file.
        """
        # Load .env file and experiment config file
        load_env_file(env_file_path)
        self.config = load_json_file(config_file_path)

        # Initialize model
        self.model_selector = ModelSelector(self.config["dataset"]["scale_factor"])
        self.model = self.model_selector.select_model(self.config["model"]["name"])

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(self.config["monitoring"]["experiment_name"])
        experiment_id = mlflow.get_experiment_by_name(self.config["monitoring"]["experiment_name"]).experiment_id

        # Initialize experiment monitoring
        self.run_info = {"experiment_id": experiment_id}

        # Initialize model with wrapper
        self.model_wrapper = ModelWrapper(model=self.model)

        # Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

        # Initialize trainer
        if self.device.type == 'cuda':
            self.trainer = pl.Trainer(accelerator='gpu',
                                      devices=1)
        else:
            self.trainer = pl.Trainer(accelerator='cpu',
                                      max_epochs=self.config["model"]["max_epochs"])

    def load_dataset(self) -> Dict[str, CombinedLoader]:
        """
        Load dataset and return a dictionary of data loaders.
        :return: Dictionary of data loaders. {train: train_loader, val: val_loader, test: test_loader}
        """
        # Place holder for dataset loader
        dataset_dict = {}

        # Initialize dataset
        dataset_selector = DatasetSelector(self.config["dataset"]["scale_factor"])

        def _load_dataset_split(dataset_name_list: List[str], split_name: str) -> CombinedLoader:
            """
            Load dataset split.
            :param dataset_name_list: List of dataset names
            :param split_name: Name of split
            :return: CombinedLoader object on top of DataLoader dictionary like {"MNIST": DataLoader, "CIFAR10": DataLoader}
            """
            dataset_loader_dict = {}
            for dataset_name in dataset_name_list:
                dataset = dataset_selector.select_dataset(dataset_name)
                train_dataloader = DataLoader(dataset,
                                              batch_size=self.config["dataset"][f"{split_name}_batch_size"])
                dataset_loader_dict[dataset_name] = train_dataloader
            return CombinedLoader(dataset_loader_dict)

        # Load train, val and test dataset
        dataset_dict["train"] = _load_dataset_split(self.config["dataset"]["train_dataset_names"], "train")
        dataset_dict["val"] = _load_dataset_split(self.config["dataset"]["val_dataset_names"], "val")
        dataset_dict["test"] = _load_dataset_split(self.config["dataset"]["test_dataset_names"], "test")

        return dataset_dict

    def train(self, train_dataset_loader: CombinedLoader, val_dataset_loader: CombinedLoader):
        """
        Run the training process and save the model.
        """
        # Train model
        mlflow.pytorch.autolog()
        with mlflow.start_run(run_name=self.config["monitoring"]["run_name"]) as run:
            self.trainer.fit(
                self.model_wrapper,
                train_dataloaders=train_dataset_loader,
                val_dataloaders=val_dataset_loader,
            )
            self.run_info["run_id"] = run.info.run_id
            self.run_info["run_name"] = run.info.run_name

    def test(self, test_dataset_loader: CombinedLoader) -> Dict[str, Any]:
        """
        Run the test process.
        :return: Output of test result?
        """
        # Test model
        self.trainer.test(self.model_wrapper, test_dataset_loader)

        # Load trained model
        test_model = self.model_wrapper.load_model(f"{os.environ.get('MLFLOW_REGISTRY_URI')}/{self.run_info['experiment_id']}/{self.run_info['run_id']}/artifacts/model")

        # Test inference
        x = torch.randn(1, 3, 128, 128)
        test_output = test_model(x)
        return test_output


if __name__ == '__main__':
    experiment_runner = ExperimentRunner(env_file_path=".env",
                                         config_file_path="config/srcnn.json")

    dataset_loader = experiment_runner.load_dataset()

    experiment_runner.train(train_dataset_loader=dataset_loader["train"],
                            val_dataset_loader=dataset_loader["val"])

    experiment_runner.test(test_dataset_loader=dataset_loader["test"])
