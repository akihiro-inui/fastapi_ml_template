from typing import Optional, Any
from omegaconf import DictConfig, ListConfig

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER, MLFLOW_SOURCE_NAME

from common_tools.src.custom_logger import logger


class MLFlowMonitor:
    def __init__(self,
                 tracking_uri: str,
                 registry_uri: str,
                 artifact_location: str,
                 model_name: str,
                 experiment_name: Optional[str],
                 run_name: Optional[str] = "",
                 user_name: Optional[str] = "",
                 source_name: Optional[str] = ""
                 ):
        """
        Initialize MLFlow Monitor
        :param tracking_uri: MLFlow tracking DB URI
        :param registry_uri: MLFlow registry URI
        :param artifact_location: Artifact location
        :param model_name: Model name
        :param experiment_name: Experiment name
        :param run_name: Run name
        :param user_name: User name
        :param source_name: Source name
        """

        self.model_name = model_name
        self.artifact_location = artifact_location

        # Create MLFlow client
        self.client = MlflowClient(tracking_uri=tracking_uri,
                                   registry_uri=registry_uri)

        # Get or create MLFlow experiment
        self.experiment = self._get_or_create_experiment(experiment_name, artifact_location)

        tags = {MLFLOW_RUN_NAME: run_name,
                MLFLOW_USER: user_name,
                MLFLOW_SOURCE_NAME: source_name
                }

        # Create new run ID
        self.run_id = self._create_new_run(tags)

        # Get or create new model
        self.model_version = self._create_new_model(model_name=model_name,
                                                    model_source=source_name)

        logger.info("New experiment started")
        logger.info(f"Name: {self.experiment.name}")
        logger.info(f"Experiment_id: {self.experiment.experiment_id}")
        logger.info(f"Artifact Location: {self.experiment.artifact_location}")
        logger.info(f"New model version created: {self.model_version}")

    def _get_or_create_experiment(self,
                                  experiment_name: str,
                                  artifact_location: Optional[str] = ""):
        """
        Create or get experiment from MLFlow
        :param experiment_name: Experiment name
        :param artifact_location: Artifact location
        :return: MLFlow Experiment
        """
        try:
            experiment_id = self.client.create_experiment(experiment_name, artifact_location)
        except Exception as e:
            logger.error(e)
            experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        return self.client.get_experiment(experiment_id)

    def _create_new_model(self,
                          model_name: str,
                          model_source: str) -> int:
        """
        Create new model on MLFlow
        :param model_name: Model name
        :param model_source: Model source
        :return: Model version
        """
        try:
            self.client.create_registered_model(model_name)
            # Create model version on MLFlow and return the version number
            model = self.client.create_model_version(
                name=model_name,
                source=model_source,
                run_id=self.run_id
            )
            model_version = model.version
        except Exception as e:
            model_version = self.client.get_latest_versions(model_name)
            logger.error(e)

        return model_version

    def log_params_from_omegaconf_dict(self, params: dict):
        """
        Log parameters from OmegaConf dictionary
        :param params: OmegaConf dictionary
        """
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name: str, element: dict):
        """
        Explore recursively the OmegaConf dictionary and log parameters
        :param parent_name: Parent name
        :param element: Element to be explored
        """
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)
        else:
            self.client.log_param(self.run_id, f'{parent_name}', element)

    def log_param(self, key: str, value: Any):
        """
        Log parameter
        :param key: Parameter key
        :param value: Parameter value
        """
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key: str, value: Any):
        """
        Log metric
        :param key: Metric key
        :param value: Metric value
        """
        self.client.log_metric(self.run_id, key, value)

    def log_metric_step(self, key: str, value: Any, step: int):
        """
        Log metric step
        :param key: Metric key
        :param value: Metric value
        :param step: Metric step
        """
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self):
        """
        Log artifact
        """
        self.client.log_artifact(self.run_id, self.artifact_location)

    def log_dict(self, dictionary: dict, file_path: str):
        """
        Log dictionary
        :param dictionary: Dictionary to be logged
        :param file_path: File path
        """
        self.client.log_dict(self.run_id, dictionary, file_path)

    def log_figure(self, figure, file_path):
        """
        Log figure
        :param figure: Figure to be logged
        :param file_path: File path
        """
        self.client.log_figure(self.run_id, figure, file_path)

    def log_model(self, model, model_name: str):
        """
        Log model
        :param model: Torch model
        :param model_name: Model name
        """
        mlflow.pytorch.log_model(model, model_name)

    def set_terminated(self):
        """
        Set run status to terminated
        """
        self.client.set_terminated(self.run_id)

    def _create_new_run(self, tags: Optional[dict] = None) -> str:
        """
        Create new run on MLFlow
        :param tags: Tags to be added to the run
        """
        self.run = self.client.create_run(experiment_id=self.experiment.experiment_id,
                                          tags=tags)
        run_id = self.run.info.run_id
        return run_id

    def save_model(self, model, model_name: str):
        """
        Save model
        :param model: Torch model
        :param model_name: Model name
        """
        mlflow.pytorch.save_model(model, model_name)

