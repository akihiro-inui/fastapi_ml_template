from ml.src.models.cnn import SRCNN
from common_tools.src.custom_error_handlers import ModelError


class ModelSelector:
    def __init__(self, scale_factor: int):
        """
        Initialize model selector map.
        Load the model class here and store them into dictionary.
        :param scale_factor: Scale factor for the model
        """
        self.srcnn = SRCNN(scale_factor)

        # Model selector
        self.model_map = {
            "SRCNN": self.srcnn
        }

    def select_model(self, model_name: str):
        """
        Select model from pre-defined map
        :param model_name: Name of model
        """
        try:
            return self.model_map[model_name]
        except KeyError:
            raise ModelError(f"Selected model '{model_name}' does not exist")
        except Exception as err:
            raise ModelError(f"Error while selecting model: {err}")
