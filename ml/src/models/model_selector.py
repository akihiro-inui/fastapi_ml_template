from ml.src.models.cnn import SRCNN
from ml.src.models.vdsr import VDSR
from ml.src.models.espcn import ESPCN
from ml.src.models.srresnet import SRResNet
from ml.src.models.edsr import EDSR

from common_tools.src.custom_error_handlers import ModelError


class ModelSelector:
    def __init__(self, scale_factor: int):
        """
        Initialize model selector map.
        Load the model class here and store them into dictionary.
        :param scale_factor: Scale factor for the model
        """
        self.srcnn = SRCNN(scale_factor)
        self.vdsr = VDSR(scale_factor)
        self.espcn = ESPCN(scale_factor)
        self.srresnet = SRResNet(scale_factor)
        self.edsr = EDSR(scale_factor)

        # Model selector
        self.model_map = {
            "SRCNN": self.srcnn,
            "VDSR": self.vdsr,
            "ESPCN": self.espcn,
            "SRResNet": self.srresnet,
            "EDSR": self.edsr,
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
