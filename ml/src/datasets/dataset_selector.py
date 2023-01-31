from common_tools.src.custom_error_handlers import DataSetError

from ml.src.datasets.bsds100 import BSDS100
from ml.src.datasets.bsds200 import BSDS200
from ml.src.datasets.bsds300 import BSDS300
from ml.src.datasets.bsds500 import BSDS500
from ml.src.datasets.set5 import Set5
from ml.src.datasets.set14 import Set14
from ml.src.datasets.manga109 import Manga109
from ml.src.datasets.t91 import T91
from ml.src.datasets.urban100 import Urban100


class DatasetSelector:
    def __init__(self, scale_factor: int):
        """
        Initialize dataset selector map.
        Load the dataset class here and store them into dictionary.
        :param scale_factor: Scale factor for the dataset
        """
        self.bsds100 = BSDS100(scale_factor)
        self.bsds200 = BSDS200(scale_factor)
        self.bsds300 = BSDS300(scale_factor)
        self.bsds500 = BSDS500(scale_factor)
        self.set5 = Set5(scale_factor)
        self.set14 = Set14(scale_factor)
        self.manga109 = Manga109(scale_factor)
        self.t91 = T91(scale_factor)
        self.urban100 = Urban100(scale_factor)

        # Dataset map
        self.dataset_map = {
            "BSDS100": self.bsds100,
            "BSDS200": self.bsds200,
            "BSDS300": self.bsds300,
            "BSDS500": self.bsds500,
            "Set5": self.set5,
            "Set14": self.set14,
            "Manga109": self.manga109,
            "T91": self.t91,
            "Urban100": self.urban100
        }

    def select_dataset(self, dataset_name: str):
        """
        Select dataset from pre-defined map
        :param dataset_name: Name of dataset
        """
        try:
            return self.dataset_map[dataset_name]
        except KeyError:
            raise DataSetError(f"Selected dataset '{dataset_name}' does not exist")
        except Exception as err:
            raise DataSetError(f"Error while selecting dataset: {err}")
