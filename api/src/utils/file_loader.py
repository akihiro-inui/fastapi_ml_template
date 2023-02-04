import os

import PIL.Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from ml.src.models.model_selector import ModelSelector
from ml.src.models.model_wrapper import ModelWrapper

from typing import ByteString


def load_model():
    """
    Load pre-trained model from disk.
    :return: Loaded model
    """
    # Initialize model
    model_selector = ModelSelector(int(os.environ.get("IMAGE_SCALE_FACTOR")))
    model = model_selector.select_model(os.environ.get("MODEL_NAME"))
    model_wrapper = ModelWrapper(model)
    return model_wrapper.load_model(os.environ.get("MODEL_FILE_PATH"))


def load_image(image_size_after_resize: int, pil_image: PIL.Image.Image, with_gpu: bool = False) -> torch.Tensor:
    """
    Load image from disk and return a tensor.
    :param image_size_after_resize: Size of image after resizing.
    :param pil_image: Image file content as PIL.Image.Image
    :return: Image file content as tensor.
    """
    try:
        resize = transforms.Resize([image_size_after_resize, image_size_after_resize])
        img = resize(pil_image)
        to_tensor = transforms.ToTensor()

        # apply transformation and convert to Pytorch tensor
        tensor = to_tensor(img)

        # add another dimension at the front to get NCHW shape
        tensor = tensor.unsqueeze(0)
        return tensor.cuda() if with_gpu else tensor

    except Exception as err:
        raise Exception(f"Failed to load image file: {err}")
