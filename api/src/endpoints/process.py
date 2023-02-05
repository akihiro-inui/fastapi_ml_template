import os
import uuid

import torch
from PIL import Image
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms

from common_tools.src.custom_logger import logger

from api.src.utils.file_loader import load_model
from api.src.utils.file_loader import load_image


try:
    model = load_model()
    if model:
        logger.info("Model loaded successfully")
except Exception as err:
    logger.error(f"Error while loading model: {err}")


router = APIRouter()

IMAGEDIR = "./"
image_reshape_size = int(os.environ.get("RESHAPE_IMAGE_SIZE"))

def tensor_to_img(tensor):
    # image=tensor.cpu().clone()
    image = torch.squeeze(tensor)
    image = transforms.ToPILImage()(image)
    return image


@router.post("/process")
def process(file: UploadFile = File(...)) -> JSONResponse:
    """
    Run some process here
    :return: Up to you
    """
    try:
        file.filename = f"{uuid.uuid4()}.jpg"
        pil_image = Image.open(file.file).convert('RGB')
        image_shape = pil_image.size

        contents = load_image(image_reshape_size, pil_image)

        # Inference
        prediction_result = model(contents)

        image_tensor = torch.squeeze(prediction_result)
        resize = transforms.Resize([image_shape[1], image_shape[0]])
        resized_image = resize(image_tensor)
        image = transforms.ToPILImage()(resized_image)
        image.show()

        # TODO: Save image to disk and return file content
        # image.save(f"{IMAGEDIR}/{file.filename}")

        return {"filename": file.filename}

    except Exception as err:
        # Implement some logic here
        logger.error(f"Process failed: {err}")
        return JSONResponse(status_code=500,
                            content={"data": {},
                                     "message": "Failed to process the uploaded image file"})


