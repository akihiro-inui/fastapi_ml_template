from fastapi import APIRouter
from fastapi.responses import JSONResponse
from api.src.utils.custom_logger import logger

router = APIRouter()


@router.get("/process")
def get_process() -> JSONResponse:
    """
    Run some process here
    :return: Up to you
    """
    try:
        # Implement some logic here
        return JSONResponse(status_code=200,
                            content={"data": {},
                                     "message": "Service is healthy"})
    except Exception as err:
        # Implement some logic here
        logger.error(f"Process failed: {err}")
        return JSONResponse(status_code=500,
                            content={"data": {},
                                     "message": "Service is not healthy"})


