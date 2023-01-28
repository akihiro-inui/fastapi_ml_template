from fastapi import APIRouter
from fastapi.responses import JSONResponse
from api.src.utils.custom_logger import logger

router = APIRouter()


@router.get("/status")
def get_status() -> JSONResponse:
    """
    Return application status
    :return: JSONResponse 200 or 500
    """
    try:
        # Implement some custom check here
        return JSONResponse(status_code=200,
                            content={"data": {},
                                     "message": "Service is healthy"})
    except Exception as err:
        logger.error(f"Healthy check failed: {err}")
        return JSONResponse(status_code=500,
                            content={"data": {},
                                     "message": "Service is not healthy"})



