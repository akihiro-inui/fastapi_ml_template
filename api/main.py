import os
import sys
sys.path.insert(0, os.getcwd())

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from common_tools.src.custom_logger import logger
from common_tools.src.config_loader import load_env_file
from common_tools.src.custom_error_handlers import PydanticError, BaseSystemError

# Try to load environment variables from file
load_env_file('.env')

from api.src.endpoints import status, docs, process


# Create API Application
app = FastAPI()


@app.exception_handler(PydanticError)
async def validation_exception_handler(request, err) -> JSONResponse:
    """
    Pydantic model exceptions will be caught here
    :param request: Request to endpoint
    :param err: Error message caught in lower level
    :return: JSONResponse with error message
    """
    base_error_message = f"Failed to execute: {request.method}: {request.url}"
    logger.error(base_error_message)
    return JSONResponse(status_code=400,
                        content={"message": f"{base_error_message}: Error in data format", "detail": f"{err}"})


@app.exception_handler(BaseSystemError)
async def unknown_exception_handler(request, err):
    """
    All non-defined exceptions will be caught here
    :param request: Request to endpoint
    :param err: Error message caught in lower level
    :return: JSONResponse with error message
    """
    base_error_message = f"Unknown error. Failed to execute: {request.method}: {request.url}"
    logger.error(base_error_message)
    return JSONResponse(status_code=400,
                        content={"message": f"{base_error_message}", "detail": f"{err}"})


# Add endpoints
app.include_router(docs.router)
app.include_router(status.router)
app.include_router(process.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=int(os.environ.get("API_PORT")),
                log_level=os.environ.get("LOG_LEVEL"),
                reload=True)
