from fastapi import APIRouter
from fastapi.responses import RedirectResponse
router = APIRouter()


@router.get("/")
def get_docs() -> RedirectResponse:
    """
    Get Swagger document. FastAPI creates /docs Swagger document by default
    :return: Redirect to Swagger doc endpoint
    """
    return RedirectResponse(url="/docs/")
