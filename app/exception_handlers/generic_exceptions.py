import traceback

from fastapi import status, Request, HTTPException
from fastapi.responses import JSONResponse


async def handle_internal_error(request: Request, exc: HTTPException):
    """
    Generates more verbose error messages when crashing, which includes the stack trace.

    :param request: Initial request that caused the crash
    :param exc: Exception raised in FastAPI app
    :return: JSON response containing the stack trace and exception raised
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": str(exc),
            "trace": traceback.format_exc().split("\n")
        })