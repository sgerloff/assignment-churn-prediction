import uvicorn

from fastapi import FastAPI, Response, Request, status
from fastapi.responses import JSONResponse

from app.payloads import Payload
from app.exception_handlers import handle_internal_error

import logging

logging.basicConfig(level=logging.INFO)


def create_app():
    return FastAPI(
        exception_handlers={
            status.HTTP_500_INTERNAL_SERVER_ERROR: handle_internal_error
        }
    )


app = create_app()


@app.post("/v1/inference")
def handle_inference(item: Payload,
                     response: Response,
                     request: Request,
                     ):
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success_status": True}
    )


@app.get('/health')
def check_health():
    """Check if the dependent service are healthy"""
    return Response(status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
