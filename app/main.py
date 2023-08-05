import uvicorn
import logging

from fastapi import FastAPI, Response, status, Depends
from fastapi.responses import JSONResponse

from app.payloads import Payload
from app.exception_handlers import handle_internal_error
from app.dependencies.churn_prediction_model import get_churn_prediction_inference

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
                     model=Depends(get_churn_prediction_inference)):

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"churn_probability": model(item)[0][1]}
    )


@app.get('/health')
def check_health():
    """Check if the dependent service are healthy"""
    return Response(status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
