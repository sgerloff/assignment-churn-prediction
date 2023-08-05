FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./app /app/app/
COPY requirements_app.txt requirements_app.txt

COPY ./churn_prediction_model /app/churn_prediction_model
COPY requirements_module.txt requirements_module.txt

RUN apt-get update && apt-get install -y \
    python3 python3-pip

RUN pip install --upgrade pip && pip install -r requirements_app.txt && pip install -r requirements_module.txt