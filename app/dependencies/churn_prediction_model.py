import mlflow
import polars
import os

from churn_prediction_model.feature_extraction import build_feature_columns
from app.payloads import Payload


class ChurnPredictionInference:
    def __init__(
            self,
            model_path: str
    ):
        self.model = mlflow.sklearn.load_model(model_path)

    def __call__(self, payload: Payload):
        input_df = polars.DataFrame(
            payload.dict(),
            schema_overrides=dict(
                USER_ID=polars.Int64,
                REGISTRATION_AT=polars.Utf8,
                TOTAL_VISIT_COUNT=polars.Int64,
                LAST_VISIT_AT=polars.Utf8,
                TOTAL_POST_COUNT=polars.Int64,
                LAST_POST_AT=polars.Utf8,
                TOTAL_LIKES_RECEIVED=polars.Int64,
                LAST_LIKE_RECEIVED_AT=polars.Utf8,
                TOTAL_COMMENTS_RECEIVED=polars.Int64,
                LAST_COMMENT_RECEIVED_AT=polars.Utf8,
                TOTAL_LIKES_GIVEN=polars.Int64,
                LAST_LIKE_GIVEN_AT=polars.Utf8,
                TOTAL_COMMENTS_WRITTEN=polars.Int64,
                LAST_COMMENT_WRITTEN_AT=polars.Utf8,
                CHURNED=polars.Boolean
            )
        )
        input_df = build_feature_columns(input_df)
        input_df = input_df.drop("label")
        return self.model.predict_proba(input_df)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

churn_prediction_inference = ChurnPredictionInference(
    os.path.join(DATA_DIR, "churn_prediction_clf")
)


def get_churn_prediction_inference():
    return churn_prediction_inference
