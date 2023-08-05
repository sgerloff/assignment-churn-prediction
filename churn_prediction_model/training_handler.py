import polars
import functools
import logging
import random
import omegaconf
import mlflow
import pandas
import os
import hydra

from typing import Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn import metrics


class TrainingHandler:
    def __init__(
            self,
            training_data: polars.DataFrame,
            base_classifier: omegaconf.DictConfig,
            hyperparameter: Optional[dict] = None,
            cv_parameter: Optional[dict] = None,
            n_iter: Optional[int] = None,
            random_seed: int = 42
    ):
        self.logger = logging.getLogger("trainer")

        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.training_data = training_data
        self.base_classifier = base_classifier

        self.hyperparameter = hyperparameter
        if not self.hyperparameter:
            self.hyperparameter = dict()

        self.cv_parameter = cv_parameter
        if not self.cv_parameter:
            self.cv_parameter = dict()

        self.static_parameters, self.search_space = None, None
        self.n_iter = n_iter
        self.mlflow_tags = dict()
        self.hyperparameter_search_cv = self._build_hyperparameter_search_cv()

        self.train_labels, self.test_labels = None, None
        self.train_features, self.test_features = None, None

        self.display_collection = dict(
            roc_curve=metrics.RocCurveDisplay.from_estimator,
            pr_curve=metrics.PrecisionRecallDisplay.from_estimator,
            confusion_matrix=metrics.ConfusionMatrixDisplay.from_estimator,
            calibration_curve=functools.partial(CalibrationDisplay.from_estimator, n_bins=20)
        )

    def _build_hyperparameter_search_cv(self):
        self.search_space = dict()
        self.static_parameters = dict()
        for key, value in self.hyperparameter.items():
            if isinstance(value, omegaconf.ListConfig):
                self.search_space[key] = list(value)
            else:
                self.static_parameters[key] = value

        if isinstance(self.n_iter, int):
            parameter_search_cv = RandomizedSearchCV(
                hydra.utils.instantiate(self.base_classifier, **self.static_parameters),
                self.search_space,
                **self.cv_parameter,
                n_iter=self.n_iter
            )
            self.mlflow_tags["_search_cv"] = "RandomizedSearchCV"
            self.mlflow_tags["_estimator"] = self.base_classifier._target_
        else:
            parameter_search_cv = GridSearchCV(
                hydra.utils.instantiate(self.base_classifier, **self.static_parameters),
                self.search_space,
                **self.cv_parameter
            )
            self.mlflow_tags["_search_cv"] = "GridSearchCV"
            self.mlflow_tags["_estimator"] = self.base_classifier._target_
        return parameter_search_cv

    def train(self):
        self.logger.info("Build train and test data...")
        self._build_train_and_test_data()

        self.logger.info("Setup mlflow...")
        self._log_run()
        self.logger.info("Find best hyperparameters ...")
        self.hyperparameter_search_cv.fit(self.train_features, self.train_labels)
        self._log_hyperparameter_search()
        self._log_best_parameters()

        self.logger.info("Fit final model on whole data...")
        model_pipeline = self._fit_final_model_pipelines()
        mlflow.sklearn.log_model(artifact_path="move_probability_clf", sk_model=model_pipeline)

    def _build_train_and_test_data(self):
        train_df, test_df = train_test_split(
            self.training_data,
            test_size=0.2,
            shuffle=True,
            random_state=self.random_seed
        )

        self.train_features = train_df.drop("label")
        self.train_labels = train_df["label"]

        self.test_features = test_df.drop("label")
        self.test_labels = test_df["label"]

    def _log_run(self):
        mlflow.set_tags(self.mlflow_tags)
        self._log_trainer_parameter()
        self._log_search_space_as_parameter()

    def _log_trainer_parameter(self):
        mlflow.log_params(dict(
            trainer_random_seed=self.random_seed,
            trainer_n_iter=self.n_iter
        ))

    def _log_search_space_as_parameter(self):
        mlflow.log_params(
            {"search_space_" + key: list(value) for key, value in self.search_space.items()}
        )

    def _log_best_parameters(self):
        mlflow.log_params(self.hyperparameter_search_cv.best_params_)
        if self.static_parameters:
            mlflow.log_params(self.static_parameters)

    def _log_hyperparameter_search(self):
        csv_file = "/tmp/cv_results.csv"
        pandas.DataFrame(self.hyperparameter_search_cv.cv_results_).to_csv(csv_file)
        mlflow.log_artifact(csv_file, artifact_path="hyperparameter_search")

        mlflow.log_dict(self.hyperparameter_search_cv.best_params_,
                        artifact_file="hyperparameter_search/best_params.yaml")

        self._log_test_predictions(self.hyperparameter_search_cv, path="hyperparameter_search")
        self._log_test_metrics(self.hyperparameter_search_cv, path="hyperparameter_search")
        self._log_train_metrics(self.hyperparameter_search_cv, path="hyperparameter_search")

    def _log_test_metrics(self, estimator, path: str = ""):
        mlflow.log_text(
            metrics.classification_report(
                self.test_labels,
                estimator.predict(self.test_features)
            ),
            os.path.join(path, "test_clf_report.txt")
        )

        for name, display in self.display_collection.items():
            plot = display(estimator, self.test_features, self.test_labels)
            mlflow.log_figure(plot.figure_, os.path.join(path, f"test_{name}.png"))

    def _log_train_metrics(self, estimator, path: str = ""):
        mlflow.log_text(
            metrics.classification_report(
                self.train_labels,
                estimator.predict(self.train_features)
            ),
            os.path.join(path, "train_clf_report.txt")
        )

        for name, display in self.display_collection.items():
            plot = display(estimator, self.train_features, self.train_labels)
            mlflow.log_figure(plot.figure_, os.path.join(path, f"train_{name}.png"))

    def _log_test_classification_report(self, estimator):
        report_dict = metrics.classification_report(
            self.test_labels,
            estimator.predict(self.test_features),
            output_dict=True
        )
        flat_report_dict = dict()
        for key, value in report_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_report_dict["_".join([sub_key, key])] = sub_value
            else:
                flat_report_dict[key] = value
        mlflow.log_metrics(flat_report_dict)

    def _log_test_predictions(self, estimator, path: str = ""):
        prediction_data = dict(
            pred_proba=[_pred[1] for _pred in estimator.predict_proba(self.test_features)],
            true=self.test_labels
        )
        csv_file = "/tmp/test_pred_proba.csv"
        pandas.DataFrame(prediction_data).to_csv(csv_file)
        mlflow.log_artifact(csv_file, artifact_path=path)

    def _fit_final_model_pipelines(self):
        model = hydra.utils.instantiate(self.base_classifier, **self.static_parameters,
                                        **self.hyperparameter_search_cv.best_params_)

        model.fit(
            polars.concat([self.train_features, self.test_features]),
            polars.concat([self.train_labels, self.test_labels])
        )
        return model
