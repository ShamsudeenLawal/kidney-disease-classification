import numpy as np
import tensorflow as tf
import mlflow
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from kidney_disease_classifier.entity.config_entity import EvaluationConfig
from kidney_disease_classifier.utils.common import save_json
from kidney_disease_classifier.models.model_factory import get_preprocessor
from kidney_disease_classifier import logger

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.scores = None

    def load_datasets(self):
        try:
            image_size = self.config.input_shape[:-1]

            self.test_ds = tf.keras.utils.image_dataset_from_directory(
                self.config.test_dir,
                image_size=image_size,
                batch_size=self.config.batch_size,
                shuffle=False
            )

            preprocessor = get_preprocessor(model_name=self.config.model_name)

            self.test_ds = self.test_ds.map(
                lambda x, y: (preprocessor(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)

            logger.info("Test dataset loaded")

        except Exception as e:
            raise Exception(e, sys)


    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.config.model_path)
            logger.info("Model loaded")

        except Exception as e:
            raise Exception(e, sys)


    def evaluate(self, prefix="test"):
        try:
            self.load_model()
            self.load_datasets()

            y_true, y_pred = [], []

            for x, y in self.test_ds:
                preds = self.model.predict(x, verbose=0)

                # -------- HANDLE PREDICTIONS --------
                if preds.shape[-1] > 1:
                    preds = np.argmax(preds, axis=1)
                else:
                    preds = (preds > 0.5).astype(int).ravel()

                # -------- HANDLE LABELS --------
                y = y.numpy()
                if len(y.shape) > 1 and y.shape[-1] > 1:
                    y = np.argmax(y, axis=1)
                else:
                    y = y.ravel()

                y_true.extend(y)
                y_pred.extend(preds)

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # ---------------- METRICS ----------------
            self.scores = self._compute_metrics(y_true, y_pred, prefix=prefix)

            logger.info(f"Evaluation scores: {self.scores}")

            # ---------------- MLFLOW ----------------
            self._log_to_mlflow()

            # ---------------- SAVE ----------------
            self.save_scores()

        except Exception as e:
            raise Exception(e, sys)

    def _compute_metrics(self, y_true, y_pred, prefix="test"):
        metric_list = self.config.metrics

        scores = {}

        if "accuracy" in metric_list:
            scores[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred)

        if "precision" in metric_list:
            scores[f"{prefix}_precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

        if "recall" in metric_list:
            scores[f"{prefix}_recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        if "f1" in metric_list:
            scores[f"{prefix}_f1"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        return scores
    
    def _log_to_mlflow(self):
        try:
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)

            run_id = self.config.run_id_file.read_text().strip()

            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(self.scores)
                mlflow.set_tag("stage", "evaluation")

        except Exception as e:
            raise Exception(e, sys)
        
    def save_scores(self):
        save_json(self.config.scores_path, self.scores)

    def run(self, prefix="test"):
        logger.info("Starting evaluation pipeline...")
        self.evaluate(prefix)
        logger.info("Evaluation completed")