import os
import sys
import mlflow
import tensorflow as tf
from pathlib import Path

from kidney_disease_classifier import logger
from kidney_disease_classifier.models.model_factory import get_preprocessor
from kidney_disease_classifier.utils.mlflow_utils import set_or_create_experiment
from kidney_disease_classifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        optimizer = self._get_optimizer()

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        logger.info("Model loaded and compiled")


    def _get_optimizer(self):
        lr = self.config.learning_rate

        if self.config.optimizer.lower() == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)

        elif self.config.optimizer.lower() == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=lr)

        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
    def load_datasets(self):
        image_size = self.config.input_shape[:-1]

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.train_dir,
            image_size=image_size,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=42
        )

        self.valid_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.val_dir,
            image_size=image_size,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        logger.info("Datasets loaded")

    def build_pipeline(self):
        preprocessor = get_preprocessor(model_name=self.config.model_name)

        aug_cfg = self.config.augmentation

        if aug_cfg["enabled"]:
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal") if aug_cfg["horizontal_flip"] else tf.keras.layers.Lambda(lambda x: x),
                tf.keras.layers.RandomRotation(aug_cfg["rotation"]),
                tf.keras.layers.RandomZoom(aug_cfg["zoom"]),
                tf.keras.layers.RandomTranslation(
                    aug_cfg["translation"], aug_cfg["translation"]
                ),
            ])
        else:
            augmentation = None

        def train_map(x, y):
            if augmentation:
                x = augmentation(x, training=True)
            x = preprocessor(x)
            return x, y

        def val_map(x, y):
            return preprocessor(x), y

        self.train_ds = self.train_ds.map(train_map, num_parallel_calls=tf.data.AUTOTUNE)
        self.valid_ds = self.valid_ds.map(val_map, num_parallel_calls=tf.data.AUTOTUNE)

        self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)
        self.valid_ds = self.valid_ds.prefetch(tf.data.AUTOTUNE)

        logger.info("Pipeline built")

    def _get_callbacks(self):
        cb_cfg = self.config.callbacks

        callbacks = []

        if "early_stopping" in cb_cfg:
            es = cb_cfg["early_stopping"]
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=es["monitor"],
                    patience=es["patience"],
                    restore_best_weights=es["restore_best_weights"]
                )
            )

        return callbacks
    

    def train(self):
        set_or_create_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name=f"{self.config.model_name}_training") as run:

            mlflow.log_params({
                "model": self.config.model_name,
                "input_shape": self.config.input_shape,
                "weights": self.config.weights,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "lr": self.config.learning_rate,
                "optimizer": self.config.optimizer,
            })

            # logging dictionary data
            mlflow.log_params(self.config.augmentation)
            mlflow.log_params(self.config.transfer_learning)
            mlflow.log_params(self.config.callbacks)


            history = self.model.fit(
                self.train_ds,
                validation_data=self.valid_ds,
                epochs=self.config.epochs,
                callbacks=self._get_callbacks()
            )

            train_loss, train_acc = self.model.evaluate(self.train_ds, verbose=0)
            val_loss, val_acc = self.model.evaluate(self.valid_ds, verbose=0)

            mlflow.log_metrics({
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            # log full training history (VERY USEFUL)
            for epoch, acc in enumerate(history.history["accuracy"]):
                mlflow.log_metric("epoch_train_accuracy", acc, step=epoch)

            for epoch, val_acc in enumerate(history.history["val_accuracy"]):
                mlflow.log_metric("epoch_val_accuracy", val_acc, step=epoch)

            # ---------------- SAVE MODEL ----------------
            mlflow.keras.log_model(self.model, "model")

            self._save_model(self.config.trained_model_path, self.model)

            os.makedirs(os.path.dirname(self.config.run_id_file), exist_ok=True)
            self.config.run_id_file.write_text(run.info.run_id)

            logger.info("Training completed successfully")

    @staticmethod
    def _save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

    def run(self):
        logger.info("Starting training pipeline...")

        self.get_base_model()
        self.load_datasets()
        self.build_pipeline()
        self.train()

        logger.info("Training finished successfully")

