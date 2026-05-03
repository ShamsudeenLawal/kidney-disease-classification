import os
import sys
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from kidney_disease_classifier.models.model_factory import get_model
from kidney_disease_classifier.entity.config_entity import PrepareBaseModelConfig
from kidney_disease_classifier import logger


# ---------------------- PREPARE BASE MODEL ----------------------
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    # ---------------------- LOAD BASE MODEL ----------------------
    def get_base_model(self):
        self.model = get_model(
                    model_name=self.config.model_type,
                    input_shape=self.config.input_shape,
                    weights=self.config.weights
                )
        self._save_model(self.config.base_model_path, self.model)

    # ---------------------- BUILD FULL MODEL ----------------------
    def _prepare_full_model(
        self,
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool = True,
        freeze_till: int = 10,
        learning_rate: float = 0.001
    ) -> tf.keras.Model:
        """
        Adds classification head + handles transfer learning setup
        """

        # -------- FREEZE LAYERS --------
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        elif freeze_till:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # -------- CLASSIFICATION HEAD --------
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(128, activation="relu")(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=outputs
        )

        # -------- COMPILE --------
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        logger.info("Full model built successfully")
        full_model.summary()

        return full_model

    # ---------------------- UPDATE MODEL ----------------------
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.num_classes,
            freeze_all=self.config.freeze_all,
            freeze_till=self.config.freeze_till,
            learning_rate=self.config.learning_rate
        )

        self._save_model(self.config.updated_base_model_path, self.full_model)

    # ---------------------- SAVE MODEL ----------------------
    @staticmethod
    def _save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info(f"Model saved at {path}")

    def run(self):
        """
        Executes the full base model preparation pipeline
        """
        try:
            logger.info("Starting base model preparation...")

            self.get_base_model()
            logger.info("Base model loaded and saved.")

            self.update_base_model()
            logger.info("Full model updated and saved.")

            logger.info("Base model preparation completed successfully.")

        except Exception as e:
            
            raise Exception(e, sys)
            # raise CustomException(e, sys)
