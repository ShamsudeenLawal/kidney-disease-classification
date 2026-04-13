
import os
import tensorflow as tf
from pathlib import Path
from kidney_disease_classifier.entity.config_entity import TrainingConfig


def select_preprocessor(config):
    model_type = config.params_model_type.lower()

    preprocessors = {
        "vgg16": tf.keras.applications.vgg16.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
        "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    }

    if model_type not in preprocessors:
        raise ValueError(
            "Invalid model type. Choose from: ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']"
        )

    return preprocessors[model_type]


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def load_datasets(self):
        # Resolve dataset directory
        data_dir = os.path.join(
            self.config.training_data,
            os.listdir(self.config.training_data)[0]
        )

        image_size = self.config.params_image_size[:-1]
        batch_size = self.config.params_batch_size

        # Load datasets
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )

        self.valid_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )

    def build_pipeline(self):
        preprocessor = select_preprocessor(self.config)

        # Data augmentation (ONLY for training)
        if self.config.params_is_augmentation:
            self.data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
            ])
        else:
            self.data_augmentation = tf.keras.Sequential([])

        # Apply pipeline
        self.train_ds = self.train_ds.map(
            lambda x, y: (preprocessor(self.data_augmentation(x, training=True)), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        self.valid_ds = self.valid_ds.map(
            lambda x, y: (preprocessor(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Performance optimization
        self.train_ds = self.train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.valid_ds = self.valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
            ]
        self.model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=self.config.params_epochs,
            callbacks=callbacks
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )