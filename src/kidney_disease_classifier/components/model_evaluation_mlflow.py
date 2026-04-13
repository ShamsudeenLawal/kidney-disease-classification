import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from kidney_disease_classifier.entity.config_entity import EvaluationConfig
from kidney_disease_classifier.utils.common import read_yaml, create_directories, save_json

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


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def load_datasets(self):
        # Resolve dataset directory
        data_dir = os.path.join(
            self.config.training_data,
            os.listdir(self.config.training_data)[0]
        )

        image_size = self.config.params_image_size[:-1]
        batch_size = self.config.params_batch_size

        # Load validation datasets
        self.valid_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )

        preprocessor = select_preprocessor(self.config)
        self.valid_ds = self.valid_ds.map(
            lambda x, y: (preprocessor(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Performance optimization
        self.valid_ds = self.valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluate(self):
        self.model = self.load_model(self.config.path_of_model)
        self.load_datasets()
        self.score = self.model.evaluate(self.valid_ds)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                registered_model_name = self.config.params_model_type.upper()
                mlflow.keras.log_model(self.model, "model", registered_model_name=registered_model_name)
            else:
                mlflow.keras.log_model(self.model, "model")
