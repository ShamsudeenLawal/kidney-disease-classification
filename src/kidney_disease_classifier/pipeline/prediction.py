import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from kidney_disease_classifier.config.configuration import ConfigurationManager


def select_preprocessor(params):
    model_type = params.MODEL_TYPE.lower()

    preprocessors = {
        "vgg16": tf.keras.applications.vgg16.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
        "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    }

    if model_type not in preprocessors:
        raise ValueError(
            f"Invalid model type: {params.MODEL_TYPE}. "
            "Choose from ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']"
        )

    return preprocessors[model_type]


class PredictionPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.config
        self.params = config_manager.params

        # Load model once
        self.model = load_model(self.config.training.trained_model_path)

        # Load preprocessor once
        self.preprocessor = select_preprocessor(self.params)

        # Optional: class labels (can be moved to config.yaml)
        self.class_labels = {
            0: "Normal",
            1: "Tumor"
        }

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and convert image to array"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = image.load_img(image_path, target_size=self.params.IMAGE_SIZE)
        img_array = image.img_to_array(img)
        return img_array

    def _preprocess(self, img_array: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline"""
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocessor(img_array)
        return img_array

    def _predict(self, processed_image: np.ndarray) -> np.ndarray:
        """Run model inference"""
        preds = self.model.predict(processed_image)
        return preds

    def _postprocess(self, preds: np.ndarray) -> dict:
        """Convert model output to human-readable format"""
        class_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        label = self.class_labels.get(class_idx, "Unknown")

        return {
            "label": label,
            "confidence": round(confidence, 4)
        }

    def predict(self, image_path: str) -> dict:
        """Full prediction pipeline"""
        try:
            img = self._load_image(image_path)
            processed = self._preprocess(img)
            preds = self._predict(processed)
            result = self._postprocess(preds)

            return result

        except Exception as e:
            return {
                "error": str(e)
            }