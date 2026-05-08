import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from kidney_disease_classifier.models.model_factory import get_preprocessor
from src.kidney_disease_classifier.config.configuration import ConfigurationManager
from src.kidney_disease_classifier.utils.common import get_class_labels

class PredictionPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.config
        self.params = config_manager.params

        # Load model once
        self.model = load_model(self.config.training.trained_model_path)

        # Load preprocessor once
        self.preprocessor = get_preprocessor(self.params.model.name)

        # Optional: class labels (can be moved to config.yaml)
        self.class_labels = get_class_labels(self.config.prepare_base_model.class_labels_path)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and convert image to array"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = image.load_img(image_path, target_size=self.params.model.input_shape[:-1])
        img_array = image.img_to_array(img)
        return img_array

    def _preprocess(self, img_array: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline"""
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocessor(img_array)
        return img_array

    def _predict(self, processed_image: np.ndarray) -> np.ndarray:
        """Run model inference"""
        preds = self.model.predict(processed_image, verbose=0)
        return preds

    def _postprocess(self, preds: np.ndarray) -> dict:
        """Convert model output to human-readable format"""

        preds = np.squeeze(preds)  # remove batch dimension safely

        # Multi-class case
        if preds.ndim > 0 and len(preds) > 1:
            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))

        # Binary case (sigmoid output)
        else:
            prob = float(preds)
            class_idx = 1 if prob > 0.5 else 0
            confidence = prob if class_idx == 1 else 1 - prob

        label = self.class_labels[class_idx]

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