import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from kidney_disease_classifier.entity.config_entity import PrepareBaseModelConfig

def select_model(config):
    param_size = config.params_image_size
    weights=config.params_weights
    model_type = config.params_model_type

    if model_type.lower() == "vgg16":
        model = tf.keras.applications.VGG16(
            input_shape=param_size,
            weights=weights,
            include_top=False
        )
    elif model_type.lower() == "resnet50":
        model = tf.keras.applications.ResNet50(
            input_shape=param_size,
            weights=weights,
            include_top=False
        )
    elif model_type.lower() == "mobilenetv2":
        model = tf.keras.applications.MobileNetV2(
            input_shape=param_size,
            weights=weights,
            include_top=False
        )
    elif model_type.lower() == "efficientnetb0":
        model = tf.keras.applications.EfficientNetB0(
            input_shape=param_size,
            weights=weights,
            include_top=False
        )
    else:
        raise Exception("You can only choose from these models: ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']")
    
    return model

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = select_model(self.config)
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
