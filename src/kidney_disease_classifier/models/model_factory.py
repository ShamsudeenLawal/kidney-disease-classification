import tensorflow as tf

MODEL_REGISTRY = {
    "vgg16": tf.keras.applications.VGG16,
    "resnet50": tf.keras.applications.ResNet50,
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
}


# def get_model(model_name: str, input_shape, weights="imagenet"):
def get_model(model_name, input_shape, weights):
    
    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name](
        input_shape=input_shape,
        weights=weights,
        include_top=False
    )

# ---------------------- PREPROCESSOR ----------------------
def get_preprocessor(model_name):
    
    model_name = model_name.lower()

    preprocessors = {
        "vgg16": tf.keras.applications.vgg16.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
        "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    }

    if model_name not in preprocessors:
        raise ValueError(f"Invalid model type: {model_name}")

    return preprocessors[model_name]