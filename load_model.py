from pathlib import Path

from tensorflow.keras.applications import Xception, MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

from tools import get_file_name

def get_base_model(model_name, img_width, img_height, weight_dir=None):
    base_model = None
    if model_name == 'mobilenetv3':
        base_model = MobileNetV3Small(
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights='imagenet', # pretrained on imagenet
            input_tensor=None,
            pooling=None,
            dropout_rate=0.2,
            input_shape=(img_width, img_height, 3),
            include_preprocessing=True
        )
    elif model_name == 'xception':
        weights = Path(weight_dir)
        xception = "xception_weights_tf_dim_ordering_tf_kernels_notop.h5" # weights pretrained on imagenet
        weights = weights / xception
        base_model = Xception(weights=str(weights),
                              include_top=False,
                              input_shape=(img_width, img_height, 3))
    else:
        raise Exception('Requested base model does not exist.')

    return base_model

def get_model(weight_dir, out_dir, base_model_name, fullyconnected=False, finetune=False, reg=False):
    num_out = 4
    if reg:
        num_out = 1

    if finetune:
        model_name = get_file_name(base_model_name, fullyconnected=fullyconnected)

        base_model = load_model(str(out_dir) + model_name)

        base_model.trainable = True
        for layer in base_model.layers[:finetune_at_layer]:
            layer.trainable = False

        return base_model

    else:
        base_model = get_base_model(base_model_name, img_width, img_height)

        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        if fullyconnected:
            x = Dense(128, activation="relu", name="fc1")(x)
            x = Dense(64, activation="relu", name="fc2")(x)
        boredom = Dense(num_out, name="y1")(x)
        engagement = Dense(num_out, name="y2")(x)
        confusion = Dense(num_out, name="y3")(x)
        frustration = Dense(num_out, name="y4")(x)
        model = Model(inputs=base_model.input,
                      outputs=[boredom, engagement, confusion, frustration])
    return model
