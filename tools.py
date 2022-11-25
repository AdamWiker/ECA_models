
import tensorflow as tf

AVAILABLE_MODELS = [
    "mobilenetv3"
]

CLASS_NAMES = [
    'Boredom',
    'Engagement',
    'Confusion',
    'Frustration'
]

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def get_file_name(model_name, fullyconnected=False, finetune=False, reg=False):
    file_name = base_model_name
    if finetune:
        file_name += "_finetune"
    if fullyconnected:
        file_name += "_fc"
    if reg:
        file_name += "_reg"
    file_name += ".h5"

    return file_name
