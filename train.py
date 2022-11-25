import argparse
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from focal_loss import SparseCategoricalFocalLoss

from data_processing import get_dataset, img_width, img_height
from tools import get_file_name, focal_loss
from load_model import get_model

finetune_at_layer = 116
base_learning_rate = 0.0001
epochs = 1
ft_epochs = 1

def train(weight_dir, numpy_dir, out_dir, base_model_name, fullyconnected=False, finetune=False, reg=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    save_name = get_file_name(base_model_name, fullyconnected=fullyconnected, finetune=finetune, reg=reg)

    train_ds = get_dataset("Train", numpy_dir, reg=reg)
    validation_ds = get_dataset("Validation", numpy_dir, reg=reg)
    model = get_model(weight_dir, out_dir, base_model_name, fullyconnected, finetune, reg)

    model_path = str(out_dir) + "/" + save_name

    if finetune:
        lr = base_learning_rate / 10
        finetune_epochs = ft_epochs
    else:
        lr = base_learning_rate
        finetune_epochs = 0

    loss_func = None
    acc_func = None
    if reg:
        loss_func = "mean_squared_error"
        acc_func = "mean_squared_error"
    else:
        loss_func = SparseCategoricalFocalLoss(gamma=2, from_logits=True)
        # loss_func = SparseCategoricalCrossentropy(from_logits=True)
        acc_func = "sparse_categorical_accuracy"

    model.compile(optimizer=RMSprop(learning_rate=lr),
                  loss={"y1": loss_func,
                        "y2": loss_func,
                        "y3": loss_func,
                        "y4": loss_func},
                  metrics={"y1": acc_func,
                           "y2": acc_func,
                           "y3": acc_func,
                           "y4": acc_func})
    print(model.summary())

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-2,
                      patience=2, verbose=1)
    ]

    total_epochs = epochs + finetune_epochs
    history = model.fit(train_ds,
                        epochs=total_epochs,
                        initial_epoch=0,
                        callbacks=callbacks,
                        validation_data=validation_ds)

    print(history.history)
    model.save(save_name)

def run_training(args):
    train(args.weight_dir, args.numpy_dir, args.out_dir, args.model, fullyconnected=True)

    train(args.weight_dir, args.numpy_dir, args.out_dir, args.model, fullyconnected=True, finetune=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get trained model.")
    grp = parser.add_argument_group("Required arguments")
    grp.add_argument("-i", "--weight_dir", type=str, required=True,
                     help="Directory which contains pretrained weights for "
                     "Xception.")
    grp.add_argument("-n", "--numpy_dir", type=str, required=True,
                     help="Directory which contains filepath and label array.")
    grp.add_argument("-o", "--out_dir", type=str, required=True,
                     help="Directory to store trained model and logs.")
    grp.add_argument("-m", "--model", type=str, required=True,
                     help="Name of base model to use when training.")

    args = parser.parse_args()

    run_training(args)
