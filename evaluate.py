import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from input_pipeline import get_dataset
from tools import CLASS_NAMES
from load_model import get_model

def print_accuracy(result):
    print("Accuracy: ")
    for i, acc in enumerate(result[5:]):
        print(f"{CLASS_NAMES[i]}: {acc*100}")

def test(dataset, model, name, out_dir):
    test_results = model.evaluate(dataset)
    np.save(f"{str(out_dir)}/{name}", np.array(test_results))

    print(f"Evaluating for model: {name}.")
    print_accuracy(test_results)

def evaluate(model_path, name, numpy_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model = load_model(model_path)

    test_data = get_dataset("Test", numpy_dir)

    test(test_data, model, name, odir1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model')
    grp = parser.add_argument_group('Required arguments')
    grp.add_argument("-i", "--model_dir", type=str, required=True,
                     help="Directory which contains models.")
    grp.add_argument("-n", "--numpy_dir", type=str, required=True,
                     help="Directory which contains filepath and label array.")
    grp.add_argument("-o", "--out_dir", type=str, required=True,
                     help="Directory to store result.")
    grp.add_argument("-m", "--name", type=str, required=True,
                     help="Model name.")
    args = parser.parse_args()
    evaluate(args.model_dir, args.numpy_dir, args.out_dir)
