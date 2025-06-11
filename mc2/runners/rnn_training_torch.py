import torch
import argparse

from mc2.data_management import load_data_into_pandas_df
from mc2.training.routine import train_recursive_nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recursive NNs")
    parser.add_argument("-t", "--tag", default=None, required=False, help="an identifier/tag/comment for the trials")
    parser.add_argument(
        "-m",
        "--material",
        default=None,
        required=False,
        help="Material label to train on. Leave blank for all materials",
    )
    parser.add_argument("-e", "--epochs", default=100, required=False, type=int, help="Number of epochs to train")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_recursive_nn(material=args.material, n_epochs=args.epochs)


if __name__ == "__main__":
    main()
