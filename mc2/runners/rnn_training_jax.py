import argparse
import jax

from mc2.training.jax_routine import train_recursive_nn


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
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Run in debug mode with reduced data")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_recursive_nn(material=args.material, n_epochs=args.epochs, debug=args.debug)


if __name__ == "__main__":
    main()
