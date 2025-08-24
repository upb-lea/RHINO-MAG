import argparse
import jax

from mc2.training.routine import train_recursive_nn, SUPPORTED_ARCHS

jax.config.update("jax_platform_name", "cpu")


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
    parser.add_argument("-s", "--tbptt_size", default=1024, required=False, type=int, help="TBPTT size")
    parser.add_argument(
        "-j",
        "--n_jobs",
        default=5,
        required=False,
        type=int,
        help="Number of parallel jobs to run. Default is 5.",
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="gru",
        required=False,
        help="Architecture to use. Must be one of: " + ", ".join(SUPPORTED_ARCHS),
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_recursive_nn(
        material=args.material,
        n_epochs=args.epochs,
        debug=args.debug,
        n_jobs=args.n_jobs,
        model_arch=args.arch,
        tbptt_size=args.tbptt_size,
    )


if __name__ == "__main__":
    main()
