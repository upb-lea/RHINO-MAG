from rhmag.runners.rnn_training_jax import train_model_jax

import argparse
import jax

from rhmag.data_management import AVAILABLE_MATERIALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recursive NNs")
    # parser.add_argument("-t", "--tag", default=None, required=False, help="an identifier/tag/comment for the trials")
    parser.add_argument(
        "--material",
        required=True,
        help=f"Material label to train on. One of {AVAILABLE_MATERIALS}",
    )
    parser.add_argument(
        "--gpu_id",
        default=-1,
        type=int,
        required=False,
        help="id of the gpu to use for the experiments. '-1' for using the CPU.",
    )
    parser.add_argument("--disable_f64", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    accuracy_tag = "-f32" if args.disable_f64 else "-f64"

    if args.gpu_id != -1:
        gpus = jax.devices()
        default_device = gpus[args.gpu_id]
    elif args.gpu_id == -1:
        default_device = jax.devices("cpu")[0]

    if args.material == "A":
        epochs = 25_000
        model_types = ["GRU32", "GRU48", "GRU64"]
        dyn_avg_kernel_size = 11
        past_size = 28
    elif args.material == "B":
        epochs = 1500
        model_types = ["GRU2", "GRU4", "GRU6", "GRU8", "GRU10", "GRU12", "GRU16", "GRU24", "GRU32", "GRU48", "GRU64"]
        dyn_avg_kernel_size = 11
        past_size = 28
    elif args.material == "C":
        epochs = 1500
        model_types = ["GRU2", "GRU4", "GRU6", "GRU8", "GRU10", "GRU12", "GRU16", "GRU24", "GRU32", "GRU48", "GRU64"]
        dyn_avg_kernel_size = 11
        past_size = 1
    elif args.material == "D":
        epochs = 1500
        model_types = ["GRU2", "GRU4", "GRU6", "GRU8", "GRU10", "GRU12", "GRU16", "GRU24", "GRU32", "GRU48", "GRU64"]
        dyn_avg_kernel_size = 11
        past_size = 28
    elif args.material == "E":
        epochs = 2500
        model_types = ["GRU2", "GRU4", "GRU6", "GRU8", "GRU10", "GRU12", "GRU16", "GRU24", "GRU32", "GRU48", "GRU64"]
        dyn_avg_kernel_size = 11
        past_size = 28
    else:
        raise ValueError(f"Material '{args.material} is unknown.")

    with jax.default_device(default_device):
        train_model_jax(
            material_name=args.material,
            model_types=model_types,
            seeds=[201, 202, 345, 567, 899],
            exp_name=f"pareto-front{accuracy_tag}",
            loss_type="adapted_RMS",
            gpu_id=args.gpu_id,
            epochs=epochs,
            batch_size=512,
            tbptt_size=156,
            past_size=past_size,
            time_shift=0,
            noise_on_data=0.0,
            tbptt_size_start=None,
            dyn_avg_kernel_size=dyn_avg_kernel_size,
            disable_f64=args.disable_f64,
            disable_features="reduce",
            transform_H=False,
            use_all_data=True,
        )
