from mc2.runners.rnn_training_jax import train_model_jax
from mc2.data_management import AVAILABLE_MATERIALS
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recursive NNs")
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

    for material_name in AVAILABLE_MATERIALS[:-5]:  # dont train for ABCDE

        train_model_jax(
            material_name=material_name,
            model_types=["GRU8"],
            seeds=[12],
            exp_name=f"reduced-features{accuracy_tag}",
            loss_type="adapted_RMS",
            gpu_id=args.gpu_id,
            epochs=1000,
            batch_size=512,
            tbptt_size=156,
            past_size=28,
            time_shift=0,
            noise_on_data=0.0,
            tbptt_size_start=None,
            dyn_avg_kernel_size=11,
            disable_f64=args.disable_f64,
            disable_features="reduce",
            transform_H=False,
            use_all_data=False,
        )