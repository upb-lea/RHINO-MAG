from mc2.runners.rnn_training_jax import main as train_model_jax

import argparse
from mc2.data_management import AVAILABLE_MATERIALS


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
    material = args.material
    disable_f64 = args.disable_f64
    accuracy_tag = "-f32" if args.disable_f64 else "-f64"

    if args.material == "A":
        epochs = 10_000
    else:
        epochs = 2500

    if args.material == "C":
        past_size = 1
    else:
        past_size = 28

    ## reduced feature set
    train_model_jax(
        material=args.material,
        model_type=["GRU10"],
        seeds=[12, 53, 66],
        exp_name=f"reduced-features{accuracy_tag}",
        loss_type="adapted_RMS",
        gpu_id=args.gpu_id,
        epochs=epochs,
        batch_size=512,
        tbptt_size=156,
        past_size=past_size,
        time_shift=0,
        noise_on_data=0.0,
        tbptt_size_start=None,
        disable_f64=args.disable_f64,
        disable_features="reduce",
        transform_H=False,
        use_all_data=False,
    )

    ## Default setup
    train_model_jax(
        material=args.material,
        model_type=["GRU10"],
        seeds=[12, 53, 66],
        exp_name=f"default{accuracy_tag}",
        loss_type="adapted_RMS",
        gpu_id=args.gpu_id,
        epochs=epochs,
        batch_size=512,
        tbptt_size=156,
        past_size=past_size,
        time_shift=0,
        noise_on_data=0.0,
        tbptt_size_start=None,
        disable_f64=args.disable_f64,
        disable_features=False,
        transform_H=False,
    )

    ## larger kernel
    train_model_jax(
        material=args.material,
        model_type=["GRU10"],
        seeds=[12, 53, 66],
        exp_name=f"larger-kernel{accuracy_tag}",
        loss_type="adapted_RMS",
        gpu_id=args.gpu_id,
        epochs=epochs,
        batch_size=512,
        tbptt_size=156,
        past_size=past_size,
        time_shift=0,
        noise_on_data=0.0,
        tbptt_size_start=None,
        dyn_avg_kernel_size=51,
        disable_f64=args.disable_f64,
        disable_features=False,
        transform_H=False,
        use_all_data=False,
    )

    # ## Transformed
    # train_model_jax(
    #     material=args.material,
    #     model_type=["GRU8"],
    #     seeds=[12, 53, 66, 105, 6],
    #     exp_name=f"transformed{accuracy_tag}",
    #     loss_type="adapted_RMS",
    #     gpu_id=args.gpu_id,
    #     epochs=epochs,
    #     batch_size=512,
    #     tbptt_size=156,
    #     past_size=past_size,
    #     time_shift=0,
    #     noise_on_data=0.0,
    #     tbptt_size_start=None,
    #     disable_f64=args.disable_f64,
    #     disable_features=False,
    #     transform_H=True,
    # )

    # ## incl shift
    # train_model_jax(
    #     material=args.material,
    #     model_type=["GRU8"],
    #     seeds=[12, 53, 66, 105, 6],
    #     exp_name=f"shift{accuracy_tag}",
    #     loss_type="adapted_RMS",
    #     gpu_id=args.gpu_id,
    #     epochs=epochs,
    #     batch_size=512,
    #     tbptt_size=156,
    #     past_size=past_size,
    #     time_shift=5,
    #     noise_on_data=0.0,
    #     tbptt_size_start=None,
    #     disable_f64=args.disable_f64,
    #     disable_features=False,
    #     transform_H=True,
    # )

    # ## incl shift
    # train_model_jax(
    #     material=args.material,
    #     model_type=["GRU8"],
    #     seeds=[12, 53, 66, 105, 6],
    #     exp_name=f"long{accuracy_tag}",
    #     loss_type="adapted_RMS",
    #     gpu_id=args.gpu_id,
    #     epochs=epochs * 2,
    #     batch_size=512,
    #     tbptt_size=156,
    #     past_size=past_size,
    #     time_shift=5,
    #     noise_on_data=0.0,
    #     tbptt_size_start=None,
    #     disable_f64=args.disable_f64,
    #     disable_features=False,
    #     transform_H=True,
    # )
