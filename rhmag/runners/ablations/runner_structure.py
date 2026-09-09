import jax

from rhmag.runners.rnn_training_jax import train_model_jax
from rhmag.data_management import FINAL_MATERIALS


def run_ablation_experiment(gpu_id, tag, loss_function, init_type, feature_type):

    disable_f64 = True

    accuracy_tag = "-f32" if disable_f64 else "-f64"

    if gpu_id != -1:
        gpus = jax.devices()
        default_device = gpus[gpu_id]
    elif gpu_id == -1:
        default_device = jax.devices("cpu")[0]

    for material in FINAL_MATERIALS:

        if material == "A":
            epochs = 10_000
            model_types = ["GRU8"]
            dyn_avg_kernel_size = 11
            past_size = 28
        elif material == "B":
            epochs = 1500
            model_types = ["GRU8"]
            dyn_avg_kernel_size = 11
            past_size = 28
        elif material == "C":
            epochs = 1500
            model_types = ["GRU8"]
            dyn_avg_kernel_size = 11
            past_size = 1
        elif material == "D":
            epochs = 1500
            model_types = ["GRU8"]
            dyn_avg_kernel_size = 11
            past_size = 28
        elif material == "E":
            epochs = 2500
            model_types = ["GRU8"]
            dyn_avg_kernel_size = 11
            past_size = 28
        else:
            raise ValueError(f"Material '{material} is unknown.")

        if init_type == "default":
            pass
        elif init_type == "ignore_warmup":
            past_size = 1
        elif init_type == "force_zero_start_warmup":
            model_types = ["GRUZeroStart8"]
        elif init_type == "force_zero_start_warmup_linear_out":
            model_types = ["GRULinearOut8"]
        elif init_type == "H_as_input":
            model_types = ["GRUwInputH8"]

        if feature_type == "full":
            disable_features = False
        elif feature_type == "reduce":
            disable_features = "reduce"
        elif feature_type == "No_features":
            disable_features = True

        ## Default setup
        with jax.default_device(default_device):
            train_model_jax(
                material_names=[material],
                model_types=model_types,
                seeds=[12, 53, 66, 105, 6],
                exp_name=f"{tag}{accuracy_tag}",
                loss_type=loss_function,
                epochs=epochs,
                batch_size=512,
                tbptt_size=156,
                past_size=past_size,
                time_shift=0,
                noise_on_data=0.0,
                tbptt_size_start=None,
                dyn_avg_kernel_size=dyn_avg_kernel_size,
                disable_f64=disable_f64,
                disable_features=disable_features,
                transform_H=False,
                use_all_data=True,
            )
