from rhmag.runners.rnn_training_jax import train_model_jax
from rhmag.data_management import FINAL_MATERIALS


def run_ablation_experiment(gpu_id, tag, loss_function, init_type):

    disable_f64 = True

    accuracy_tag = "-f32" if disable_f64 else "-f64"

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

        ## Default setup
        train_model_jax(
            material_names=[material],
            model_types=model_types,
            seeds=[12, 53, 66, 105, 6],
            exp_name=f"{tag}{accuracy_tag}",
            loss_type=loss_function,
            gpu_id=gpu_id,
            epochs=epochs,
            batch_size=512,
            tbptt_size=156,
            past_size=past_size,
            time_shift=0,
            noise_on_data=0.0,
            tbptt_size_start=None,
            dyn_avg_kernel_size=dyn_avg_kernel_size,
            disable_f64=disable_f64,
            disable_features="reduce",
            transform_H=False,
            use_all_data=True,
        )
