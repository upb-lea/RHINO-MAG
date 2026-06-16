from copy import deepcopy

from rhmag.runners.rnn_training_jax import train_model_jax

import argparse
from rhmag.data_management import AVAILABLE_MATERIALS

if __name__ == "__main__":

    material_names_wo_A = deepcopy(AVAILABLE_MATERIALS)
    material_names_wo_A.remove("A")

    train_model_jax(
        material_names=material_names_wo_A,
        model_types=["GRU8"],
        seeds=[49],
        exp_name=f"MagNetHub-reduced-features-f32",
        loss_type="adapted_RMS",
        gpu_id=0,
        epochs=1500,
        batch_size=512,
        tbptt_size=156,
        past_size=28,
        time_shift=0,
        noise_on_data=0.0,
        tbptt_size_start=None,
        dyn_avg_kernel_size=11,
        disable_f64=True,
        disable_features="reduce",
        transform_H=False,
        use_all_data=True,
    )
