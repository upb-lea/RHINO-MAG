import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # disable preallocation of memory

from rhmag.runners.rnn_training_jax import train_model_jax


train_model_jax(
    material_names=["X"],
    model_types=["GRU8", "GRU16", "GRU32"],
    seeds=[
        42,
    ],
    exp_name="pretraining",
    loss_type="adapted_RMS",
    gpu_id=0,
    epochs=1000,
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
