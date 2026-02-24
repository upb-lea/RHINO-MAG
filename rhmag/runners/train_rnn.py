import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # disable preallocation of memory

from rhmag.runners.rnn_training_jax import train_model_jax


train_model_jax(
    material_name="E",
    model_types=["GRUwLinearModel32"],
    seeds=[
        1,
    ],
    exp_name="demonstration",
    loss_type="adapted_RMS",
    gpu_id=1,
    epochs=1000,
    batch_size=512,
    tbptt_size=156,
    past_size=28,
    time_shift=0,
    noise_on_data=0.0,
    tbptt_size_start=None,
    dyn_avg_kernel_size=11,
    disable_f64=True,
    disable_features=False,
    transform_H=False,
    use_all_data=False,
)
