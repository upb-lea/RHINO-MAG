import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pandas as pd
import logging as log
from tqdm import trange
from mc2.data_management import (
    load_data_into_pandas_df,
    MaterialSet,
    FrequencySet,
)
from mc2.training.data_sampling import draw_data_uniformly
from mc2.training.optimization import make_step


DO_TRANSFORM_H = True
H_FACTOR = 1.2
VAL_EVERY = 100


def normalize_material_set(mat_set: MaterialSet, max_d: dict[str, float], reduce_train=False) -> MaterialSet:
    normalized_freq_sets = []
    for freq_set in mat_set.frequency_sets:

        B_norm = freq_set.B / max_d["B"]
        H_norm = freq_set.H / max_d["H"]
        T_norm = freq_set.T / max_d["T"]

        if reduce_train:
            norm_freq_set = FrequencySet(
                material_name=freq_set.material_name,
                frequency=freq_set.frequency,
                B=B_norm[::10],
                H=H_norm[::10],
                T=T_norm[::10],
            )
        else:
            norm_freq_set = FrequencySet(
                material_name=freq_set.material_name,
                frequency=freq_set.frequency,
                B=B_norm,
                H=H_norm,
                T=T_norm,
            )
        normalized_freq_sets.append(norm_freq_set)

    return MaterialSet(
        material_name=mat_set.material_name,
        frequency_sets=normalized_freq_sets,
        frequencies=mat_set.frequencies,
    )


@eqx.filter_value_and_grad
def compute_MSE_loss(
    model: eqx.Module,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
):
    # f: jax.Array,
    pred_H = (model.normalized_call)(B_past, H_past, B_future, T)  # , f
    return jnp.mean((pred_H - H_future) ** 2)


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    # f: jax.Array,
    loss, grads = compute_MSE_loss(model, B_past, H_past, B_future, H_future, T)  # , f
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def process_freq_set(model, opt_state, key, freq_set, optimizer, past_size, tbptt_size, batch_size):
    key, subkey = jax.random.split(key)
    batch_H, batch_B, batch_T, _ = draw_data_uniformly(
        freq_set,
        training_sequence_length=tbptt_size + past_size,
        training_batch_size=batch_size,
        loader_key=subkey,
    )
    # batch_H = batch_H[:, :, 0]
    # batch_B = batch_B[:, :, 0]
    # batch_T = batch_T[:, 0, 0]
    # batch_f = freq_set.frequency

    batch_H_past = batch_H[:, :past_size]
    batch_H_future = batch_H[:, past_size:]
    batch_B_past = batch_B[:, :past_size]
    batch_B_future = batch_B[:, past_size:]
    loss, model, opt_state = make_step(
        model,
        batch_B_past,
        batch_H_past,
        batch_B_future,
        batch_H_future,
        batch_T,
        optimizer,
        opt_state,
    )  # batch_f,

    return loss, model, opt_state, key


def train_step(model, opt_state, train_set_norm, optimizer, key, past_size, tbptt_size, batch_size):

    train_loss = 0.0
    for freq_set in train_set_norm.frequency_sets:
        loss, model, opt_state, key = process_freq_set(
            model, opt_state, key, freq_set, optimizer, past_size, tbptt_size, batch_size
        )
        train_loss += loss / len(train_set_norm.frequencies)

    return train_loss, model, opt_state


@eqx.filter_jit
def process_freq_set_val(freq_set, model, past_size):
    B = freq_set.B
    H = freq_set.H
    T = freq_set.T
    # f = freq_set.frequency
    batch_H_past = H[:, :past_size]
    batch_H_future = H[:, past_size:]
    batch_B_past = B[:, :past_size]
    batch_B_future = B[:, past_size:]

    pred_H = model.normalized_call(batch_B_past, batch_H_past, batch_B_future, T)  # , f
    loss = jnp.mean((pred_H - batch_H_future) ** 2)
    return loss


def val_test(set, model, past_size):

    val_loss = 0.0
    for freq_set in set.frequency_sets:
        loss = process_freq_set_val(freq_set, model, past_size)
        val_loss += loss / len(set.frequencies)

    return val_loss


def train_model(
    model,
    optimizer,
    material_name,
    key: jax.random.PRNGKey,
    seed: int,
    n_steps: int,
    val_every: int,
    tbptt_size: int,
    past_size: int,
    batch_size: int,
):

    data_dict = load_data_into_pandas_df(material=material_name)
    mat_set = MaterialSet.from_pandas_dict(data_dict)

    train_set, val_set, test_set = mat_set.split_into_train_val_test(
        train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=12
    )

    log.info(
        f"train size: {sum(freq_set.H.shape[0] for freq_set in train_set.frequency_sets)}, "
        f"val size: {sum(freq_set.H.shape[0] for freq_set in val_set.frequency_sets)}, "
        f"test size: {sum(freq_set.H.shape[0] for freq_set in test_set.frequency_sets)}"
    )
    train_set_norm = train_set.normalize(normalizer=model.normalizer, transform_H=True)
    val_set_norm = val_set.normalize(normalizer=model.normalizer, transform_H=True)
    test_set_norm = test_set.normalize(normalizer=model.normalizer, transform_H=True)

    logs = {
        "material": material_name,
        "loss_trends_train": [],
        "loss_trends_val": [],
        "start_time": pd.Timestamp.now().round(freq="s"),
    }
    opt_state = optimizer.init(model)

    pbar = trange(n_steps, desc=f"Seed {seed}", position=seed, unit="step")
    test_loss = val_test(test_set_norm, model, past_size)
    log.info(f"Test loss seed {seed}: {test_loss:.6f} A/m")
    for step in pbar:
        train_loss = 0
        key, subkey = jax.random.split(key)
        train_loss, model, opt_state = train_step(
            model, opt_state, train_set_norm, optimizer, subkey, past_size, tbptt_size, batch_size
        )
        pbar_str = f"Loss {train_loss:.2e}"
        if step % val_every == 0:
            val_loss = val_test(val_set_norm, model, past_size)
            logs["loss_trends_val"].append(val_loss)
        pbar_str += f"| val loss {val_loss:.2e}"
        logs["loss_trends_train"].append(train_loss)
        pbar.set_postfix_str(pbar_str)

    test_loss = val_test(test_set_norm, model, past_size)
    log.info(f"Test loss seed {seed}: {test_loss:.6f} A/m")

    logs["end_time"] = pd.Timestamp.now().round(freq="s")
    logs["seed"] = seed
    return logs, model
