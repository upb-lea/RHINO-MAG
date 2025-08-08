import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pandas as pd
import torch
import json
import random
import logging as log
from tqdm.notebook import trange
from joblib import Parallel, delayed
from typing import Dict
from torchinfo import summary
from mc2.data_management import (
    AVAILABLE_MATERIALS,
    book_keeping,
    get_train_val_test_pandas_dicts,
    load_data_into_pandas_df,
    setup_package_logging,
    MaterialSet,
    FrequencySet,
    NormalizedFrequencySet,
)
from mc2.features.features_jax import add_fe
from mc2.models.RNN import BaseRNN
from mc2.training.data_sampling import draw_data_uniformly
from mc2.training.optimization import make_step

from IPython.display import clear_output, display

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


def train_recursive_nn(
    material=None,
    debug: bool = False,
    n_steps: int = 100,
    seed: int = 5,
    tbptt_size: int = 64,
    batch_size: int = 1024,
):

    n_steps = 3 if debug else n_steps

    if material is None:
        material = "N87"
    data_dict = load_data_into_pandas_df(material=material)
    mat_set = MaterialSet.from_pandas_dict(data_dict)

    train_set, val_set, test_set = mat_set.split_into_train_val_test(
        train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=12
    )
    max_l_d = {"B": [], "H": []}
    max_d = {"B": None, "H": None, "T": 100}  # max temperature is 100°C

    for freq_set in train_set.frequency_sets:
        max_B = jnp.abs(freq_set.B).max()
        max_H = jnp.abs(freq_set.H).max()
        max_l_d["B"].append(float(max_B))
        max_l_d["H"].append(float(max_H))

    for k, max_l in max_l_d.items():
        max_v = max(max_l)
        log.info(f"max {k} value in training set: {max_v:.2f} T or A/m or °C")
        max_d[k] = max_v

    print(max_l_d)
    # normalize
    train_set_norm = normalize_material_set(train_set, max_d, reduce_train=debug)
    val_set_norm = normalize_material_set(val_set, max_d)
    test_set_norm = normalize_material_set(test_set, max_d)
    del train_set, val_set, test_set  # free memory

    log.info(
        f"train size: {sum(freq_set.H.shape[0] for freq_set in train_set_norm.frequency_sets)}, "
        f"val size: {sum(freq_set.H.shape[0] for freq_set in val_set_norm.frequency_sets)}, "
        f"test size: {sum(freq_set.H.shape[0] for freq_set in test_set_norm.frequency_sets)}"
    )

    # FREQ_CATEGORIES = jnp.array([50000.0, 80000.0, 125000.0, 200000.0, 320000.0, 500000.0, 800000.0])

    def featurize(B, H, T, f):
        # fes = add_fe(jnp.reshape(B, (1, -1)), n_s=tbptt_size)
        return jnp.concatenate([B, T, f], axis=-1), jnp.tanh(H_FACTOR * H)  # fes[0],

    @eqx.filter_jit
    def process_freq_set(model, opt_state, key, freq_set, optimizer):
        key, subkey = jax.random.split(key)
        batch_H, batch_B, batch_T, _ = draw_data_uniformly(
            freq_set,
            training_sequence_length=tbptt_size,
            training_batch_size=batch_size,
            loader_key=subkey,
        )
        batch_f = freq_set.frequency * jnp.ones_like(batch_B) / 800_000

        batch_x, batch_y = jax.vmap(featurize)(batch_B, batch_H, batch_T, batch_f)
        loss, model, opt_state = make_step(model, batch_x, batch_y, optimizer, opt_state)

        return loss, model, opt_state, key

    def train_step(model, opt_state, train_set_norm, optimizer, featurize, key):

        train_loss = 0.0
        for freq_set in train_set_norm.frequency_sets:
            loss, model, opt_state, key = process_freq_set(model, opt_state, key, freq_set, optimizer)
            train_loss += loss / len(train_set_norm.frequencies)

        return train_loss, model, opt_state

    @eqx.filter_jit
    def process_freq_set_val(freq_set, model, featurize):
        f = freq_set.frequency / 800_000
        B = freq_set.B[..., None]
        H = freq_set.H[..., None]
        T = jnp.broadcast_to(freq_set.T[:, None, None], B.shape)
        F = f * jnp.ones_like(B)

        batch_x, batch_y = jax.vmap(featurize)(B, H, T, F)
        pred_y = jax.vmap(model)(batch_x)
        loss = jnp.mean((pred_y - batch_y) ** 2)
        return loss

    def val_test(set, model, featurize):

        val_loss = 0.0
        for freq_set in set.frequency_sets:
            loss = process_freq_set_val(freq_set, model, featurize)
            val_loss += loss / len(set.frequencies)

        return val_loss

    key = jax.random.PRNGKey(seed)
    logs = {
        "material": material,
        "loss_trends_train": [],
        "loss_trends_val": [],
        "start_time": pd.Timestamp.now().round(freq="s"),
    }

    hidden_size = 8
    in_size = 3
    out_size = 1
    model = BaseRNN(in_size, out_size, hidden_size, key=key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    pbar = trange(n_steps, desc=f"Seed {seed}", position=seed, unit="step")
    test_loss = val_test(test_set_norm, model, featurize)
    log.info(f"Test loss seed {seed}: {test_loss:.3f} A/m")
    for step in pbar:
        train_loss = 0
        # train_data: (N, S, features), (N, S, output_features)
        key, subkey = jax.random.split(key)
        train_loss, model, opt_state = train_step(model, opt_state, train_set_norm, optimizer, featurize, subkey)
        pbar_str = f"Loss {train_loss:.2e}"
        if step % VAL_EVERY == 0:
            val_loss = val_test(val_set_norm, model, featurize)
            logs["loss_trends_val"].append(val_loss)
        pbar_str += f"| val loss {val_loss:.2e}"
        logs["loss_trends_train"].append(train_loss)
        pbar.set_postfix_str(pbar_str)

    test_loss = val_test(test_set_norm, model, featurize)
    log.info(f"Test loss seed {seed}: {test_loss:.3f} A/m")

    logs["end_time"] = pd.Timestamp.now().round(freq="s")
    logs["seed"] = seed
    return logs, model


################################################################
################################################################


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
    n_steps: int = 1000,
    val_every: int = 100,
    seed: int = 5,
    tbptt_size: int = 64,
    past_size: int = 750,
    batch_size: int = 1024,
):

    if material_name is None:
        material_name = "N87"
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

    key = jax.random.PRNGKey(seed)
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
