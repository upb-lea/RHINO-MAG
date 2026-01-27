from typing import Callable, Tuple
import pandas as pd
import logging as log

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# from tqdm.notebook import trange  #
from tqdm import trange
from rhmag.data_management import (
    load_data_into_pandas_df,
    MaterialSet,
    FrequencySet,
)
from rhmag.training.data_sampling import draw_data_uniformly, load_batches, load_batches_material_set
from rhmag.model_interfaces.model_interface import ModelInterface

# import orbax.checkpoint as ocp

DO_TRANSFORM_H = True
H_FACTOR = 1.2
VAL_EVERY = 100


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
    batch_H_rms: jax.Array,
    loss_function: Callable,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    loss, grads = loss_function(model, B_past, H_past, B_future, H_future, T, batch_H_rms)

    grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0), grads)

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def process_freq_set(model, opt_state, key, freq_set, loss_function, optimizer, past_size, tbptt_size, batch_size):
    key, subkey = jax.random.split(key)
    batch_H, batch_B, batch_T, batch_H_rms, _ = draw_data_uniformly(
        freq_set,
        training_sequence_length=tbptt_size + past_size,
        training_batch_size=batch_size,
        loader_key=subkey,
    )

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
        batch_H_rms,
        loss_function,
        optimizer,
        opt_state,
    )

    return loss, model, opt_state, key


@eqx.filter_jit
def sample_random_indices(freq_idx, seq_idx, seq_len, tbptt_size, past_size, key):
    step = tbptt_size + past_size
    max_idx = seq_len - step

    key, subkey1 = jax.random.split(key)
    first_start_idx = jax.random.randint(subkey1, shape=(), minval=0, maxval=step)

    n_chunks = (seq_len - step) // step
    indices = first_start_idx + step * jnp.arange(n_chunks)

    key, subkey2 = jax.random.split(key)
    last_random_idx = jax.lax.cond(
        seq_len % step <= first_start_idx,
        lambda _: jax.random.randint(subkey2, shape=(1,), minval=0, maxval=max_idx + 1),
        lambda _: jnp.array([first_start_idx + step * n_chunks]),
        operand=None,
    )

    indices = jnp.concatenate([jnp.array([0]), indices, last_random_idx, jnp.array([max_idx])], axis=0)

    seq_idx_array = jnp.full(indices.shape, seq_idx)
    freq_idx_array = jnp.full(indices.shape, freq_idx)
    pairs = jnp.stack([freq_idx_array, seq_idx_array, indices], axis=1)
    return pairs


@eqx.filter_jit
def create_batch_pairs(pairs, batch_size):
    n = pairs.shape[0]
    n_batches = n // batch_size
    truncated_pairs = pairs[: n_batches * batch_size]
    batched_pairs = truncated_pairs.reshape(n_batches, batch_size, 3)
    return batched_pairs


@eqx.filter_jit
def batched_step(model, batch_H, batch_B, batch_T, batch_H_RMS, past_size, loss_function, optimizer, opt_state):
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
        batch_H_RMS,
        loss_function,
        optimizer,
        opt_state,
    )

    return loss, model, opt_state


def add_gaussian_noise(in_data, noise_key, noise_std: float = 0.002):
    return in_data + jax.random.normal(noise_key, shape=in_data.shape) * noise_std


@eqx.filter_jit
def single_batch_step(
    material_set,
    tbptt_size,
    past_size,
    batch_pairs,
    model,
    loss_function,
    optimizer,
    opt_state,
    noise_key,
    noise_on_data,
):
    n_frequency_indices = batch_pairs[:, 0]
    n_sequence_indices = batch_pairs[:, 1]
    starting_points = batch_pairs[:, 2]
    batch_H, batch_B, batch_T, batch_H_RMS = load_batches_material_set(
        material_set,
        n_frequency_indices,
        n_sequence_indices,
        starting_points,
        training_sequence_length=tbptt_size + past_size,
    )

    if noise_on_data > 0.0:
        batch_B = add_gaussian_noise(batch_B, noise_key, noise_on_data)

    loss, new_model, new_opt_state = batched_step(
        model, batch_H, batch_B, batch_T, batch_H_RMS, past_size, loss_function, optimizer, opt_state
    )
    return new_model, new_opt_state, loss


def train_step(model, opt_state, train_set_norm, loss_function, optimizer, key, past_size, tbptt_size, batch_size):
    train_loss = 0.0
    for freq_set in train_set_norm.frequency_sets:
        loss, model, opt_state, key = process_freq_set(
            model, opt_state, key, freq_set, loss_function, optimizer, past_size, tbptt_size, batch_size
        )
        train_loss += loss / len(train_set_norm.frequencies)

    return train_loss, model, opt_state


@eqx.filter_jit
def scan_step(carry, batch_indices, model, train_set_norm, tbptt_size, past_size, optimizer):
    model_arrays, opt_state = carry
    full_model = eqx.combine(model_arrays, model)
    full_model, opt_state, loss = single_batch_step(
        train_set_norm, tbptt_size, past_size, batch_indices, full_model, optimizer, opt_state
    )
    return (eqx.filter(full_model, eqx.is_inexact_array), opt_state), loss


@eqx.filter_jit
def train_epoch(
    model, opt_state, train_set_norm, loss_function, optimizer, key, past_size, tbptt_size, batch_size, noise_on_data
):
    all_pairs_list = []
    for freq_idx in range(len(train_set_norm.frequency_sets)):
        freq_set = train_set_norm.frequency_sets[freq_idx]
        n_sequences, seq_len = freq_set.H.shape
        seq_indices = jnp.arange(n_sequences)

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_sequences)
        idx_pairs = jax.vmap(sample_random_indices, in_axes=(None, 0, None, None, None, 0))(
            freq_idx, seq_indices, seq_len, tbptt_size, past_size, keys
        )
        all_pairs = jnp.concatenate(idx_pairs, axis=0)
        all_pairs_list.append(all_pairs)

    all_pairs_global = jnp.concatenate(all_pairs_list, axis=0)
    key, subkey = jax.random.split(key)
    shuffled_pairs = jax.random.permutation(subkey, all_pairs_global)
    all_batch_pairs = create_batch_pairs(shuffled_pairs, batch_size)

    trainable_model_arrays = eqx.filter(model, eqx.is_inexact_array)
    carry_init = (trainable_model_arrays, opt_state, key)

    def scan_step(carry, batch_indices):
        model_arrays, opt_state, key = carry

        key, noise_key = jax.random.split(key)
        full_model = eqx.combine(model_arrays, model)
        full_model, opt_state, loss = single_batch_step(
            train_set_norm,
            tbptt_size,
            past_size,
            batch_indices,
            full_model,
            loss_function,
            optimizer,
            opt_state,
            noise_key,
            noise_on_data,
        )
        return (eqx.filter(full_model, eqx.is_inexact_array), opt_state, key), loss

    (final_model_arrays, final_opt_state, final_key), losses = jax.lax.scan(
        scan_step,
        carry_init,
        all_batch_pairs,
    )
    model = eqx.combine(final_model_arrays, model)
    mean_loss = jnp.mean(losses)

    return mean_loss, model, final_opt_state


def train_epoch2(model, opt_state, train_set_norm, optimizer, key, past_size, tbptt_size, batch_size):
    all_pairs_list = []
    for freq_idx in range(len(train_set_norm.frequency_sets)):
        freq_set = train_set_norm.frequency_sets[freq_idx]
        n_sequences, seq_len = freq_set.H.shape
        seq_indices = jnp.arange(n_sequences)

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_sequences)
        idx_pairs = jax.vmap(sample_random_indices, in_axes=(None, 0, None, None, None, 0))(
            freq_idx, seq_indices, seq_len, tbptt_size, past_size, keys
        )
        all_pairs = jnp.concatenate(idx_pairs, axis=0)
        all_pairs_list.append(all_pairs)

    all_pairs_global = jnp.concatenate(all_pairs_list, axis=0)
    key, subkey = jax.random.split(key)
    shuffled_pairs = jax.random.permutation(subkey, all_pairs_global)
    all_batch_pairs = create_batch_pairs(shuffled_pairs, batch_size)
    # Filterbare Arrays für das Model
    trainable_model_arrays = eqx.filter(model, eqx.is_inexact_array)

    losses = []
    for batch_indices in all_batch_pairs:
        model, opt_state, loss = single_batch_step(
            train_set_norm, tbptt_size, past_size, batch_indices, model, optimizer, opt_state
        )
        losses.append(loss)

    mean_loss = jnp.mean(jnp.array(losses))
    return mean_loss, model, opt_state


@eqx.filter_jit
def process_freq_set_val(freq_set, model: ModelInterface, past_size):
    B = freq_set.B
    H = freq_set.H
    T = freq_set.T

    batch_H_past = H[:, :past_size]
    batch_H_future = H[:, past_size:]
    batch_B_past = B[:, :past_size]
    batch_B_future = B[:, past_size:]

    pred_H = model(batch_B_past, batch_H_past, batch_B_future, T)  # , f

    H_rms_full = jnp.sqrt(jnp.mean(jnp.square(H), axis=1))
    rms_loss = jnp.sqrt(jnp.mean((pred_H - batch_H_future) ** 2, axis=1))
    norm_rms_loss = rms_loss / H_rms_full
    loss = jnp.mean(norm_rms_loss)
    return loss, pred_H, batch_H_future


def val_test(set, model, past_size):
    if set is None:
        return jnp.array(-1.0), [jnp.array([-1.0])], [jnp.array([-1.0])]

    val_loss = 0.0
    val_pred_l = []
    val_gt_l = []
    for freq_set in set.frequency_sets:
        loss, pred, gt = process_freq_set_val(freq_set, model, past_size)
        val_loss += loss / len(set.frequencies)
        val_pred_l.append(pred)
        val_gt_l.append(gt)

    return val_loss, val_pred_l, val_gt_l


def train_model(
    model,
    loss_function,
    optimizer,
    material_name,
    data_tuple: Tuple[MaterialSet, MaterialSet, MaterialSet],
    key: jax.random.PRNGKey,
    seed: int,
    n_steps: int,
    n_epochs: int,
    val_every: int,
    tbptt_size: int,
    past_size: int,
    batch_size: int,
    time_shift: int,
    noise_on_data: float,
    tbptt_size_start: list[int] | None = None,  # (size, n_epochs_steps)
    **kwargs,
):
    train_set, val_set, test_set = data_tuple

    if test_set is not None and val_set is not None:
        log.info(
            f"train size: {sum(freq_set.H.shape[0] for freq_set in train_set.frequency_sets)}, "
            f"val size: {sum(freq_set.H.shape[0] for freq_set in val_set.frequency_sets)}, "
            f"test size: {sum(freq_set.H.shape[0] for freq_set in test_set.frequency_sets)}"
        )
    else:
        log.info(
            f"train size: {sum(freq_set.H.shape[0] for freq_set in train_set.frequency_sets)}, "
            f"val size: {val_set}, "
            f"test size: {test_set}"
        )

    train_set_norm = train_set.normalize(normalizer=model.normalizer, transform_H=None)

    logs = {
        "material": material_name,
        "loss_trends_train": [],
        "loss_trends_val": [],
        "start_time": str(pd.Timestamp.now().round(freq="s")),
    }
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


    best_val_loss = float("inf")
    best_model = jax.tree.map(lambda x: x, model)

    test_loss, *_ = val_test(test_set, model, past_size)
    log.info(f"Test loss seed {seed}: {test_loss:.6f} A/m")
    if (n_steps > 0 and n_epochs > 0) or (n_steps == 0 and n_epochs == 0):
        raise ValueError("Please set either `n_steps` or `n_epochs` to a value greater than 0.")
    if n_steps > 0:
        pbar = trange(n_steps, desc=f"Seed {seed}", position=seed, unit="step")
        train_func = train_step
    elif n_epochs > 0:
        pbar = trange(n_epochs, desc=f"Seed {seed}", position=seed, unit="epoch")
        train_func = train_epoch

    for step_epoch in pbar:
        if tbptt_size_start is not None:
            tbptt_size_st, n_epochs_start = tbptt_size_start
            if step_epoch < n_epochs_start:
                current_tbptt_size = tbptt_size_st
            else:
                current_tbptt_size = tbptt_size
        else:
            current_tbptt_size = tbptt_size

        train_loss = 0
        key, subkey = jax.random.split(key)
        train_loss, model, opt_state = train_func(
            model,
            opt_state,
            train_set_norm,
            loss_function,
            optimizer,
            subkey,
            past_size,
            current_tbptt_size,
            batch_size,
            noise_on_data,
        )
        pbar_str = f"Loss {train_loss:.2e}"
        if step_epoch % val_every == 0:
            val_loss, *_ = val_test(val_set, model, past_size)  # val_set_norm
            logs["loss_trends_val"].append(val_loss.item())
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model = jax.tree.map(lambda x: x, model)
        pbar_str += f"| val loss {val_loss:.2e}"
        logs["loss_trends_train"].append(train_loss.item())
        pbar.set_postfix_str(pbar_str)

    pbar.close()

    if val_set is not None:
        if val_every > 0 and val_every < n_epochs:
            final_model = best_model
        else:
            final_model = model
    else:
        final_model = model

    test_loss, test_pred_l, test_gt_l = val_test(test_set, final_model, past_size)  # test_set_norm
    log.info(f"Test loss seed {seed}: {test_loss:.6f} A/m")

    logs["end_time"] = str(pd.Timestamp.now().round(freq="s"))
    logs["seed"] = seed
    for i, (test_pred, test_gt) in enumerate(zip(test_pred_l, test_gt_l)):
        logs[f"predictions_MS_{i}"] = test_pred
        logs[f"ground_truth_MS_{i}"] = test_gt
    return logs, final_model
