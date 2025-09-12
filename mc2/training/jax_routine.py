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
from mc2.training.data_sampling import draw_data_uniformly, load_batches, load_batches_material_set
from mc2.training.optimization import make_step
from itertools import zip_longest

DO_TRANSFORM_H = True
H_FACTOR = 1.2
VAL_EVERY = 100


@eqx.filter_value_and_grad
def compute_MSE_loss(
    model: eqx.Module,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
):
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
    )

    return loss, model, opt_state, key


@eqx.filter_jit
def sample_random_indices(freq_idx, seq_idx, seq_len, tbptt_size, past_size, key):
    max_idx = seq_len - (tbptt_size + past_size)
    step = tbptt_size + past_size
    key, subkey = jax.random.split(key)
    first_start_idx = jax.random.randint(minval=0, maxval=(step), shape=(), key=subkey)
    n_chunks = (seq_len - step) // step
    indices = first_start_idx + (step * ((jnp.arange(n_chunks))))  #
    indices = indices.reshape(-1)
    last_random_idx = jax.lax.cond(
        seq_len % step <= first_start_idx,
        lambda _: jnp.array([-1], dtype=jnp.int32),
        lambda _: jnp.array([first_start_idx + step * n_chunks], dtype=jnp.int32),
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
def batched_step(model, batch_H, batch_B, batch_T, past_size, optimizer, opt_state):
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
    )

    return loss, model, opt_state


@eqx.filter_jit
def single_batch_step(material_set, tbptt_size, past_size, batch_pairs, model, optimizer, opt_state):
    n_frequency_indices = batch_pairs[:, 0]
    n_sequence_indices = batch_pairs[:, 1]
    starting_points = batch_pairs[:, 2]
    batch_H, batch_B, batch_T = load_batches_material_set(
        material_set,
        n_frequency_indices,
        n_sequence_indices,
        starting_points,
        training_sequence_length=tbptt_size + past_size,
    )
    loss, new_model, new_opt_state = batched_step(model, batch_H, batch_B, batch_T, past_size, optimizer, opt_state)
    return new_model, new_opt_state, loss


def train_step(model, opt_state, train_set_norm, optimizer, key, past_size, tbptt_size, batch_size):

    train_loss = 0.0
    for freq_set in train_set_norm.frequency_sets:
        loss, model, opt_state, key = process_freq_set(
            model, opt_state, key, freq_set, optimizer, past_size, tbptt_size, batch_size
        )
        train_loss += loss / len(train_set_norm.frequencies)

    return train_loss, model, opt_state


def train_epoch(model, opt_state, train_set_norm, optimizer, key, past_size, tbptt_size, batch_size):
    all_pairs_list = []
    for freq_idx in range(len(train_set_norm.frequency_sets)):
        freq_set = train_set_norm.frequency_sets[freq_idx]
        seq_len = freq_set.H.shape[1]
        n_sequences = freq_set.H.shape[0]
        seq_indices = jnp.arange(n_sequences)

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_sequences)
        idx_pairs = jax.vmap(sample_random_indices, in_axes=(None, 0, None, None, None, 0))(
            freq_idx, seq_indices, seq_len, tbptt_size, past_size, keys
        )
        all_pairs = jnp.concatenate(idx_pairs, axis=0)
        mask = all_pairs[:, 2] != -1
        all_pairs_filtered = all_pairs[mask]

        all_pairs_list.append(all_pairs_filtered)

    all_pairs_global = jnp.concatenate(all_pairs_list, axis=0)
    key, subkey = jax.random.split(key)

    shuffled_pairs = jax.random.permutation(subkey, all_pairs_global)
    all_batch_pairs = create_batch_pairs(shuffled_pairs, batch_size)

    trainable_model_arrays = eqx.filter(model, eqx.is_inexact_array)
    carry_init = (trainable_model_arrays, opt_state)

    def scan_step(carry, batch_indices):
        model_arrays, opt_state = carry
        full_model = eqx.combine(model_arrays, model)
        full_model, opt_state, loss = single_batch_step(
            train_set_norm, tbptt_size, past_size, batch_indices, full_model, optimizer, opt_state
        )
        return (eqx.filter(full_model, eqx.is_inexact_array), opt_state), loss

    (final_model_arrays, final_opt_state), losses = jax.lax.scan(
        scan_step,
        carry_init,
        all_batch_pairs,
    )
    model = eqx.combine(final_model_arrays, model)
    mean_loss = jnp.mean(losses)

    return mean_loss, model, final_opt_state


@eqx.filter_jit
def process_freq_set_val(freq_set, model, past_size):
    B = freq_set.B
    H = freq_set.H
    T = freq_set.T
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
    n_epochs: int,
    val_every: int,
    tbptt_size: int,
    past_size: int,
    batch_size: int,
    subsampling_freq: int,
):

    data_dict = load_data_into_pandas_df(material=material_name)
    mat_set = MaterialSet.from_pandas_dict(data_dict)
    mat_set.subsample(sampling_freq=subsampling_freq)

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
        "start_time": str(pd.Timestamp.now().round(freq="s")),
    }
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    test_loss = val_test(test_set_norm, model, past_size)
    log.info(f"Test loss seed {seed}: {test_loss:.6f} A/m")
    if n_steps > 0 and n_epochs > 0:
        raise ValueError("Please set either `n_steps` or `n_epochs` grater than 0, not both.")
    elif n_steps == 0 and n_epochs == 0:
        raise ValueError("Please set either `n_steps` or `n_epochs` to a value greater than 0.")
    if n_steps > 0:
        pbar = trange(n_steps, desc=f"Seed {seed}", position=seed, unit="step")
        train_func = train_step
    elif n_epochs > 0:
        pbar = trange(n_epochs, desc=f"Seed {seed}", position=seed, unit="epoch")
        train_func = train_epoch

    for step in pbar:
        train_loss = 0
        key, subkey = jax.random.split(key)
        train_loss, model, opt_state = train_func(
            model, opt_state, train_set_norm, optimizer, subkey, past_size, tbptt_size, batch_size
        )
        pbar_str = f"Loss {train_loss:.2e}"
        if step % val_every == 0:
            val_loss = val_test(val_set_norm, model, past_size)
            logs["loss_trends_val"].append(val_loss.item())
        pbar_str += f"| val loss {val_loss:.2e}"
        logs["loss_trends_train"].append(train_loss.item())
        pbar.set_postfix_str(pbar_str)

    pbar.close()

    test_loss = val_test(test_set_norm, model, past_size)
    log.info(f"Test loss seed {seed}: {test_loss:.6f} A/m")

    logs["end_time"] = str(pd.Timestamp.now().round(freq="s"))
    logs["seed"] = seed
    return logs, model, (train_set, val_set, test_set)
