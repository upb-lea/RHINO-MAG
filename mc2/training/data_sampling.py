"""From DataSets to proper batches.

This is a very rough implementation based on code from DMPE.. Really needs a refactoring.
Especially, the functions can only batch over all sequences in the current dataset -> i.e. you cannot actually set the batch size..
"""

import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def precompute_starting_points(n_train_steps, k, sequence_length, training_batch_size, loader_key):
    index_normalized = jax.random.uniform(loader_key, shape=(n_train_steps, training_batch_size)) * (
        k + 1 - sequence_length
    )
    starting_points = index_normalized.astype(jnp.int32)
    (loader_key,) = jax.random.split(loader_key, 1)

    return starting_points, loader_key


@eqx.filter_jit
def load_single_batch(dataset, starting_points, sequence_length):

    slice = jnp.linspace(
        start=starting_points, stop=starting_points + sequence_length, num=sequence_length, dtype=int
    ).T

    batched_H = dataset.H[:, slice]
    batched_B = dataset.B[:, slice]

    batched_H = batched_H[:, :, :]
    batched_B = batched_B[:, :, :]
    return batched_H, batched_B


@eqx.filter_jit
def get_data(dataset, sequence_length, training_batch_size, loader_key):

    n_sequences, full_sequence_length = dataset.H.shape
    starting_points, loader_key = precompute_starting_points(
        1, full_sequence_length, sequence_length, training_batch_size, loader_key
    )
    batched_H, batched_B = load_single_batch(dataset, starting_points, sequence_length)

    batched_H = jnp.squeeze(batched_H)[..., None]
    batched_B = jnp.squeeze(batched_B)[..., None]

    # return a batched dataset ?

    return batched_H, batched_B, loader_key
