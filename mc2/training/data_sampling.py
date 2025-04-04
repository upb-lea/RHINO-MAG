"""From DataSets to proper batches."""

import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def sample_batch_indices(n_sequences, full_sequence_length, sequence_length, training_batch_size, loader_key):

    updated_loader_key, batch_key, starting_point_key = jax.random.split(loader_key, 3)

    n_sequence_indices = jax.random.randint(batch_key, shape=(training_batch_size), minval=0, maxval=n_sequences - 1)

    index_normalized = jax.random.uniform(starting_point_key, shape=(training_batch_size)) * (
        full_sequence_length + 1 - sequence_length
    )
    starting_points = index_normalized.astype(jnp.int32)

    return n_sequence_indices, starting_points, updated_loader_key


@eqx.filter_jit
def load_batches(dataset, n_sequence_indices, starting_points, sequence_length):

    slice = jnp.linspace(
        start=starting_points, stop=starting_points + sequence_length, num=sequence_length, dtype=int
    ).T

    batched_H = dataset.H[n_sequence_indices[..., None], slice]
    batched_B = dataset.B[n_sequence_indices[..., None], slice]

    return batched_H, batched_B


@eqx.filter_jit
def get_data(dataset, sequence_length, training_batch_size, loader_key):

    n_sequences, full_sequence_length = dataset.H.shape
    n_sequence_indices, starting_points, loader_key = sample_batch_indices(
        n_sequences, full_sequence_length, sequence_length, training_batch_size, loader_key
    )
    batched_H, batched_B = load_batches(dataset, n_sequence_indices, starting_points, sequence_length)

    batched_H = jnp.squeeze(batched_H)[..., None]
    batched_B = jnp.squeeze(batched_B)[..., None]

    return batched_H, batched_B, loader_key
