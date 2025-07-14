"""From DataSets to proper batches."""

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import FrequencySet


@eqx.filter_jit
def sample_batch_indices(
    n_sequences: int,
    full_sequence_length: jax.Array,
    training_sequence_length: int,
    training_batch_size: int,
    loader_key: jax.Array,
):
    """Sample batch indices for the data loader.

    Note: In the current implementation, the full_sequence length can be varied without recompilation of the
    function. The other parameters are fixed at compile time and cannot be jax.Array as a result. Might make
    sense to change this in the future.

    Args:
        n_sequences (int): Number of sequences in the dataset.
        full_sequence_length (jax.Array): Length of the full sequence.
        training_sequence_length (int): Length of the training sequence.
        training_batch_size (int): Size of the training batch.
        loader_key (jax.Array): Random key for sampling.

    Returns:
        n_sequence_indices (jax.Array): Indices of the sequences.
        starting_points (jax.Array): Starting points for the sequences.
        updated_loader_key (jax.Array): Updated random key.
    """

    updated_loader_key, batch_key, starting_point_key = jax.random.split(loader_key, 3)

    n_sequence_indices = jax.random.randint(batch_key, shape=(training_batch_size), minval=0, maxval=n_sequences - 1)

    index_normalized = jax.random.uniform(starting_point_key, shape=(training_batch_size)) * (
        full_sequence_length + 1 - training_sequence_length
    )
    starting_points = index_normalized.astype(jnp.int32)

    return n_sequence_indices, starting_points, updated_loader_key


@eqx.filter_jit
def load_batches(
    dataset: FrequencySet, n_sequence_indices: jax.Array, starting_points: jax.Array, training_sequence_length: int
):
    """Load batches of data from the dataset according to the sampled indices.

    Args:
        dataset (DataSet): Dataset object containing the data.
        n_sequence_indices (jax.Array): Indices of the sequences.
        starting_points (jax.Array): Starting points for the sequences.
        training_sequence_length (int): Length of the training sequence.

    Returns:
        batched_H (jax.Array): Batched H data.
        batched_B (jax.Array): Batched B data.
    """

    slice = jnp.linspace(
        start=starting_points, stop=starting_points + training_sequence_length, num=training_sequence_length, dtype=int
    ).T

    batched_H = dataset.H[n_sequence_indices[..., None], slice]
    batched_B = dataset.B[n_sequence_indices[..., None], slice]
    batched_T = dataset.T[n_sequence_indices]

    return batched_H, batched_B, batched_T


@eqx.filter_jit
def draw_data_uniformly(
    dataset: FrequencySet, training_sequence_length: int, training_batch_size: int, loader_key: jax.Array
):
    """Draw data uniformly from the dataset.

    Args:
        dataset (DataSet): Dataset object containing the data.
        training_sequence_length (int): Length of the training sequence.
        training_batch_size (int): Size of the training batch.
        loader_key (jax.Array): Random key for sampling.

    Returns:
        batched_H (jax.Array): Batched H data.
        batched_B (jax.Array): Batched B data.
        loader_key (jax.Array): Updated random key.
    """

    n_sequences, full_sequence_length = dataset.H.shape
    n_sequence_indices, starting_points, loader_key = sample_batch_indices(
        n_sequences, full_sequence_length, training_sequence_length, training_batch_size, loader_key
    )
    batched_H, batched_B, batched_T = load_batches(
        dataset, n_sequence_indices, starting_points, training_sequence_length
    )

    batched_H = jnp.squeeze(batched_H)[..., None]
    batched_B = jnp.squeeze(batched_B)[..., None]
    batched_T = jnp.squeeze(batched_T)[:, None, None]
    batched_T = jnp.broadcast_to(batched_T, batched_B.shape)

    return batched_H, batched_B, batched_T, loader_key


def data_loader(dataset, training_sequence_length, training_batch_size, loader_key):
    """TODO: Sets up a deterministic sequence of data that is iterated through.
    Deterministic means here that is it predefined at the start of the training.

    """
    raise NotImplementedError("Deterministic data loader is not implemented yet.")
