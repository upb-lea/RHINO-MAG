"""From DataSets to proper batches."""

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import FrequencySet, MaterialSet


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
    dataset: FrequencySet,
    n_sequence_indices: jax.Array,
    starting_points: jax.Array,
    training_sequence_length: int,
):
    """Load batches of data from the frequency set using dynamic slicing."""

    def slice_sequence(sequence, start_idx):
        return jax.lax.dynamic_slice(sequence, (start_idx,), (training_sequence_length,))

    def get_H_B_T(seq_idx, start_idx):
        H_seq = slice_sequence(dataset.H[seq_idx], start_idx)
        B_seq = slice_sequence(dataset.B[seq_idx], start_idx)
        T_val = dataset.T[seq_idx]
        H_rms = dataset.H_RMS[seq_idx]
        # H_full = dataset.H[seq_idx]
        # H_rms_full = jnp.sqrt(jnp.mean(jnp.square(H_full)))

        return H_seq, B_seq, T_val, H_rms

    batched_H, batched_B, batched_T, batch_H_rms_full = jax.vmap(get_H_B_T)(n_sequence_indices, starting_points)

    return batched_H, batched_B, batched_T, batch_H_rms_full


@eqx.filter_jit
def load_batches_material_set(
    material_set: MaterialSet,
    n_frequency_indices: jax.Array,
    n_sequence_indices: jax.Array,
    starting_points: jax.Array,
    training_sequence_length: int,
):
    """Load batches of data from the material set using dynamic slicing."""

    def slice_sequence(sequence, start_idx):
        return jax.lax.dynamic_slice(sequence, (start_idx,), (training_sequence_length,))

    def get_H_B_T(freq_idx, seq_idx, start_idx):
        def make_case(i):
            dataset = material_set.frequency_sets[i]

            H_seq = slice_sequence(dataset.H[seq_idx], start_idx)
            B_seq = slice_sequence(dataset.B[seq_idx], start_idx)
            T_val = dataset.T[seq_idx]
            H_rms = dataset.H_RMS[seq_idx]
            # H_full = dataset.H[seq_idx]
            # H_rms = jnp.sqrt(jnp.mean(jnp.square(H_full)))

            return H_seq, B_seq, T_val, H_rms

        cases = [lambda i=i: make_case(i) for i in range(len(material_set.frequency_sets))]
        return jax.lax.switch(freq_idx, cases)

    batched_H, batched_B, batched_T, batch_H_rms_full = jax.vmap(get_H_B_T)(
        n_frequency_indices, n_sequence_indices, starting_points
    )

    return batched_H, batched_B, batched_T, batch_H_rms_full


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
    batched_H, batched_B, batched_T, batch_H_rms_full = load_batches(
        dataset, n_sequence_indices, starting_points, training_sequence_length
    )

    batched_H = jnp.squeeze(batched_H)
    batched_B = jnp.squeeze(batched_B)
    batched_T = jnp.squeeze(batched_T)

    return batched_H, batched_B, batched_T, batch_H_rms_full, loader_key


def data_loader(dataset, training_sequence_length, training_batch_size, loader_key):
    """TODO: Sets up a deterministic sequence of data that is iterated through.
    Deterministic means here that is it predefined at the start of the training.

    """
    raise NotImplementedError("Deterministic data loader is not implemented yet.")
