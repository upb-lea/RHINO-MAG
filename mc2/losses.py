"""Implementation of training loss functions."""

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.model_interfaces.model_interface import ModelInterface


def MSE_loss(
    model: ModelInterface,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
    *args,
    **kwargs,
) -> jax.Array:
    """Computes the MSE between H_future and the prediction of the given model.

    Args:
        model (ModelInterface): Model to use for prediction
        B_past (jax.Array): The normalized flux density values from time step k0
            to k1 with shape (n_batches, past_sequence_length)
        H_past (jax.Array): The normalized field values from time step k0 to k1
            with shape (n_batches, past_sequence_length)
        B_future (jax.Array): The physical normalized flux density values from
            time step k1 to k2 with shape (n_batches, future_sequence_length)
        T (float): The normalized temperature of the material with shape (n_batches,)
        *args, **kwargs: The loss can take further arguments that are ignored so that
            all losses can be called in the same way, even if some require additional
            arguments

    Returns:
        The resulting MSE value as a jax.Array
    """
    pred_H = (model.normalized_call)(B_past, H_past, B_future, T)
    return jnp.mean((pred_H - H_future) ** 2)


def adapted_RMS_loss(
    model: ModelInterface,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
    batch_H_rms: jax.Array,
    *args,
    **kwargs,
) -> jax.Array:
    """Computes the adapted RMS loss between H_future and the prediction of the given model.

    Args:
        model (ModelInterface): Model to use for prediction
        B_past (jax.Array): The normalized flux density values from time step k0
            to k1 with shape (n_batches, past_sequence_length)
        H_past (jax.Array): The normalized field values from time step k0 to k1
            with shape (n_batches, past_sequence_length)
        B_future (jax.Array): The physical normalized flux density values from
            time step k1 to k2 with shape (n_batches, future_sequence_length)
        T (float): The normalized temperature of the material with shape (n_batches,)
        batch_H_rms (jax.Array): The RMS value of each of the sequences with shape
            (n_batches,)
        *args, **kwargs: The loss can take further arguments that are ignored so that
            all losses can be called in the same way, even if some require additional
            arguments

    Returns:
        The resulting adapted RMS loss value as a jax.Array
    """
    pred_H = (model.normalized_call)(B_past, H_past, B_future, T)

    # approximate dB/dt
    B_last_past = B_past[:, -1:]
    B_concat = jnp.concatenate([B_last_past, B_future], axis=1)
    abs_dB_future = jnp.abs(jnp.diff(B_concat, axis=1))

    # denormalize prediction because of tanh at the output
    pred_H_inv_transf = model.normalizer.H_inverse_transform(pred_H)
    H_future_inv_transf = model.normalizer.H_inverse_transform(H_future)

    # actual loss computation
    H_rms_error = jnp.sqrt(jnp.mean((pred_H_inv_transf - H_future_inv_transf) ** 2 * abs_dB_future, axis=1))  #

    # normalization with H_rms
    batch_H_rms_norm = batch_H_rms / model.normalizer.H_max
    H_rms_norm = H_rms_error / batch_H_rms_norm

    loss = jnp.mean(H_rms_norm)
    loss = jnp.nan_to_num(loss, nan=0.0, posinf=1e7, neginf=-1e7)
    return loss
