import jax
import jax.numpy as jnp
import equinox as eqx


def MSE_loss(
    model: eqx.Module,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
    *args,
    **kwargs,
) -> jax.Array:
    pred_H = (model.normalized_call)(B_past, H_past, B_future, T)
    return jnp.mean((pred_H - H_future) ** 2)


def adapted_RMS_loss(
    model: eqx.Module,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: jax.Array,
    batch_H_rms: jax.Array,
    *args,
    **kwargs,
) -> jax.Array:
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
