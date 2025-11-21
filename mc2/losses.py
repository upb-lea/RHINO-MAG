import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import Normalizer


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
    return adapted_RMS_trajectory_based(pred_H, B_past, B_future, H_future, batch_H_rms, model.normalizer)


def adapted_RMS_trajectory_based(
    pred_H: jax.Array,
    B_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    batch_H_rms: jax.Array,
    normalizer: Normalizer,
) -> jax.Array:

    # approximate dB/dt
    B_last_past = B_past[:, -1:]
    B_concat = jnp.concatenate([B_last_past, B_future], axis=1)
    abs_dB_future = jnp.abs(jnp.diff(B_concat, axis=1))

    # denormalize prediction because of tanh at the output
    pred_H_inv_transf = normalizer.H_inverse_transform(pred_H)
    H_future_inv_transf = normalizer.H_inverse_transform(H_future)

    # actual loss computation
    H_rms_error = jnp.sqrt(jnp.mean((pred_H_inv_transf - H_future_inv_transf) ** 2 * abs_dB_future, axis=1))  #

    # normalization with H_rms
    batch_H_rms_norm = batch_H_rms / normalizer.H_max
    H_rms_norm = H_rms_error / batch_H_rms_norm

    loss = jnp.mean(H_rms_norm)
    loss = jnp.nan_to_num(loss, nan=0.0, posinf=1e7, neginf=-1e7)
    return loss


def pinn_gru_loss(
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

    H_pred_LH = (model.normalized_call)(B_past, H_past, B_future, T)
    rms_loss = adapted_RMS_trajectory_based(H_pred_LH, B_past, B_future, H_future, batch_H_rms, model.normalizer)

    # mse_loss = MSE_loss(model, B_past, H_past, B_future, H_future, T)

    ##
    mu_0 = 4 * jnp.pi * 10 ** (-7)
    TAU = 1 / 16e6

    def coth_stable(x):
        eps = 1e-7
        x = jnp.where(jnp.abs(x) < eps, eps * jnp.sign(x), x)
        return 1 / jnp.tanh(x)

    # H_pred_LH = (model.normalized_call)(B_past, H_past, B_future, T)

    def fn_dM_dH(model, M, dB_dt, B, B_next, T, H_pred_LH):
        H_e = H_pred_LH + model.model.alpha * M
        M_an = model.model.Ms * (coth_stable(H_e / model.model.a) - model.model.a / H_e)
        delta_m = 0.5 * (1 + jnp.sign((M_an - M) * dB_dt))

        dM_an_dH_e = (
            model.model.Ms / model.model.a * (1 - (coth_stable(H_e / model.model.a)) ** 2 + (model.model.a / H_e) ** 2)
        )
        delta = jnp.sign(dB_dt)

        numerator = delta_m * (M_an - M) + model.model.c * model.model.k * delta * dM_an_dH_e
        denominator = model.model.k * delta - model.model.alpha * numerator

        dM_dH = numerator / denominator

        return dM_dH - numerator / denominator

    def physics(model, B_past, B_future, H_past, T, H_pred_LH):
        B_ext = jnp.concatenate([B_past[-1:], B_future])
        dB_dt_est = jnp.diff(B_ext) / TAU
        # dB_dt_est = (B_next - B) / TAU
        M = B_future / mu_0 - H_pred_LH
        dM_dH = fn_dM_dH(model, M, dB_dt_est, B_past, B_future, T, H_pred_LH)
        dM_dB = dM_dH / (mu_0 * (1 + dM_dH))
        dM_dt = dM_dB * dB_dt_est
        dH_dt = 1 / mu_0 * dB_dt_est - dM_dt

        return dH_dt

    physics_at_collocation_points = (
        eqx.filter_vmap(physics, in_axes=(None, 0, 0, 0, None, 0))(model, B_past, B_future, H_past, T, H_pred_LH) * TAU
    )

    print(physics_at_collocation_points.shape)

    # def dH_dt_GRU(H_past, H_pred_LH, TAU):
    #     H_ext = jnp.concatenate([H_past[-1:], H_pred_LH])
    #     return jnp.diff(H_ext) / TAU

    # loss_GRU_points = eqx.filter_vmap(dH_dt_GRU, in_axes=(0, 0, None))(H_past, H_pred_LH, TAU)

    # print(H_past.shape)
    # print(H_pred_LH.shape)
    # print(loss_GRU_points.shape)

    physics_loss = 1 / jnp.size(H_pred_LH) * jnp.sum(jnp.square(physics_at_collocation_points - H_pred_LH))

    return rms_loss + model.model.physics_weight_lambda * physics_loss
