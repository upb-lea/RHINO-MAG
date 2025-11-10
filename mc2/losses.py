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

    mse_loss = MSE_loss(model, B_past, H_past, B_future, H_future, T)

    ##
    mu_0 = 4 * jnp.pi * 10 ** (-7)
    TAU = 1 / 16e6

    # H_e, after (4)
    def He_fn(model, H, M):
        return H + model.model.alpha * M

    # Man (6)
    def Man_fn(model, H, M):
        return model.model.Ms * (jnp.tanh(He_fn(model, H, M) / model.model.a) - (model.model.a / He_fn(model, H, M)))

    # delta
    def delta_fn(H):
        H_new = jnp.squeeze(H)
        diff_H = jnp.diff(H_new, append=1)
        delta = jnp.sign(diff_H)

        return delta

    #  eq (19)
    def fn_dM_dH(model, H, M):
        numerator = Man_fn(model, H, M) - M
        numerator = jnp.squeeze(numerator)

        part1 = delta_fn(H) * model.model.k / mu_0
        part2 = model.model.alpha * (Man_fn(model, H, M) - M)
        part2_sqee = jnp.squeeze(part2)

        # denominator = delta_fn(H)*model.model.k/mu_0 - model.model.alpha*(Man_fn(model,H,M)-M)
        denominator = (delta_fn(H) * model.model.k) / mu_0 - part2_sqee

        M_rev = model.model.c * (Man_fn(model, H, M) - M)
        M_rev = jnp.squeeze(M_rev)

        # (19) + (31)
        dM_dH = numerator / denominator + M_rev
        return dM_dH

    def physics(model, H, B, B_next):
        dB_dt_est = (B_next - B) / TAU
        dB_dt_est = jnp.squeeze(dB_dt_est)
        M = B / mu_0 - H
        dM_dH = fn_dM_dH(model, H, M)
        dM_dB = dM_dH / (mu_0 * (1 + dM_dH))

        dM_dt = dM_dB * dB_dt_est
        dH_dt = 1 / mu_0 * dB_dt_est - dM_dt

        return dH_dt

    physics_at_collocation_points = physics(model, H_past, B_past, B_future)

    physics_loss = 0.5 * jnp.mean(jnp.square(physics_at_collocation_points))

    return mse_loss + 1e-20 * physics_loss
