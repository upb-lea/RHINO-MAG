"""Differentiable Preisach model in JAX based on https://github.com/roussel-ryan/diff_hysteresis.

WARNING: This code implements H to B since this is what is usually found in the literature.
We will need to invert the model to get B to H in the future.
"""

from copy import deepcopy

import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx


class HysteronDensity(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.leaky_relu,
            final_activation=jnn.sigmoid,
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, alpha_beta):
        return self.mlp(alpha_beta)


@eqx.filter_jit
def hysteron_operator(H, initial_field, initial_output, alpha_beta, T):

    alpha = alpha_beta[0]
    beta = alpha_beta[1]

    def true_fun(H, initial_field, initial_output, alpha, beta, T):
        jax.debug.print("THIS SHOULD NOT HAPPEN")
        return jnp.ones(initial_field.shape)

    def false_fun(H, initial_field, initial_output, alpha, beta, T):
        def _true_fun(H, initial_output, alpha, beta, T):
            return jax.lax.min(initial_output + (1 + jnp.tanh((H - alpha) / jnp.abs(T))), jnp.array([1.0]))

        def _false_fun(H, initial_output, alpha, beta, T):
            return jax.lax.max(initial_output - (1 + jnp.tanh(-(H - beta) / jnp.abs(T))), jnp.array([-1.0]))

        return jax.lax.cond((H > initial_field)[0], _true_fun, _false_fun, H, initial_output, alpha, beta, T)

    return jax.lax.cond((H == initial_field)[0], true_fun, false_fun, H, initial_field, initial_output, alpha, beta, T)


class DifferentiablePreisach(eqx.Module):
    hysteron_density: HysteronDensity
    A: jax.Array

    def __init__(self, width_size, depth, *, model_key, **kwargs):
        super().__init__(**kwargs)

        poly_params, nn_key = jax.random.split(model_key)
        # self.A = jax.random.uniform(poly_params, shape=(3,), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        self.A = jnp.array([10.0, 0.0, 0.0], dtype=jnp.float32)

        self.hysteron_density = HysteronDensity(
            width_size=width_size,
            depth=depth,
            key=nn_key,
        )

    @eqx.filter_jit
    def __call__(self, H, initial_field, initial_operator_values, alpha_beta_grid, T=1):
        hysteron_density_values = jax.vmap(self.hysteron_density)(alpha_beta_grid)
        hysteron_operator_values = jax.vmap(hysteron_operator, in_axes=(None, None, 0, 0, None))(
            H, initial_field, initial_operator_values, alpha_beta_grid, T
        )

        est_B = jnp.mean(hysteron_density_values * hysteron_operator_values)[None]
        est_B = self.A[0] * est_B + self.A[1] * H + self.A[2]

        return est_B, hysteron_operator_values


class ArrayPreisach(eqx.Module):
    hysteron_density: jax.Array
    A: jax.Array

    def __init__(self, hysteron_density, **kwargs):
        super().__init__(**kwargs)
        self.hysteron_density = hysteron_density
        self.A = jnp.array([10.0, 0.0, 0.0], dtype=jnp.float32)

    @eqx.filter_jit
    def __call__(self, H, initial_field, initial_operator_values, alpha_beta_grid, T=1):
        hysteron_density_values = self.hysteron_density
        hysteron_operator_values = jax.vmap(hysteron_operator, in_axes=(None, None, 0, 0, None))(
            H, initial_field, initial_operator_values, alpha_beta_grid, T
        )

        est_B = jnp.mean(hysteron_density_values * hysteron_operator_values)[None]
        est_B = self.A[0] * est_B + self.A[1] * H + self.A[2]

        return est_B, hysteron_operator_values


def update_state(H, carry):

    def true_fun(H, carry):
        # case that we are going in a positive direction

        def _true_fun(carry):
            # positive direction and sign change -> update initial states
            positive_direction, initial_field, last_H, initial_operator_values, last_operator_values = carry
            initial_operator_values = last_operator_values
            initial_field = last_H
            positive_direction = jnp.array([False])
            return positive_direction, initial_field, initial_operator_values

        def _false_fun(carry):
            # positive direction and no sign change -> return values as they are
            positive_direction, initial_field, last_H, initial_operator_values, last_operator_values = carry
            return positive_direction, initial_field, initial_operator_values

        last_H = carry[2]
        return jax.lax.cond((H < last_H)[0], _true_fun, _false_fun, carry)

    def false_fun(H, carry):
        def _true_fun(carry):
            # negative direction and sign change -> update initial states
            positive_direction, initial_field, last_H, initial_operator_values, last_operator_values = carry
            initial_operator_values = last_operator_values
            initial_field = last_H
            positive_direction = jnp.array([True])

            return positive_direction, initial_field, initial_operator_values

        def _false_fun(carry):
            # negative direction and no sign change -> return values as they are
            positive_direction, initial_field, last_H, initial_operator_values, last_operator_values = carry
            return positive_direction, initial_field, initial_operator_values

        last_H = carry[2]
        return jax.lax.cond((H > last_H)[0], _true_fun, _false_fun, carry)

    positive_direction = carry[0]
    return jax.lax.cond(positive_direction[0], true_fun, false_fun, H, carry)


def estimate_B(H_trajectory, model, alpha_beta_grid, initial_field=None, initial_operator_values=None, T=1):
    """Estimate B from H using the Preisach model."""
    initial_field = jnp.array([-100.0]) if initial_field is None else initial_field
    initial_operator_values = (
        -jnp.ones((alpha_beta_grid.shape[0], 1)) if initial_operator_values is None else initial_operator_values
    )

    last_H = deepcopy(initial_field)
    last_operator_values = deepcopy(initial_operator_values)

    positive_direction = H_trajectory[0] > initial_field

    def body(carry, H):
        positive_direction, initial_field, last_H, initial_operator_values, last_operator_values = carry

        # update initial_field and initial_operator_values based on sign change
        positive_direction, initial_field, initial_operator_values = update_state(H, carry)

        B_est_single, operator_values = model(
            H=H,
            initial_field=initial_field,
            initial_operator_values=initial_operator_values,
            alpha_beta_grid=alpha_beta_grid,
            T=T,
        )

        last_H = H
        last_operator_values = operator_values

        return (positive_direction, initial_field, last_H, initial_operator_values, last_operator_values), B_est_single

    _, B_est = jax.lax.scan(
        body, (positive_direction, initial_field, last_H, initial_operator_values, last_operator_values), H_trajectory
    )
    return B_est
