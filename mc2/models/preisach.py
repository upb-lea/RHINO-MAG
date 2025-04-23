"""Differentiable Preisach model in JAX based on https://github.com/roussel-ryan/diff_hysteresis.

WARNING: This code implements H to B since this is what is usually found in the literature.
We will need to invert the model to get B to H in the future.
"""

from copy import deepcopy
import numpy as np

import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx

from mc2.models.preisach_utils import analyticalPreisachFunction2, preisachIntegration, filter_function, filter_grid


class HysteronDensityMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size: int, depth: int, *, key, **kwargs):
        """Simple comfort wrapper for MLP specifically for the hysteron density."""

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
def hysteron_operator(
    H: jax.Array,
    last_H: jax.Array,
    initial_field: jax.Array,
    initial_output: jax.Array,
    alpha_beta: jax.array,
    T: float,
) -> jax.Array:
    """Differentiable hysteron operator.

    TODO: The behavior of the operator is bit strange the saturation is not fully reached.
    With very low T values this is not really a problem, since it approximates the standard
    hysteron operator for this case. I am not sure if this is a problem or if we should just
    use the standard hysteron operator in the first place.

    Args:
        H (jax.Array): Current field value.
        last_H (jax.Array): Last field value.
        initial_field (jax.Array): Initial field value after the last direction change of H.
        initial_output (jax.Array): Initial output value after the last direction change of H.
        alpha_beta (jax.Array): Discretized grid in the alpha-beta plane.
        T (float): Scaling factor that governs the steepness of the transition.

    Returns:
        hysteron_operator_value (jax.Array): Hysteron operator value.

    """
    alpha = alpha_beta[0]
    beta = alpha_beta[1]

    def true_fun(H, last_H, initial_field, initial_output, alpha, beta, T):
        return last_H

    def false_fun(H, last_H, initial_field, initial_output, alpha, beta, T):
        def _true_fun(H, initial_output, alpha, beta, T):
            return jax.lax.min(initial_output + (1 + jnp.tanh((H - alpha) / jnp.abs(T))), jnp.array([1.0]))

        def _false_fun(H, initial_output, alpha, beta, T):
            return jax.lax.max(initial_output - (1 + jnp.tanh(-(H - beta) / jnp.abs(T))), jnp.array([-1.0]))

        return jax.lax.cond((H > initial_field)[0], _true_fun, _false_fun, H, initial_output, alpha, beta, T)

    return jax.lax.cond(
        (H == initial_field)[0], true_fun, false_fun, H, last_H, initial_field, initial_output, alpha, beta, T
    )


class DifferentiablePreisach(eqx.Module):
    hysteron_density: HysteronDensityMLP
    A: jax.Array

    def __init__(self, width_size, depth, *, model_key, **kwargs):
        super().__init__(**kwargs)

        # poly_params, nn_key = jax.random.split(model_key)
        # self.A = jax.random.uniform(poly_params, shape=(3,), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        self.A = jnp.array([10.0, 0.0, 0.0], dtype=jnp.float32)

        self.hysteron_density = HysteronDensityMLP(
            width_size=width_size,
            depth=depth,
            key=model_key,
        )

    @eqx.filter_jit
    def __call__(self, H, last_H, initial_field, initial_operator_values, alpha_beta_grid, T=1e-3):
        hysteron_density_values = jax.vmap(self.hysteron_density)(alpha_beta_grid)
        hysteron_operator_values = jax.vmap(hysteron_operator, in_axes=(None, None, None, 0, 0, None))(
            H, last_H, initial_field, initial_operator_values, alpha_beta_grid, T
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

    @classmethod
    def from_parameters(
        cls,
        points_per_dim: int,
        dim: int = 2,
        low: float = -1,
        high: float = 1,
        A: float = 1,
        Hc: float = 0.01,
        sigma: float = 0.03,
    ) -> tuple["ArrayPreisach", jax.Array]:
        """Create a Preisach model from parameters."""
        xs = [jnp.linspace(low, high, points_per_dim) for _ in range(dim)]
        alpha_grid, beta_grid = jnp.meshgrid(*xs)

        alpha_beta_grid = jnp.concatenate([alpha_grid.flatten()[..., None], beta_grid.flatten()[..., None]], axis=-1)

        preisach = analyticalPreisachFunction2(
            A=A, Hc=Hc, sigma=sigma, beta=np.array(beta_grid), alpha=np.array(alpha_grid)
        )
        preisach = preisachIntegration(w=2 * 1 / (points_per_dim - 1), Z=preisach)

        preisach = preisach / jnp.max(preisach)
        preisach = jnp.fliplr(preisach)

        preisach = preisach.flatten()
        valid_points = jax.vmap(filter_function)(alpha_beta_grid) == 0
        preisach = preisach[jnp.where(valid_points == True)][:, None]
        preisach = jnp.array(preisach)

        alpha_beta_grid = filter_grid(alpha_beta_grid)

        return cls(preisach), alpha_beta_grid

    @eqx.filter_jit
    def __call__(self, H, last_H, initial_field, initial_operator_values, alpha_beta_grid, T=1e-3):
        hysteron_density_values = self.hysteron_density
        hysteron_operator_values = jax.vmap(hysteron_operator, in_axes=(None, None, None, 0, 0, None))(
            H, last_H, initial_field, initial_operator_values, alpha_beta_grid, T
        )

        est_B = jnp.mean(hysteron_density_values * hysteron_operator_values)[None]
        est_B = self.A[0] * est_B + self.A[1] * H + self.A[2]

        return est_B, hysteron_operator_values


@eqx.filter_jit
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


@eqx.filter_jit
def estimate_B(H_trajectory, model, alpha_beta_grid, initial_field=None, initial_operator_values=None, T=1e-3):
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
            last_H=last_H,
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
