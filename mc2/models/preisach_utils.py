import numpy as np

import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx


def build_grid(dim, low, high, points_per_dim):
    """Build a uniform grid of points in the given dimension."""
    xs = [jnp.linspace(low, high, points_per_dim) for _ in range(dim)]

    x_g = jnp.meshgrid(*xs)
    x_g = jnp.stack([_x for _x in x_g], axis=-1)
    x_g = x_g.reshape(-1, dim)

    assert x_g.shape[0] == points_per_dim**dim
    return x_g


def filter_function(x):
    return jnn.relu(x[1] - x[0])


def filter_grid(x):
    valid_points = jax.vmap(filter_function)(x) == 0
    return x[jnp.where(valid_points == True)]


def build_alpha_beta_grid(points_per_dim):
    return filter_grid(build_grid(2, -1, 1, points_per_dim))


def analyticalPreisachFunction2(A: float, Hc: float, sigma: float, beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    From https://github.com/fddf22/Preisachmodel

    Function based on Paper 'Removing numerical instabilities in the Preisach model identification
    using genetic algorithms' by G. Consolo G. Finocchio, M. Carpentieri, B. Azzerboni.
    """
    nom1 = 1
    den1 = 1 + ((beta - Hc) * sigma / Hc) ** 2
    nom2 = 1
    den2 = 1 + ((alpha + Hc) * sigma / Hc) ** 2
    preisach = A * (nom1 / den1) * (nom2 / den2)
    # set lower right diagonal to zero
    for i in range(preisach.shape[0]):
        preisach[i, (-i - 1) :] = 0
    return preisach


def preisachIntegration(w: float, Z: np.ndarray) -> np.ndarray:
    """
    From https://github.com/fddf22/Preisachmodel

    Perform 2D- integration of the Preisach distribution function.
    """
    flipped = np.fliplr(np.flipud(w * Z))
    flipped_integral = np.cumsum(np.cumsum(flipped, axis=0), axis=1)
    return np.fliplr(np.flipud(flipped_integral))


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
