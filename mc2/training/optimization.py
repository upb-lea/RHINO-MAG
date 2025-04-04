"""From batches + parameterizable model to parameter updates.

Admittedly, it could be very specific for each model, but this is the starting point on
which to build. Might be fused with the models themselves sooner or later.
"""

from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

if TYPE_CHECKING:
    import optax


@eqx.filter_value_and_grad
def compute_MSE_loss(model: eqx.Module, x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes the mean squared error between the model predictions and the given ground truth values.

    This function is decorated with `@eqx.filter_value_and_grad` to enable automatic differentiation.
    The batch dimension is expected to be the first dimension of the input data and it is dealt with
    by using `jax.vmap` to vectorize the model evaluation across the batch.

    Args:
        model(eqx.Module): The model to be evaluated.
        x(jax.Array): Input data (will generally be B values).
        y(jax.Array): Ground truth values (will generally be H values).
    """
    pred_y = jax.vmap(model)(x)
    return jnp.mean((pred_y - y) ** 2)


@eqx.filter_jit
def make_step(
    model: eqx.Module, x: jax.Array, y: jax.Array, optim: optax.GradientTransformation, opt_state: optax.OptState
) -> Tuple[jax.Array, eqx.Module, optax.OptState]:
    """Performs a single optimization step.

    This function computes the gradients of the MSE loss with respect to the model parameters,
    updates the model parameters using the optimizer, and returns the updated model and optimizer state.
    Args:
        model(eqx.Module): The model to be optimized.
        x(jax.Array): Input data (will generally be B values).
        y(jax.Array): Ground truth values (will generally be H values).
        optim(optax.GradientTransformation): The optimizer to be used.
        opt_state(optax.OptState): The current state of the optimizer.

    Returns:
        Tuple[jax.Array, eqx.Module, optax.OptState]: The loss value, updated model, and updated optimizer state.
    """
    loss, grads = compute_MSE_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
