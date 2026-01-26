from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx


class DummyModel(eqx.Module):
    theta: jax.Array

    def __init__(self, key: jax.random.PRNGKey):
        self.theta = jax.random.normal(key, shape=()) * 20

    def __call__(self, x):
        return self.theta * jnp.ones(x.shape)

