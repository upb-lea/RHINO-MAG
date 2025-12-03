import jax
import jax.numpy as jnp

import equinox as eqx


class LinearStatic(eqx.Module):
    in_size: int = eqx.field(static=True)
    # feat_in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    theta: jax.Array
    b: jax.Array

    # def __init__(self, in_size, feat_in_size, out_size, *, key):
    def __init__(self, in_size, out_size, *, key):

        assert in_size % 2 == 1
        self.in_size = in_size
        self.out_size = out_size
        # self.feat_in_size = feat_in_size

        l_key, b_key = jax.random.split(key, 2)

        self.theta = jax.random.normal(l_key, shape=(out_size, in_size))  # feat_in_size))
        self.b = jax.random.normal(b_key, shape=(out_size,))

    def predict(self, input):
        return jax.nn.tanh(self.theta @ input + self.b)

    def __call__(self, inputs):
        return eqx.filter_vmap(self.predict)(inputs)


class LinearDynamicParameters(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size, out_size, *, key):
        assert in_size % 2 == 1
        self.in_size = in_size
        self.out_size = out_size

    def predict(self, input, theta):
        return jax.nn.tanh(theta @ input)
