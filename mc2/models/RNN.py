from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.models.linear import LinearDynamicParameters, LinearStatic


class GRU(eqx.Module):
    """Basic gated recurrent unit (GRU) model."""

    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module

    def __init__(self, in_size, hidden_size, *, key):
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key)

    def __call__(self, input, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            rnn_out = self.cell(inp, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, input)
        return out

    def warmup_call(self, input, init_hidden, out_true):
        hidden = init_hidden
        # TODO: move construct hidden here?

        def f(carry, inp):
            inp_t, out_true_t = inp
            rnn_out = self.cell(inp_t, carry)
            rnn_out = rnn_out.at[0].set(out_true_t)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (input, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true, batch_size):
        return jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 1))])


class VectorfieldGRU(eqx.Module):
    n_locs: int = eqx.field(static=True)
    cell: eqx.Module

    @property
    def hidden_size(self):
        return 2 * self.n_locs

    def __init__(self, in_size, n_locs, *, key):
        self.n_locs = n_locs
        self.cell = eqx.nn.GRUCell(in_size, 2 * n_locs, key=key)

    def __call__(self, input, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            rnn_out = self.cell(inp, carry)
            return rnn_out, rnn_out

        _, out = jax.lax.scan(f, hidden, input)

        if out.ndim == 1:
            out = out.reshape((self.n_locs, 2))
        elif out.ndim == 2:
            out = out.reshape((-1, self.n_locs, 2))

        return out

    def warmup_call(self, input, init_hidden, out_true):
        raise NotImplementedError()

        hidden = init_hidden
        # TODO: move construct hidden here?

        def f(carry, inp):
            inp_t, out_true_t = inp
            rnn_out = self.cell(inp_t, carry)
            return rnn_out, rnn_out

        final_hidden, out = jax.lax.scan(f, hidden, (input, out_true))

        if out.ndim == 1:
            out = out.reshape((self.n_locs, 2))
        elif out.ndim == 2:
            out = out.reshape((-1, self.n_locs, 2))

        return out, final_hidden

    def construct_init_hidden(self, out_true, batch_size):
        return jnp.hstack([jnp.zeros((batch_size, self.hidden_size))])


class GRUwLinear(eqx.Module):

    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jax.random.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            rnn_out = self.cell(inp, carry)
            out = self.linear(rnn_out) + self.bias
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, input)
        return out


class GRUwLinearModel(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module
    linear: LinearDynamicParameters
    linear_in_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, linear_in_size, *, key):
        self.hidden_size = hidden_size
        self.linear_in_size = linear_in_size

        assert linear_in_size <= hidden_size, (
            "The linear_in_size must be smaller or equal to the hidden_size"
            + f"given values are linear_in_size={linear_in_size} > hidden_size={hidden_size}."
        )

        gru_key, l_key = jax.random.split(key, 2)

        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=gru_key)
        self.linear = LinearDynamicParameters(in_size, out_size=1, key=l_key)

    def __call__(self, input_gru, input_linear, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            inp_gru, inp_linear = inp[..., : -self.linear_in_size], inp[..., -self.linear_in_size :]

            rnn_out = self.cell(inp_gru, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)

            # take the first self.linear_in_size hidden states as the parameters for the linear model
            linear_params = rnn_out_o[..., : self.linear_in_size]
            out = self.linear.predict(inp_linear, linear_params)
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, jnp.concatenate([input_gru, input_linear], axis=-1))
        return out

    def warmup_call(self, gru_in, linear_in, init_hidden, out_true):
        hidden = init_hidden

        def f(carry, inp):
            inp_gru_t, inp_lin_t, out_true_t = inp
            rnn_out = self.cell(inp_gru_t, carry)
            rnn_out = rnn_out.at[0].set(out_true_t)

            rnn_out_o = jnp.atleast_2d(rnn_out)
            linear_params = rnn_out_o[..., : self.linear_in_size]
            out = self.linear.predict(inp_lin_t, linear_params)
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (gru_in, linear_in, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true, batch_size):
        return jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 1))])


class GRUaroundLinearModel(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module
    linear: LinearStatic
    linear_in_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, linear_in_size, *, key):
        self.hidden_size = hidden_size
        self.linear_in_size = linear_in_size

        gru_key, l_key = jax.random.split(key, 2)

        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=gru_key)
        self.linear = LinearStatic(linear_in_size, out_size=1, key=l_key)

    def __call__(self, input_gru, input_linear, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            inp_gru_t, inp_lin_t = inp

            rnn_out = self.cell(inp_gru_t, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)

            mu_scale = rnn_out_o[..., 0]
            mu_bias = rnn_out_o[..., 1]

            out = mu_scale * self.linear.predict(inp_lin_t) + mu_bias
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, (input_gru, input_linear))
        return out

    def warmup_call(self, gru_in, linear_in, init_hidden, out_true):
        hidden = init_hidden

        def f(carry, inp):
            inp_gru_t, inp_lin_t, out_true_t = inp
            rnn_out = self.cell(inp_gru_t, carry)
            rnn_out = rnn_out.at[0:2].set(out_true_t)

            rnn_out_o = jnp.atleast_2d(rnn_out)

            mu_scale = rnn_out_o[..., 0]
            mu_bias = rnn_out_o[..., 1]

            out = mu_scale * self.linear.predict(inp_lin_t) + mu_bias
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (gru_in, linear_in, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true, batch_size):
        return jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 2))])
