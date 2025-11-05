import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.models.linear import LinearDynamicParameters


class GRU(eqx.Module):
    """Very basic RNN model."""

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
            out = rnn_out_o[:, 0]
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, input)
        return out

    def warmup_call(self, input, init_hidden, H_true):
        hidden = init_hidden

        def f(carry, inp):
            inp_t, h_true_t = inp
            rnn_out = self.cell(inp_t, carry)
            rnn_out = rnn_out.at[0].set(h_true_t)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[:, 0]
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (input, H_true))
        return out, final_hidden


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

    def __call__(self, input_GRU, input_linear, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            inp_GRU, inp_linear = inp[..., : -self.linear_in_size], inp[..., -self.linear_in_size :]

            rnn_out = self.cell(inp_GRU, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)

            # take the first self.linear_in_size hidden states as the parameters for the linear model
            linear_params = rnn_out_o[..., : self.linear_in_size]
            out = self.linear.predict(inp_linear, linear_params)
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, jnp.concatenate([input_GRU, input_linear], axis=-1))
        return out
