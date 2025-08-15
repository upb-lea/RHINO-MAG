import jax
import jax.numpy as jnp
import equinox as eqx


class GRU(eqx.Module):
    """Very basic RNN model."""

    hidden_size: int = eqx.static_field()
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
