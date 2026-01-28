"""Generic RNN model implementations."""

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from rhmag.models.linear import LinearDynamicParameters, LinearStatic


class GRU(eqx.Module):
    """Basic gated recurrent unit (GRU) model."""

    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module

    def __init__(self, in_size: int, hidden_size: int, *, key):
        """Construct a basic Gated Recurrent Unit (GRU) based on the `equinox.nn.GRUCell`.

        Args:
            in_size (int): Number of input elements
            hidden_size (int): Number of hidden state elements
            key (jax.random.PRNGkey): Pseudo random number generation key for initialization of the
                model parameters
        """
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key)

    def __call__(self, input: jax.Array, init_hidden: jax.Array) -> jax.Array:
        """Use the GRU to roll over an input sequence.

        The first element of the hidden state is interpreted as the output of the GRU in each.

        NOTE: This function is not expecting a batch dimension. It is intended to vmapped over for
            batch-wise predictions!

        Args:
            input (jax.Array): Input sequence with shape (sequence_length, in_size)
            init_hidden (jax.Array): Initial vector for the hidden state with shape (hidden_size,)

        Returns:
            The output sequence as a jax.Array with shape (sequence_length, 1)
        """

        hidden = init_hidden

        def f(carry, inp):
            rnn_out = self.cell(inp, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, input)
        return out

    def warmup_call(self, input: jax.Array, init_hidden: jax.Array, out_true: jax.Array) -> jax.Array:
        """Warm up the hidden state of the GRU with an input sequence where the true outputs are known.

        The basic idea is to feed the true value into the first element of the hidden state in each step
        so that the output prediction does not diverge while the other elements of the GRU may change to
        get into shape to best predict the following sequence without known true values.

        Args:
            input (jax.Array): Input sequence with shape (sequence_length, in_size)
            init_hidden (jax.Array): Initial vector for the hidden state with shape (hidden_size,)
            out_true (jax.Array): The true output values with shape (sequence_length,)

        Returns:
            The outputs as jax.Array with shape (sequence_length,) (these will be the same as out_true) and
                the final warmed up hidden_state
        """
        hidden = init_hidden

        def f(carry, inp):
            inp_t, out_true_t = inp
            rnn_out = self.cell(inp_t, carry)
            rnn_out = rnn_out.at[0].set(out_true_t)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (input, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true: jax.Array, batch_size: int) -> jax.Array:
        """Put together the very first initial state. Concatenates the given true value with zeros."""
        return jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 1))])


class LSTM(eqx.Module):
    """Basic long short-term memory network (LSTM)."""

    hidden_size: int = eqx.field(static=True)
    cell: eqx.nn.LSTMCell

    def __init__(self, in_size: int, hidden_size: int, *, key):
        """Construct a basic long short-term memory network (LSTM) based on the `equinox.nn.LSTMCell`.

        Args:
            in_size (int): Number of input elements
            hidden_size (int): Number of hidden state elements
            key (jax.random.PRNGkey): Pseudo random number generation key for initialization of the
                model parameters
        """
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key)

    def __call__(self, input: jax.Array, init_hidden: tuple[jax.Array, jax.Array]) -> jax.Array:
        """Use the LSTM to roll over an input sequence.

        The first element of the hidden state is interpreted as the output of the GRU in each.

        NOTE: This function is not expecting a batch dimension. It is intended to vmapped over for
            batch-wise predictions!

        Args:
            input (jax.Array): Input sequence with shape (sequence_length, in_size)
            init_hidden (jax.Array): Initial vector for the hidden state with shape (hidden_size,)

        Returns:
            The output sequence as a jax.Array with shape (sequence_length, 1)
        """

        hidden = init_hidden

        def f(carry, inp):
            (hidden_state, cell_state) = self.cell(inp, carry)
            out = hidden_state[0]
            return (hidden_state, cell_state), out

        _, out = jax.lax.scan(f, hidden, input)
        return out

    def warmup_call(self, input: jax.Array, init_hidden: tuple[jax.Array, jax.Array], out_true: jax.Array) -> jax.Array:
        """Warm up the hidden state of the LSTM with an input sequence where the true outputs are known.

        Args:
            input (jax.Array): Input sequence with shape (sequence_length, in_size)
            init_hidden (jax.Array): Initial vector for the hidden state with shape (hidden_size,)
            out_true (jax.Array): The true output values with shape (sequence_length,)

        Returns:
            The outputs as jax.Array with shape (sequence_length,) (these will be the same as out_true) and
                the final warmed up hidden_state
        """
        hidden = init_hidden

        def f(carry, inp):
            inp_t, out_true_t = inp
            (hidden_state, cell_state) = self.cell(inp_t, carry)
            hidden_state = hidden_state.at[0].set(out_true_t)
            out = hidden_state[0]
            return (hidden_state, cell_state), out

        final_hidden, out = jax.lax.scan(f, hidden, (input, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true: jax.Array, batch_size: int) -> jax.Array:
        """Put together the very first initial state. Concatenates the given true value with zeros."""
        return (
            jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 1))]),
            jnp.zeros((batch_size, self.hidden_size)),
        )


class GRU2(eqx.Module):
    """Basic gated recurrent unit (GRU) model. All init hidden states set to gt."""

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
        out_true = jnp.squeeze(out_true)
        return jnp.broadcast_to(out_true[:, None], (batch_size, self.hidden_size))


class ExpGRU(GRU):

    def __call__(self, input, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            rnn_out = self.cell(inp, carry)
            rnn_out = rnn_out.at[..., -1].set(jnp.exp(rnn_out[..., -1]))

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
            rnn_out = rnn_out.at[..., -1].set(jnp.exp(rnn_out[..., -1]))

            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (input, out_true))
        return out, final_hidden


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
    linear: LinearStatic = eqx.field(static=True)
    linear_in_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, linear_in_size, *, key):
        self.hidden_size = hidden_size
        self.linear_in_size = linear_in_size

        gru_key, l_key = jax.random.split(key, 2)

        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=gru_key)
        # linear_dummy = LinearStatic(linear_in_size, out_size=1, key=l_key)
        from rhmag.utils.model_evaluation import reconstruct_model_from_exp_id

        self.linear = reconstruct_model_from_exp_id("3C90_Linear_a3943263-1c37-48").model
        assert self.linear.in_size == linear_in_size

    def __call__(self, input_gru, input_linear, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            inp_gru_t, inp_lin_t = inp

            rnn_out = self.cell(inp_gru_t, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)

            mu_bias = rnn_out_o[..., 0]

            out = self.linear.predict(inp_lin_t) + mu_bias
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, (input_gru, input_linear))
        return out

    def debug_call(self, input_gru, input_linear, init_hidden):
        hidden = init_hidden

        def f(carry, inp):
            inp_gru_t, inp_lin_t = inp

            rnn_out = self.cell(inp_gru_t, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)

            mu_bias = rnn_out_o[..., 0]

            linear_out = self.linear.predict(inp_lin_t)

            out = linear_out + mu_bias
            return rnn_out, (out, linear_out, rnn_out)

        _, (out, linear_out, rnn_out) = jax.lax.scan(f, hidden, (input_gru, input_linear))
        return out, linear_out, rnn_out

    def warmup_call(self, gru_in, linear_in, init_hidden, out_true):
        hidden = init_hidden

        def f(carry, inp):
            inp_gru_t, inp_lin_t, out_true_t = inp
            rnn_out = self.cell(inp_gru_t, carry)
            rnn_out = rnn_out.at[0:2].set(out_true_t)

            rnn_out_o = jnp.atleast_2d(rnn_out)
            mu_bias = rnn_out_o[..., 0]

            out = self.linear.predict(inp_lin_t) + mu_bias
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (gru_in, linear_in, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true, batch_size):
        return jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 1))])
