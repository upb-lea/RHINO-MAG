import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


class PinnWithGRU(eqx.Module):
    net: eqx.Module
    Ms: jnp.ndarray  # skalare Größen, die wir lernen wollen
    a: jnp.ndarray
    alpha: jnp.ndarray
    c: jnp.ndarray
    k: jnp.ndarray
    hidden_size: int = eqx.static_field()
    physics_weight_lambda: float = eqx.static_field()

    def __init__(self, input_size, hidden_size, physics_weight_lambda, *, key):
        init_key, key = jr.split(key)
        self.hidden_size = hidden_size
        self.physics_weight_lambda = physics_weight_lambda
        self.net = eqx.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, key=init_key)

        self.Ms = jnp.array([1.6e6], dtype=jnp.float32)
        self.a = jnp.array([110], dtype=jnp.float32)
        self.alpha = jnp.array([1.6e-3], dtype=jnp.float32)
        self.c = jnp.array([0.2], dtype=jnp.float32)
        self.k = jnp.array([400.01], dtype=jnp.float32)
        # self.out_layer = eqx.nn.Linear(in_features=1,out_features=1,key=init_key)

    def __call__(self, inp, init_hidden):
        hidden = init_hidden

        def scan_fn(carry, inp):
            gru_out = self.net(inp, carry)
            gru_out_o = jnp.atleast_2d(gru_out)
            out = gru_out_o[:, 0]
            return gru_out, out

        _, out = jax.lax.scan(scan_fn, hidden, inp)
        # out_o = self.out_layer(out)
        return out

    def warmup_call(self, input, init_hidden, out_true):
        hidden = init_hidden
        # TODO: move construct hidden here?

        def f(carry, inp):
            inp_t, out_true_t = inp
            rnn_out = self.net(inp_t, carry)
            rnn_out = rnn_out.at[0].set(out_true_t)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (input, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true, batch_size):
        return jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 1))])
