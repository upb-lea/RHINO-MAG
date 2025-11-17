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

    def __init__(self, input_size, hidden_size, *, key):
        init_key, key = jr.split(key)
        self.hidden_size = hidden_size
        self.net = eqx.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, key=init_key)

        self.Ms = jnp.array([1.6e6], dtype=jnp.float32)
        self.a = jnp.array([110], dtype=jnp.float32)
        self.alpha = jnp.array([1.6e-3], dtype=jnp.float32)
        self.c = jnp.array([0.2], dtype=jnp.float32)
        self.k = jnp.array([400.01], dtype=jnp.float32)
        # self.out_layer = eqx.nn.Linear(in_features=1,out_features=1,key=init_key)

    def __call__(self, inp):
        hidden = jnp.zeros(self.net.hidden_size)

        def scan_fn(carry, inp):
            gru_out = self.net(inp, carry)
            gru_out_o = jnp.atleast_2d(gru_out)
            out = gru_out_o[:, 0]
            return gru_out, out

        _, out = jax.lax.scan(scan_fn, hidden, inp)
        # out_o = self.out_layer(out)
        return out
