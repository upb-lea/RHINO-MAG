import jax
import jax.numpy as jnp


def get_energy_loss(b: jax.Array, h: jax.Array) -> jax.Array:
    """Compute the energy loss in a hysteresis loop.
    The energy loss is computed as the area of the hysteresis loop.

    Args:
        b (jax.Array): Magnetic flux density in T.
        h (jax.Array): Magnetic field strength in A/m.
    """
    return jnp.trapezoid(h, b)
