import jax
import jax.numpy as jnp
import equinox as eqx


def db_dt(b: jax.Array) -> jax.Array:
    """Calculate the first derivative of b."""
    return jnp.gradient(b)


def d2b_dt2(b: jax.Array) -> jax.Array:
    """Calculate the first derivative of b."""
    return jnp.gradient(jnp.gradient(b))


def dyn_avg(x: jax.Array, n_s: int) -> jax.Array:
    """Calculate the dynamic average of sequence x containing n_s samples."""
    return jnp.convolve(x, jnp.ones(n_s) / n_s, mode="same")


def pwm_of_b(b: jax.Array) -> jax.Array:
    """Calculate the pwm trigger sequence that created the flux density sequence b."""
    return jnp.sign(db_dt(b))


def compute_fe_single(data_single: jax.Array, n_s: int) -> jax.Array:
    dyn = dyn_avg(data_single, n_s)
    db = db_dt(data_single)
    d2b = d2b_dt2(data_single)
    pwm = pwm_of_b(data_single)

    return jnp.stack((data_single, dyn, db, d2b, pwm), axis=-1)


@eqx.filter_jit
def add_fe(data: jax.Array, n_s: int) -> jax.Array:
    """
    Apply feature engineering to each sequence (row) of a 2D matrix.

    d = 5 Features computed:
      - original b
      - dynamic average of b
      - db/dt (first derivative)
      - d²b/dt² (second derivative)
      - PWM of b (sign of db/dt)

    :param data: m x n array (m sequences of length n)
    :param n_s: Number of samples for the dynamic average
    :return: m x n x d array with stacked features
    """
    assert data.ndim == 2, "Input must be a 2D array (m x n)"
    return jax.vmap(compute_fe_single, in_axes=(0, None))(data, n_s)
