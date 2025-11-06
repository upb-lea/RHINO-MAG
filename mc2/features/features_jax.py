import jax
import jax.numpy as jnp
import equinox as eqx

F_SAMPLE = 16e6


def db_dt(b: jax.Array) -> jax.Array:
    """Calculate the first derivative of b."""
    return jnp.gradient(b)


def d2b_dt2(b: jax.Array) -> jax.Array:
    """Calculate the first derivative of b."""
    return jnp.gradient(jnp.gradient(b))


def dyn_avg(x: jax.Array, n_s: int, mirrored_padding: bool = True) -> jax.Array:
    """Calculate the dynamic average of sequence x containing n_s samples."""

    if mirrored_padding:
        convolution_mode = "valid"
        assert (n_s - 1) % 2 == 0, "n_s must be an odd number"
        p_len = (n_s - 1) // 2
        x = jnp.pad(x, ((p_len, p_len)), mode="reflect", reflect_type="odd")
    else:
        convolution_mode = "same"
        x = x

    return jnp.convolve(x, jnp.ones(n_s) / n_s, mode=convolution_mode)


def shift_signal(x, k_0):
    x_padded = jnp.pad(x, ((jnp.abs(k_0), jnp.abs(k_0))), mode="reflect", reflect_type="odd")
    x_shifted = jnp.roll(x_padded, -k_0)
    return x_shifted[k_0:-k_0]


def pwm_of_b(b: jax.Array) -> jax.Array:
    """Calculate the pwm trigger sequence that created the flux density sequence b."""
    return jnp.sign(db_dt(b))


def find_peaks_jax(x, height=None):
    left = jnp.roll(x, 1)
    right = jnp.roll(x, -1)

    peaks = (x > left) & (x > right)
    peaks = peaks.at[0].set(False)
    peaks = peaks.at[-1].set(False)

    if height is not None:
        peaks &= x >= height

    max_peaks = x.shape[0]
    idx = jnp.where(peaks, size=max_peaks, fill_value=-1)[0]

    return idx, None


def compute_intervals(idx, fs):
    valid_mask = idx >= 0
    idx_filled = jnp.where(valid_mask, idx, 0)
    valid_intervals_mask = valid_mask[1:] & valid_mask[:-1]
    intervals = jnp.diff(idx_filled) / fs
    intervals = jnp.where(valid_intervals_mask, intervals, jnp.nan)
    return intervals


def get_frequency(signal, fs):
    min_peak_height_pos = 0.8 * jnp.max(signal)
    min_peak_height_neg = 0.8 * jnp.max(-signal)

    pos_idx, _ = find_peaks_jax(signal, height=min_peak_height_pos)
    neg_idx, _ = find_peaks_jax(-signal, height=min_peak_height_neg)

    pos_intervals = compute_intervals(pos_idx, fs)
    neg_intervals = compute_intervals(neg_idx, fs)

    median_interval_pos = jnp.nanmedian(pos_intervals)
    median_interval_neg = jnp.nanmedian(neg_intervals)

    median_interval = jax.lax.cond(
        jnp.nanvar(pos_intervals) < jnp.nanvar(neg_intervals),
        lambda _: median_interval_pos,
        lambda _: median_interval_neg,
        operand=None,
    )

    median_interval = jnp.nan_to_num(median_interval, nan=1 / (50_000))

    frequency = 1.0 / median_interval
    return frequency


def compute_fe_single(data_single: jax.Array, n_s: int) -> jax.Array:
    dyn = dyn_avg(data_single, n_s)
    db = db_dt(data_single)
    d2b = d2b_dt2(data_single)
    pwm = pwm_of_b(data_single)
    # f = get_frequency(d2b, F_SAMPLE)
    # f_repeated = jnp.full(pwm.shape, f)
    return jnp.stack((data_single, dyn, db, d2b, pwm), axis=-1)  # , f_repeated), axis=-1)


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
