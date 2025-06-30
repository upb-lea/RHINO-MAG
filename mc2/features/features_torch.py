import numpy as np
import torch


# single features
def db_dt(b: torch.Tensor) -> torch.Tensor:
    """Calculate the first derivative of b."""
    return torch.gradient(b)[0]


def d2b_dt2(b: torch.Tensor) -> torch.Tensor:
    """Calculate the first derivative of b."""
    return torch.gradient(torch.gradient(b)[0])[0]


def dyn_avg(x: torch.Tensor, n_s: int) -> torch.Tensor:
    """Calculate the dynamic average of sequence x containing n_s samples."""
    return torch.tensor(np.convolve(x, np.ones(n_s) / n_s, mode="same"))


def pwm_of_b(b: torch.Tensor) -> torch.Tensor:
    """Calculate the pwm trigger sequence that created the flux density sequence b."""
    return torch.sign(db_dt(b))


# combined features
def add_fe(data: torch.Tensor, n_s: int) -> torch.Tensor:
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
    m, n = data.shape
    features = []

    for i in range(m):
        b = data[i]
        dyn = dyn_avg(b, n_s)
        db = db_dt(b)
        d2b = d2b_dt2(b)
        pwm = pwm_of_b(b)

        stacked = torch.stack((b, dyn, db, d2b, pwm), axis=-1)  # shape: (n, d)
        features.append(stacked)

    return torch.stack(features, axis=0)
