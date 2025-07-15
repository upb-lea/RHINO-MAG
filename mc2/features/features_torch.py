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
@torch.no_grad()
def add_fe(data_MN: torch.Tensor, n_s: int) -> torch.Tensor:
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
    assert data_MN.ndim == 2, "Input must be a 2D array (m x n)"
    db_dt_MN = torch.gradient(data_MN, dim=1)[0]  # db/dt
    kernel = torch.ones(1, 1, n_s).to(torch.float64) / n_s

    featurized_data_MND = torch.stack(
        [
            data_MN,
            db_dt_MN,
            torch.gradient(db_dt_MN, dim=1)[0],  # d²b/dt²
            torch.conv1d(data_MN[:, None, :], kernel, padding="same").squeeze(),  # moving average
            torch.sign(db_dt_MN),  # PWM of b
        ],
        dim=-1,
    )
    return featurized_data_MND  # shape: (M, N, D)
