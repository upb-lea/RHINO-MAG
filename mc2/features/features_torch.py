import numpy as np
import torch
import logging as log
from typing import List


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


class Featurizer:
    def __init__(self, rolling_window_size: int = 101):
        self.n_inputs: int = None
        self.norm_consts_l: List = []
        self.norm_consts_BP: torch.Tensor = None
        self.rolling_window_size = rolling_window_size

    @torch.no_grad()
    def extract_normalization_constants(self, data_MN_l: torch.Tensor) -> None:
        """Extract normalization constants from the input data."""
        log.info("Extract normalization constants..")

        # for each time series matrix, for each feature, extend min and max
        for data_B_MN, data_T_MI in data_MN_l:
            featurized_data_MN_l = self.add_fe(data_B_MN, with_original=True, temperature_MI=data_T_MI)
            if self.norm_consts_BP is None:
                self.norm_consts_BP = torch.zeros((2, self.n_inputs), dtype=torch.float32)
            for i, feat_data_MN in enumerate(featurized_data_MN_l):
                data_max = feat_data_MN.max()
                data_min = feat_data_MN.min()
                self.norm_consts_BP[0, i] = min(self.norm_consts_BP[0, i], data_min)
                self.norm_consts_BP[1, i] = max(self.norm_consts_BP[1, i], data_max)

    @torch.no_grad()
    def normalize(self, data_MN_l: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize the input data using the previously extracted constants."""
        assert self.norm_consts_l is not None, (
            "Normalization constants not extracted. Call extract_normalization_constants first."
        )
        normalized_data_MN_l = [
            2 * (data_MN - self.norm_consts_BP[0, i]) / (self.norm_consts_BP[1, i] - self.norm_consts_BP[0, i]) - 1
            for i, data_MN in enumerate(data_MN_l)
        ]
        return normalized_data_MN_l

    # combined features
    @torch.no_grad()
    def add_fe(self, data_MN: torch.Tensor, with_original=True, temperature_MI=None) -> torch.Tensor:
        """
        Apply feature engineering to each sequence (row) of a 2D matrix.

        d = 5 Features computed:
        - original b
        - dynamic average of b
        - db/dt (first derivative)
        - d²b/dt² (second derivative)
        - PWM of b (sign of db/dt)

        :param data: m x n array (m sequences of length n)
        :return: list of 2d tensors
        """
        assert data_MN.ndim == 2, "Input must be a 2D array (m x n)"
        db_dt_MN = torch.gradient(data_MN, dim=1)[0]  # db/dt
        kernel = (
            torch.ones(1, 1, self.rolling_window_size, dtype=data_MN.dtype, device=data_MN.device)
            / self.rolling_window_size
        )

        if with_original:
            fe_l = [data_MN]
        else:
            fe_l = []

        fe_l += [
            db_dt_MN,
            torch.gradient(db_dt_MN, dim=1)[0],  # d²b/dt²
            torch.conv1d(data_MN[:, None, :], kernel, padding="same").squeeze(),  # moving average
            torch.sign(db_dt_MN),  # PWM of b
        ]

        if temperature_MI is not None:
            fe_l.append(temperature_MI.repeat(1, data_MN.shape[1])[..., None])
        if self.n_inputs is None:
            self.n_inputs = len(fe_l)
        return fe_l
