import numpy as np
import torch
import logging as log
import pandas as pd
from typing import List
from rhmag.data_management import CACHE_ROOT, NORMALIZATION_ROOT


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
    def __init__(self, rolling_window_size: int = 101, mat_lbl: str = "3C90", device: torch.device = None):
        self.n_inputs: int = None
        self.norm_consts_l: List = []
        self.norm_consts_BP: torch.Tensor = None
        self.rolling_window_size = rolling_window_size
        self.mat_lbl = mat_lbl
        self.device = device if device is not None else torch.device("cpu")

    @torch.no_grad()
    def extract_normalization_constants(self, data_MN_l: torch.Tensor) -> None:
        """Extract normalization constants from the input data."""
        log.info("Extract normalization constants..")

        # for each time series matrix, for each feature, extend min and max
        for data_B_MN, data_T_MI in data_MN_l:
            featurized_data_MN_l, _ = self.add_fe(data_B_MN, with_original=True, temperature_MI=data_T_MI)
            if self.norm_consts_BP is None:
                self.norm_consts_BP = torch.zeros((2, self.n_inputs), dtype=torch.float32)
            for i, feat_data_MN in enumerate(featurized_data_MN_l):
                data_max = feat_data_MN.max()
                data_min = feat_data_MN.min()
                self.norm_consts_BP[0, i] = min(self.norm_consts_BP[0, i], data_min)
                self.norm_consts_BP[1, i] = max(self.norm_consts_BP[1, i], data_max)
        norm_const_save_path = NORMALIZATION_ROOT / f"{self.mat_lbl}_normalization_constants.parquet"
        pd.DataFrame(self.norm_consts_BP.cpu().numpy()).to_parquet(norm_const_save_path, index=False)
        log.info(f"Normalization constants saved to {norm_const_save_path}")

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
        :return: 2-tuple: list of 2d tensors, and dbdt tensor MxN
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
        return fe_l, db_dt_MN


# Source: https://github.com/pytorch/pytorch/issues/50334
def interp(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int = -1, extrapolate: str = "constant"
) -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp.ravel(), x, right=False)

    if extrapolate == "constant":
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.ravel()[indices] * x + b.ravel()[indices]

    return values.movedim(-1, dim)

class MC2Loss(torch.nn.Module):
    """Custom loss function for the Magnet Challenge 2.
    This computes the RMSE, but all timesteps are weighted by db/dt and all whole time series are weighted by the RMSE of the H ground truth.
    These weightings are to accommodate the competition metrics, sequence relative error and normalized energy relative error."""
    def __init__(self):
        super().__init__()
        self.mse_loss_fn = torch.nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, dbdt: torch.Tensor) -> torch.Tensor:
        pred_mse_BQ = self.mse_loss_fn(pred, target)
        H_rmse_B = torch.sqrt((target.squeeze()**2).mean(dim=1))
        abs_dbdt_BQ = torch.abs(dbdt)
        weighted_pred_rmse_B = torch.sqrt((pred_mse_BQ * abs_dbdt_BQ).mean(dim=1))
        weighted_pred_rmse_B = weighted_pred_rmse_B / (H_rmse_B + 1e-12)
        return weighted_pred_rmse_B.mean()  # mean over batch