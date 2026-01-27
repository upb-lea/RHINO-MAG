import torch
import torch.nn as nn
from torch.nn import Parameter as TorchParam
from typing import List, Tuple, Optional


class Biased_Elu(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1


class SinusAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class GeneralizedCosinusUnit(nn.Module):
    def forward(self, x):
        return torch.cos(x) * x


ACTIVATION_FUNCS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "biased_elu": Biased_Elu,
    "sinus": SinusAct,
    "gcu": GeneralizedCosinusUnit,
}


class DifferenceEqLayer(nn.Module):
    """For discretized ODEs"""

    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        device = torch.device("cpu")
        self.cell = cell(*cell_args, **cell_kwargs).to(device)

    def forward(self, input_BTP, state_BQ):
        """
        This forward pass assumes output dim is the same as state dim

        Dim key:
        B: Batch size
        T: Time steps
        P: Input features
        Q: Output features (hidden state)
        """
        inputs_BP_l = input_BTP.unbind(1)
        outputs_BQ_l = []
        for in_BP in inputs_BP_l:
            out_BQ, state_BQ = self.cell(in_BP, state_BQ)
            outputs_BQ_l += [out_BQ]
        return torch.stack(outputs_BQ_l, dim=1), state_BQ


class ExplEulerCell(nn.Module):
    def __init__(self, n_inputs: int, layer_cfg=None):
        super().__init__()

        n_targets = 1

        # layer config init
        layer_default = {
            "f": [{"units": 16, "activation": "tanh"}, {"units": n_targets}],
        }
        self.layer_cfg = layer_cfg or layer_default

        # main sub NN
        f_layers = []
        f_units = n_inputs + n_targets

        for layer_specs in self.layer_cfg["f"]:
            lay = nn.Linear(f_units, layer_specs["units"])
            f_layers.append(lay)
            if layer_specs.get("activation", "linear") != "linear":
                f_layers.append(ACTIVATION_FUNCS[layer_specs["activation"]]())
            f_units = layer_specs["units"]
        self.f = nn.Sequential(*f_layers)

    def forward(self, inp, hidden):
        prev_out = hidden
        all_input = torch.cat([inp, hidden], dim=1)
        incr = self.f(all_input)
        msk = torch.abs(incr) > 1
        scaled_incr = incr**3 * msk.float() + (1 - msk.float()) * incr
        out = prev_out + scaled_incr * 1e-2
        return prev_out, out
