"""Model interface definition for models that are specifically tailored to predict
future H (magnetic field strength) based on future B (magnetic flux density) and
past values of H. (+ some utility functions)

Model implementations with this interface utilize the models from mc2.models to
predict/simulate trajectories for the MagNet Challenge 2, in adherence to the model
form specified in the challenge.
"""

from abc import abstractmethod
from typing import Type
import pathlib
import json

import numpy as np
import numpy.typing as npt

import jax
import equinox as eqx


class ModelInterface(eqx.Module):
    @abstractmethod
    def __call__(
        self,
        B_past: npt.NDArray[np.float64],
        H_past: npt.NDArray[np.float64],
        B_future: npt.NDArray[np.float64],
        T: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Model prediction interface for batched inputs, i.e. for inputs with an extra
        leading dimension.

        Args:
            B_past (np.array): The physical (non-normalized) flux density values from time
                step t0 to t1 with shape (n_batches, past_sequence_length)
            H_past (np.array): The physical (non-normalized) field values from time step
                t0 to t1 with shape (n_batches, past_sequence_length)
            B_future (np.array): The physical (non-normalized) flux density values from
                time step t1 to t2 with shape (n_batches, future_sequence_length)
            T (float): The temperature of the material with shape (n_batches,)

        Returns:
            H_future (np.array): The physical (non-normalized) field values from time
                step t1 to t2 with shape (n_batches, future_sequence_length)
        """
        pass

    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        pass


def save_model(filename: str | pathlib.Path, hyperparams: dict, model: ModelInterface):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(filename: str | pathlib.Path, model_class: Type[ModelInterface]):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_class(key=jax.random.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)


def count_model_parameters(model: ModelInterface) -> int:
    """Returns the number of Parameters in the model by summing the sizes of the jax.Arrays.
    That is, all parameters of the model must be jax.Arrays for this function to work!
    """
    return sum([p.size for p in jax.tree_leaves(eqx.filter(model, eqx.is_inexact_array, None))])
