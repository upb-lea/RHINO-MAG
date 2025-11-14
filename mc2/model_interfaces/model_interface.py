"""Model interface definition for models that are specifically tailored to predict
future H (magnetic field strength) based on future B (magnetic flux density) and
past values of H. (+ some utility functions)

Model implementations with this interface utilize the models from mc2.models to
predict/simulate trajectories for the MagNet Challenge 2, in adherence to the model
form specified in the challenge.
"""

from functools import partial
from abc import abstractmethod
from typing import Type, Callable
import pathlib
import json

import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp
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

    @abstractmethod
    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """
        Model prediction interface for normalized, batched inputs, i.e. for inputs with
        an extra leading dimension.

        Args:
            B_past_norm (np.array): The normalized flux density values from time step t0
                to t1 with shape (n_batches, past_sequence_length)
            H_past_norm (np.array): The normalized field values from time step t0 to t1
                with shape (n_batches, past_sequence_length)
            B_future_norm (np.array): The physical normalized flux density values from
                time step t1 to t2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)

        Returns:
            H_future_norm (np.array): The normalized field values from time step t1 to t2
                with shape (n_batches, future_sequence_length)
        """
        pass

    @property
    def n_params(self):
        if hasattr(self, "normalizer"):
            n_params_fe = len(self.normalizer.norm_fe_max)
        return n_params_fe + count_model_parameters(self)


def save_model(filename: str | pathlib.Path, hyperparams: dict, model: ModelInterface):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def filter_spec(f, leaf, f64_enabled):
    """Helper function for proper model loading under varying array precision.

    When a model is saved with float32 arrays and one attempts to load it in a float64
    context (and vice-versa), the loading crashes. This helper transfers the arrays to
    the proper dtypes.

    - float64 is enabled -> convert float32 arrays to float64
    - float64 is disabled -> convert float64 arrays to float32
    """
    problematic_dtype = jnp.float32 if f64_enabled else jnp.float64
    target_dtype = jnp.float64 if f64_enabled else jnp.float32

    if isinstance(leaf, jax.Array):
        loaded_leaf = jnp.load(f)
        if loaded_leaf.dtype == problematic_dtype:
            return loaded_leaf.astype(target_dtype)
        else:
            return loaded_leaf
    else:
        return eqx.default_deserialise_filter_spec(f, leaf)


def load_model(filename: str | pathlib.Path, model_class: Type[ModelInterface]):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_class(key=jax.random.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model, partial(filter_spec, f64_enabled=jax.config.x64_enabled))


def count_model_parameters(model: ModelInterface, filter: Callable = eqx.is_array_like) -> int:
    """Returns the number of Parameters in the model by summing the sizes of the jax.Arrays.
    That is, all parameters of the model must be jax.Arrays for this function to work!
    """
    return sum([p.size if hasattr(p, "size") else 1 for p in jax.tree_leaves(eqx.filter(model, filter, None))])
