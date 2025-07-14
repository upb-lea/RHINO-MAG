from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp
import equinox as eqx


class ModelInterface(ABC):

    @abstractmethod
    def __call__(
        self,
        B_past: npt.NDArray[np.float64],
        H_past: npt.NDArray[np.float64],
        B_future: npt.NDArray[np.float64],
        temperature: float,
    ) -> npt.NDArray[np.float64]:
        """Model prediction interface according to the challenge description.

        Args:
            B_past (np.array): The physical (non-normalized) flux density values from time
                step t0 to t1 with shape (past_sequence_length,)
            H_past (np.array): The physical (non-normalized) field values from time step
                t0 to t1 with shape (past_sequence_length,)
            B_future (np.array): The physical (non-normalized) flux density values from
                time step t1 to t2 with shape (future_sequence_length,)
            temperature (float): The temperature of the material as a scalar

        Returns:
            H_future (np.array): The physical (non-normalized) field values from time
                step t1 to t2 with shape (future_sequence_length,)
        """
        pass

    @abstractmethod
    def batched_prediction(
        self,
        B_past: npt.NDArray[np.float64],
        H_past: npt.NDArray[np.float64],
        B_future: npt.NDArray[np.float64],
        temperature: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Model prediction interface for batched inputs, i.e. for inputs with an extra
        leading dimension.

        Args:
            B_past (np.array): The physical (non-normalized) flux density values from time
                step t0 to t1 with shape (n_batches, past_sequence_length)
            H_past (np.array): The physical (non-normalized) field values from time step
                t0 to t1 with shape (n_batches, past_sequence_length)
            B_future (np.array): The physical (non-normalized) flux density values from
                time step t1 to t2 with shape (n_batches, past_sequence_length)
            temperature (float): The temperature of the material with shape (n_batches,)

        Returns:
            H_future (np.array): The physical (non-normalized) field values from time
                step t1 to t2 with shape (n_batches, past_sequence_length)
        """
        pass


class NODEwInterface(ModelInterface):

    def __init__(self, model, normalize, denormalize, featurize):
        self.model = model
        self._normalize = normalize
        self._denormalize = denormalize
        self._featurize = featurize

    @eqx.filter_jit
    def apply_model(
        self,
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        temperature: jax.Array,
    ) -> jax.Array:

        past_length = B_past.shape[0]

        norm_B, norm_H_past, norm_temperature = self._normalize(jnp.hstack([B_past, B_future]), H_past, temperature)

        norm_B_past = norm_B[:past_length]
        norm_B_future = norm_B[past_length:]

        featurized_input = self._featurize(norm_B_past, norm_H_past, norm_B_future, norm_temperature)

        _, norm_H_future = self.model(norm_H_past[-1], featurized_input, tau=1)
        H_future = self._denormalize(norm_H_future)
        return H_future

    def __call__(
        self,
        B_past: npt.NDArray[np.float64],
        H_past: npt.NDArray[np.float64],
        B_future: npt.NDArray[np.float64],
        temperature: float,
    ) -> npt.NDArray[np.float64]:

        assert B_past.shape[0] == H_past.shape[0], (
            "The past flux (B) and field (H) sequences must have the same length."
            + f"The given lengths are {B_past.shape[0]} for B and {H_past.shape[0]} for H."
        )

        H_future = self.apply_model(
            jnp.asarray(B_past),
            jnp.asarray(H_past),
            jnp.asarray(B_future),
            jnp.asarray(temperature),
        )
        H_future = np.array(jnp.squeeze(H_future[:-1]), dtype=np.float64)

        assert B_future.shape[0] == H_future.shape[0], (
            "Sanity Check: The future flux (B) and field (H) sequences must have "
            + f"the same length. The given lengths are {B_future.shape[0]} for B and {H_future.shape[0]} for H."
        )

        return H_future

    def batched_prediction(
        self,
        B_past: npt.NDArray[np.float64],
        H_past: npt.NDArray[np.float64],
        B_future: npt.NDArray[np.float64],
        temperature: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:

        assert B_past.shape[0] == H_past.shape[0], (
            "The past flux (B) and field (H) sequences must have the same batch_size."
            + f"The given batch_sizes are {B_past.shape[0]} for B and {H_past.shape[0]} for H."
        )
        assert B_past.shape[1] == H_past.shape[1], (
            "The past flux (B) and field (H) sequences must have the same length."
            + f"The given lengths are {B_past.shape[1]} for B and {H_past.shape[1]} for H."
        )

        H_future = eqx.filter_vmap(self.apply_model)(
            jnp.asarray(B_past),
            jnp.asarray(H_past),
            jnp.asarray(B_future),
            jnp.asarray(temperature),
        )
        H_future = np.array(jnp.squeeze(H_future[:, :-1]), dtype=np.float64)

        assert B_future.shape[0] == H_future.shape[0], (
            "The past flux (B) and field (H) sequences must have the same batch_size."
            + f"The given batch_sizes are {B_future.shape[0]} for B and {H_future.shape[0]} for H."
        )
        assert B_future.shape[1] == H_future.shape[1], (
            "Sanity Check: The future flux (B) and field (H) sequences must have "
            + f"the same length. The given lengths are {B_future.shape[1]} for B and {H_future.shape[1]} for H."
        )

        return H_future
