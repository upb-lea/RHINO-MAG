from abc import abstractmethod
from typing import Callable
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
        T: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:

        assert B_past.ndim == 2, (
            "The expected dimensions for B_past are (n_batches, past_sequence_length). "
            + f"The given array has dimension {B_past.ndim} instead."
        )
        assert H_past.ndim == 2, (
            "The expected dimensions for H_past are (n_batches, past_sequence_length). "
            + f"The given array has dimension {H_past.ndim} instead."
        )
        assert B_future.ndim == 2, (
            "The expected dimensions for B_future are (n_batches, future_sequence_length). "
            + f"The given array has dimension {B_future.ndim} instead."
        )
        assert T.ndim == 1, (
            "The expected dimensions for T are (n_batches,). " + f"The given array has dimension {T.ndim} instead."
        )

        assert B_past.shape[0] == H_past.shape[0], (
            "The past flux (B) and field (H) sequences must have the same batch_size. "
            + f"The given batch_sizes are {B_past.shape[0]} for B and {H_past.shape[0]} for H."
        )
        assert B_past.shape[1] == H_past.shape[1], (
            "The past flux (B) and field (H) sequences must have the same length. "
            + f"The given lengths are {B_past.shape[1]} for B and {H_past.shape[1]} for H."
        )

        H_future = eqx.filter_vmap(self.apply_model)(
            jnp.asarray(B_past),
            jnp.asarray(H_past),
            jnp.asarray(B_future),
            jnp.asarray(T),
        )
        H_future = np.array(H_future[:, :-1, 0], dtype=np.float64)

        assert B_future.shape[0] == H_future.shape[0], (
            "The future flux (B) and field (H) sequences must have the same batch_size."
            + f"The given batch_sizes are {B_future.shape[0]} for B and {H_future.shape[0]} for H."
        )
        assert B_future.shape[1] == H_future.shape[1], (
            "Sanity Check: The future flux (B) and field (H) sequences must have "
            + f"the same length. The given lengths are {B_future.shape[1]} for B and {H_future.shape[1]} for H."
        )

        return H_future


class RNNwInterface(ModelInterface):
    rnn: eqx.Module
    normalizer: eqx.Module
    featurize: Callable = eqx.field(static=True)

    def __call__(self, B_past, H_past, B_future, T):
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)  #  ,f_norm , f

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm)  # ,f_norm
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred)
        return batch_H_pred_denorm[:, :]

    def normalized_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(
            B_past_norm, H_past_norm, B_future_norm, T_norm
        )  # ,None , f_norm
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)
        # f_norm_broad= jnp.broadcast_to(jnp.array([f_norm]), B_future_norm.shape)

        batch_x = jnp.concatenate(
            [B_future_norm[..., None], T_norm_broad[..., None], features_norm], axis=-1
        )  # , f_norm_broad[...,None]
        init_hidden = jnp.hstack(
            [jnp.zeros((H_past_norm.shape[0], self.rnn.hidden_size - 1)), H_past_norm[:, -1, None]]
        )
        batch_H_pred = jax.vmap(self.rnn)(batch_x, init_hidden)
        return batch_H_pred[:, :, 0]
