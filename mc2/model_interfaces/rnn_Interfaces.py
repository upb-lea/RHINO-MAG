"""Interfaces between data-driven models in 'mc2.models.RNN' (and 'mc2.models.NODE') and the data format for MC2."""

from typing import Callable, Type
from mc2.data_management import Normalizer

import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.model_interfaces.model_interface import ModelInterface
from mc2.models.NODE import HiddenStateNeuralEulerODE
from mc2.models.RNN import GRU, GRUwLinear, GRUwLinearModel


class NODEwInterface(ModelInterface):
    model: HiddenStateNeuralEulerODE
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    @eqx.filter_jit
    def apply_model(
        self,
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        temperature: jax.Array,
    ) -> jax.Array:
        past_length = B_past.shape[0]

        norm_B, norm_H_past, norm_temperature = self.normalizer.normalize(
            jnp.hstack([B_past, B_future]), H_past, temperature
        )

        norm_B_past = norm_B[:past_length]
        norm_B_future = norm_B[past_length:]

        featurized_input = self.featurize(norm_B_past, norm_H_past, norm_B_future, norm_temperature)

        _, norm_H_future = self.model(norm_H_past[-1], featurized_input, tau=1)
        H_future = self.normalizer.denormalize_H(norm_H_future)
        return H_future

    @eqx.filter_jit
    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        def norm_apply_model(
            B_past_norm: jax.Array, H_past_norm: jax.Array, B_future_norm: jax.Array, T_norm: jax.Array
        ) -> jax.Array:
            featurized_input = self.featurize(B_past_norm, H_past_norm, B_future_norm, T_norm)
            _, H_future_norm = self.model(H_past_norm[-1], featurized_input, tau=1)
            return H_future_norm

        H_future = eqx.filter_vmap(norm_apply_model)(
            B_past_norm,
            H_past_norm,
            B_future_norm,
            T_norm,
        )
        H_future = H_future[:, :-1, 0]
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
    model: GRU | GRUwLinear
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    def _prepare_model_input(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_future_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)

        batch_x = jnp.concatenate([B_future_norm[..., None], T_norm_broad[..., None], features_norm], axis=-1)
        return batch_x

    def _warmup(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """Warm-up the hidden state of the RNN based on the previous trajectory data."""

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_past_norm, T_norm)
        batch_x = batch_x[:, 1:]

        init_hidden = jnp.hstack(
            [H_past_norm[:, 0, None], jnp.zeros((H_past_norm.shape[0], self.model.hidden_size - 1))]
        )
        _, final_hidden_warmup = jax.vmap(self.model.warmup_call)(batch_x, init_hidden, H_past_norm[:, 1:])
        return final_hidden_warmup

    def __call__(
        self,
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        T: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:

        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)  #  ,f_norm , f

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm, warmup)  # ,f_norm
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred)
        return batch_H_pred_denorm[:, :]

    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:

        if warmup and H_past_norm.shape[1] > 1:
            init_hidden = self._warmup(B_past_norm, H_past_norm, B_future_norm, T_norm)
        else:
            init_hidden = jnp.hstack(
                [H_past_norm[:, -1, None], jnp.zeros((H_past_norm.shape[0], self.model.hidden_size - 1))]
            )

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_future_norm, T_norm)
        batch_H_pred = jax.vmap(self.model)(batch_x, init_hidden)
        return batch_H_pred[:, :, 0]


class MagnetizationRNNwInterface(RNNwInterface):

    def _warmup(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        pass

    def normalized_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_future_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)

        batch_x = jnp.concatenate([B_future_norm[..., None], T_norm_broad[..., None], features_norm], axis=-1)
        init_hidden = jnp.hstack(
            [
                B_past_norm[:, -1, None] - jnp.arctanh(H_past_norm[:, -1, None]),
                jnp.zeros((H_past_norm.shape[0], self.model.hidden_size - 1)),
            ],
        )

        prediction_norm = jax.vmap(self.model)(batch_x, init_hidden)

        # v1:
        H_pred_norm = jnp.tanh(B_future_norm[..., None] - prediction_norm)

        # v2:
        # (B_future, prediction, _) = self.normalizer.denormalize(B_future_norm, prediction_norm, T_norm)
        # H_pred = B_future[..., None] - prediction
        # H_pred_norm = jax.vmap(jax.vmap(self.normalizer.normalize_H))(H_pred)

        return jnp.squeeze(H_pred_norm)


class GRUwLinearModelInterface(ModelInterface):
    model: GRUwLinearModel
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    def __call__(self, B_past, H_past, B_future, T):
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)
        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred_norm = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm)
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred_norm)
        return batch_H_pred_denorm

    def call_with_warmup(self, B_past, H_past, B_future, T):
        raise NotImplementedError()

    def _prepare_GRU_in(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(
            B_past_norm, H_past_norm, B_future_norm, T_norm
        )  # ,None , f_norm
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)
        # f_norm_broad= jnp.broadcast_to(jnp.array([f_norm]), B_future_norm.shape)

        batch_x = jnp.concatenate(
            [B_future_norm[..., None], T_norm_broad[..., None], features_norm], axis=-1
        )  # , f_norm_broad[...,None]

        hidden_state_init = jnp.arctanh(H_past_norm[:, -1, None]) / B_past_norm[:, -1, None]

        init_hidden = jnp.hstack([hidden_state_init, jnp.zeros((H_past_norm.shape[0], self.model.hidden_size - 1))])

        return batch_x, init_hidden

    def _prepare_linear_in(self, B_past_norm, B_future_norm):
        ## preparations for linear model
        B_all = jnp.concatenate([B_past_norm, B_future_norm], axis=1)

        past_size = B_past_norm.shape[1]
        M_per_side = int((self.model.linear_in_size - 1) / 2)

        if M_per_side > 0:

            B_all_padded = jnp.pad(B_all, ((0, 0), (M_per_side, M_per_side)), mode="reflect", reflect_type="odd")

            B_in = jnp.concatenate(
                [
                    jnp.roll(B_all_padded, idx)[..., None]
                    for idx in jnp.arange(
                        -M_per_side,
                        M_per_side + 1,
                        1,
                    )
                ],
                axis=-1,
            )[:, M_per_side + past_size : -M_per_side, :]
        else:
            B_in = B_future_norm[..., None]
        return B_in

    def normalized_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):

        GRU_in, init_hidden = self._prepare_GRU_in(B_past_norm, H_past_norm, B_future_norm, T_norm)
        linear_in = self._prepare_linear_in(B_past_norm, B_future_norm)

        batch_H_pred_norm = eqx.filter_vmap(self.model)(GRU_in, linear_in, init_hidden)

        return jnp.squeeze(batch_H_pred_norm)

    def normalized_warmup_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        raise NotImplementedError()
