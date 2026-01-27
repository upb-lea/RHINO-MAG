"""Interfaces between data-driven models in 'mc2.models.RNN' (and 'mc2.models.NODE') and the data format for MC2."""

from typing import Callable, Type
from rhmag.data_management import Normalizer

import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp
import equinox as eqx

from rhmag.model_interfaces.model_interface import ModelInterface
from rhmag.models.NODE import HiddenStateNeuralEulerODE
from rhmag.models.RNN import GRU, VectorfieldGRU, GRUwLinear, GRUwLinearModel, GRUaroundLinearModel

MU_0 = 4 * jnp.pi * 1e-7


class RNNwInterface(ModelInterface):
    """Model interface for basic recurrent neural network (RNN) models.

    Manages the interaction between generic data-driven model and input material data.

    Args:
        model (GRU | GRUwLinear): The generic RNN model
        normalizer (Normalizer): The normalization object to normalize the raw material data for the RNN
        featurize (Callable): The featurization function to add further features to the material data.
    """

    model: GRU | GRUwLinear
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    def __call__(
        self,
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        T: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:
        """Main function to be called for this model object.

        Takes the material data and produces the model prediction. In between, the data is
        normalized, featurized, a warmup of the hidden state of the RNN is performed, the
        H prediction is performed and the output is denormalized back to a physical value,
        i.e., from a numerical value back to its interpretation as `Ampere/m`.

        Args:
            B_past (jax.Array): The physical (non-normalized) flux density values from time
                step k0 to k1 with shape (n_batches, past_sequence_length)
            H_past (jax.Array): The physical (non-normalized) field values from time step
                k0 to k1 with shape (n_batches, past_sequence_length)
            B_future (jax.Array): The physical (non-normalized) flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T (float): The temperature of the material with shape (n_batches,)

        Returns:
            H_pred (jax.Array): The physical (non-normalized) field values from time
                step k1 to k2 with shape (n_batches, future_sequence_length)
        """

        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm, warmup)
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred)

        return batch_H_pred_denorm

    def _prepare_model_input(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """Prepare the input vector for the model based on the provided material data.

        Args:
            B_past_norm (jax.Array): The normalized flux density values from time step k0
                to k1 with shape (n_batches, past_sequence_length)
            H_past_norm (jax.Array): The normalized field values from time step k0 to k1
                with shape (n_batches, past_sequence_length)
            B_future_norm (jax.Array): The physical normalized flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)

        Returns:
            batch_x (jax.Array): The input to the RNN with shape (n_batches, future_sequence_length, gru_in_size)

        """
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
        """Warm-up the hidden state of the RNN based on the previous trajectory data.

        The warmup process is essentially a prediction process where the first element of H_past
        is used to initialize the first hidden state and where the first element of the hidden state
        is corrected with the other true H_past value after each step.

        NOTE: The future values of B are actually not used here but only passed for alignment with the
        other interfaces.

        Args:
            B_past_norm (jax.Array): The normalized flux density values from time step k0
                to k1 with shape (n_batches, past_sequence_length)
            H_past_norm (jax.Array): The normalized field values from time step k0 to k1
                with shape (n_batches, past_sequence_length)
            B_future_norm (jax.Array): The physical normalized flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)

        Returns:
            final_hidden_warmup (jax.Array): The warmed up hidden state.
        """

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_past_norm, T_norm)
        batch_x = batch_x[:, 1:]

        init_hidden = self.model.construct_init_hidden(
            out_true=H_past_norm[:, 0, None],
            batch_size=H_past_norm.shape[0],
        )
        _, final_hidden_warmup = jax.vmap(self.model.warmup_call)(batch_x, init_hidden, H_past_norm[:, 1:])
        return final_hidden_warmup

    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:
        """Performs the warmup and the prediction on the normalized data.

        Args:
            B_past_norm (jax.Array): The normalized flux density values from time step k0
                to k1 with shape (n_batches, past_sequence_length)
            H_past_norm (jax.Array): The normalized field values from time step k0 to k1
                with shape (n_batches, past_sequence_length)
            B_future_norm (jax.Array): The physical normalized flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)
            warmup (bool): Whether warmup should be performed or only the initial state should
                be constructed filled with the first true value and zeros.

        Returns:
            The normalized field prediction as a jax.Array with shape (n_batches, future_sequence_length)
        """
        if warmup and H_past_norm.shape[1] > 1:
            init_hidden = self._warmup(B_past_norm, H_past_norm, B_future_norm, T_norm)
        else:
            init_hidden = self.model.construct_init_hidden(
                out_true=H_past_norm[:, -1, None],
                batch_size=H_past_norm.shape[0],
            )

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_future_norm, T_norm)
        batch_H_pred = jax.vmap(self.model)(batch_x, init_hidden)
        return batch_H_pred[:, :, 0]


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

        norm_temperature_broad = jnp.broadcast_to(norm_temperature[None], norm_B_future.shape[0])
        featurized_input = jnp.concatenate(
            [norm_B_future[..., None], norm_temperature_broad[..., None], featurized_input], axis=-1
        )
        _, norm_H_future = self.model(norm_H_past[-1], featurized_input, tau=1 / 16 * 1e-6)
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

            T_norm_broad = jnp.broadcast_to(T_norm[None], B_future_norm.shape)
            featurized_input = jnp.concatenate(
                [B_future_norm[..., None], T_norm_broad[..., None], featurized_input], axis=-1
            )

            _, H_future_norm = self.model(H_past_norm[-1], featurized_input, tau=1 / 16 * 1e-6)
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
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        T: jax.Array,
    ) -> jax.Array:
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
        H_future = jnp.array(H_future[:, :-1, 0])

        assert B_future.shape[0] == H_future.shape[0], (
            "The future flux (B) and field (H) sequences must have the same batch_size."
            + f"The given batch_sizes are {B_future.shape[0]} for B and {H_future.shape[0]} for H."
        )
        assert B_future.shape[1] == H_future.shape[1], (
            "Sanity Check: The future flux (B) and field (H) sequences must have "
            + f"the same length. The given lengths are {B_future.shape[1]} for B and {H_future.shape[1]} for H."
        )

        return H_future


class MagnetizationRNNwInterface(RNNwInterface):

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
            init_hidden = self.model.construct_init_hidden(
                out_true=B_past_norm[:, -1, None] - jnp.arctanh(H_past_norm[:, -1, None]),
                batch_size=H_past_norm.shape[0],
            )

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_future_norm, T_norm)
        prediction_norm = jax.vmap(self.model)(batch_x, init_hidden)

        # v1:
        H_pred_norm = jnp.tanh(B_future_norm[..., None] - prediction_norm)

        # v2:
        # (B_future, prediction, _) = self.normalizer.denormalize(B_future_norm, prediction_norm, T_norm)
        # H_pred = B_future[..., None] - prediction
        # H_pred_norm = jax.vmap(jax.vmap(self.normalizer.normalize_H))(H_pred)

        return jnp.squeeze(H_pred_norm)

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

        init_hidden = self.model.construct_init_hidden(
            out_true=B_past_norm[:, 0, None] - jnp.arctanh(H_past_norm[:, 0, None]),
            batch_size=H_past_norm.shape[0],
        )
        _, final_hidden_warmup = jax.vmap(self.model.warmup_call)(
            batch_x,
            init_hidden,
            B_past_norm[:, 1:] - jnp.arctanh(H_past_norm[:, 1:]),
        )
        return final_hidden_warmup


class VectorfieldGRUInterface(RNNwInterface):
    model: VectorfieldGRU

    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
        warmup: bool = True,
        debug: bool = False,
    ) -> jax.Array:

        if warmup and H_past_norm.shape[1] > 1:
            init_hidden = self._warmup(B_past_norm, H_past_norm, B_future_norm, T_norm)
        else:
            init_hidden = self.model.construct_init_hidden(
                out_true=None,
                batch_size=H_past_norm.shape[0],
            )

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_future_norm, T_norm)
        mag_pred_norm = jax.vmap(self.model)(batch_x, init_hidden) / self.model.n_locs

        mag_norm = jnp.sum(mag_pred_norm[..., 0], axis=-1)  # M * mu_0

        # B_future = self.normalizer.B_max * B_future_norm
        # M = mag_norm * 1e6

        # H_pred = B_future - M
        # H_pred_norm = self.normalizer.normalize_H(H_pred)

        H_pred_norm = B_future_norm - mag_norm

        if debug:
            return jnp.squeeze(H_pred_norm), mag_pred_norm
        else:
            return jnp.squeeze(H_pred_norm)

    def _warmup(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        return self.model.construct_init_hidden(
            out_true=None,
            batch_size=H_past_norm.shape[0],
        )


class GRUwLinearModelInterface(ModelInterface):
    model: GRUwLinearModel
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    def _prepare_gru_input(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(
            B_past_norm, H_past_norm, B_future_norm, T_norm
        )  # ,None , f_norm
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)
        # f_norm_broad= jnp.broadcast_to(jnp.array([f_norm]), B_future_norm.shape)

        batch_x = jnp.concatenate(
            [B_future_norm[..., None], T_norm_broad[..., None], features_norm], axis=-1
        )  # , f_norm_broad[...,None]

        return batch_x

    def _prepare_linear_input(self, B_past_norm: jax.Array, B_future_norm: jax.Array) -> jax.Array:
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

    def _warmup(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """Warm-up the hidden state of the RNN based on the previous trajectory data."""
        gru_in = self._prepare_gru_input(B_past_norm, H_past_norm, B_past_norm, T_norm)
        gru_in = gru_in[:, 1:]

        init_hidden = self.model.construct_init_hidden(
            out_true=jnp.arctanh(H_past_norm[:, 0, None]) / B_past_norm[:, 0, None],
            batch_size=H_past_norm.shape[0],
        )
        linear_in = self._prepare_linear_input(B_past_norm[:, :1], B_past_norm[:, 1:])
        _, final_hidden_warmup = jax.vmap(self.model.warmup_call)(
            gru_in,
            linear_in,
            init_hidden,
            jnp.arctanh(H_past_norm[:, 1:]) / B_past_norm[:, 1:],
        )
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
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)
        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred_norm = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm, warmup)
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred_norm)
        return batch_H_pred_denorm

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
            init_hidden = self.model.construct_init_hidden(
                out_true=jnp.arctanh(H_past_norm[:, -1, None]) / B_past_norm[:, -1, None],
                batch_size=H_past_norm.shape[0],
            )

        gru_in = self._prepare_gru_input(B_past_norm, H_past_norm, B_future_norm, T_norm)
        linear_in = self._prepare_linear_input(B_past_norm, B_future_norm)

        batch_H_pred_norm = eqx.filter_vmap(self.model)(gru_in, linear_in, init_hidden)

        return jnp.squeeze(batch_H_pred_norm)


class GRUaroundLinearModelInterface(GRUwLinearModelInterface):
    model: GRUaroundLinearModel
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    def _warmup(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """Warm-up the hidden state of the RNN based on the previous trajectory data."""
        gru_in = self._prepare_gru_input(B_past_norm, H_past_norm, B_past_norm, T_norm)
        gru_in = gru_in[:, 1:]

        init_hidden = self.model.construct_init_hidden(
            out_true=None,  # TODO
            batch_size=H_past_norm.shape[0],
        )
        linear_in = self._prepare_linear_input(B_past_norm[:, :1], B_past_norm[:, 1:])
        _, final_hidden_warmup = jax.vmap(self.model.warmup_call)(
            gru_in,
            linear_in,
            init_hidden,
            None,  # TODO
        )
        return final_hidden_warmup

    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:

        gru_in = self._prepare_gru_input(B_past_norm, H_past_norm, B_future_norm, T_norm)
        linear_in = self._prepare_linear_input(B_past_norm, B_future_norm)

        # if warmup and H_past_norm.shape[1] > 1:
        #     init_hidden = self._warmup(B_past_norm, H_past_norm, B_future_norm, T_norm)
        # else:
        linear_out = eqx.filter_vmap(self.model.linear.predict)(linear_in[:, 0, :])

        init_hidden = self.model.construct_init_hidden(
            out_true=H_past_norm[:, -1][..., None] - linear_out,  # TODO: we need the linaer out for one step earlier...
            batch_size=H_past_norm.shape[0],
        )

        batch_H_pred_norm = eqx.filter_vmap(self.model)(gru_in, linear_in, init_hidden)

        return jnp.squeeze(batch_H_pred_norm)
