from abc import abstractmethod
from typing import Callable, Type
import pathlib
import json

import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import Normalizer


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


class NODEwInterface(ModelInterface):
    model: eqx.Module
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
    model: eqx.Module
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

    def call_with_warmup(self, B_past, H_past, B_future, T):
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)  #  ,f_norm , f

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred = self.normalized_warmup_call(B_past_norm, H_past_norm, B_future_norm, T_norm)  # ,f_norm
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
            [jnp.zeros((H_past_norm.shape[0], self.model.hidden_size - 1)), H_past_norm[:, -1, None]]
        )
        batch_H_pred = jax.vmap(self.model)(batch_x, init_hidden)
        return batch_H_pred[:, :, 0]

    def normalized_warmup_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):

        # warmstart with past
        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(
            B_past_norm, H_past_norm, B_past_norm, T_norm
        )  # ,None , f_norm
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_past_norm.shape)
        # f_norm_broad= jnp.broadcast_to(jnp.array([f_norm]), B_future_norm.shape)

        batch_x = jnp.concatenate(
            [B_past_norm[:, 1:, None], T_norm_broad[:, 1:, None], features_norm[:, 1:]], axis=-1
        )  # , f_norm_broad[...,None]
        init_hidden = jnp.hstack(
            [jnp.zeros((H_past_norm.shape[0], self.model.hidden_size - 1)), H_past_norm[:, 0, None]]
        )
        _, final_hidden_warmup = jax.vmap(self.model.warmup_call)(batch_x, init_hidden, H_past_norm[:, 1:])

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(
            B_past_norm, H_past_norm, B_future_norm, T_norm
        )  # ,None , f_norm
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)
        # f_norm_broad= jnp.broadcast_to(jnp.array([f_norm]), B_future_norm.shape)

        batch_x = jnp.concatenate(
            [B_future_norm[..., None], T_norm_broad[..., None], features_norm], axis=-1
        )  # , f_norm_broad[...,None]
        init_hidden = final_hidden_warmup
        batch_H_pred = jax.vmap(self.model)(batch_x, init_hidden)
        return batch_H_pred[:, :, 0]


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


from typing import Callable


class JAwInterface(ModelInterface):
    model: eqx.Module
    normalizer: eqx.Module
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
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)
        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]
        batch_H_pred_norm = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm)
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred_norm)
        return batch_H_pred_denorm

    def normalized_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        B_all_norm = jnp.concatenate([B_past_norm, B_future_norm], axis=1)
        B_all, H_past, T = self.normalizer.denormalize(B_all_norm, H_past_norm, T_norm)
        B_past = B_all[:, : B_past_norm.shape[1]]
        B_future = B_all[:, B_past_norm.shape[1] - 1 :]
        H0 = H_past[:, -1]

        def single_batch(H0_i, B_future_i):
            H_seq_i = self.model(H0_i, B_future_i)
            return H_seq_i

        batch_H_pred = jax.vmap(single_batch)(H0, B_future)
        batch_H_pred_norm = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)
        return batch_H_pred_norm


class JAwGRUwInterface(ModelInterface):
    model: eqx.Module
    normalizer: eqx.Module
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
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)  #  ,f_norm , f

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred = self.normalized_warmup_call(B_past_norm, H_past_norm, B_future_norm, T_norm)  # ,f_norm
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred)
        return batch_H_pred_denorm[:, :]

    def normalized_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        B_all_norm = jnp.concatenate([B_past_norm, B_future_norm], axis=1)
        B_all, H_past, T = self.normalizer.denormalize(B_all_norm, H_past_norm, T_norm)
        B_past = B_all[:, : B_past_norm.shape[1]]
        B_curr_and_future = B_all[:, B_past_norm.shape[1] - 1 :]
        H0 = H_past[:, -1]

        def single_batch(H0_i, B_future_i):
            H_seq_i = self.model.ja(H0_i, B_future_i)
            return H_seq_i

        batch_H_pred = jax.vmap(single_batch)(H0, B_curr_and_future)
        batch_H_pred_norm_ja = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_future_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)

        batch_x = jnp.concatenate([batch_H_pred_norm_ja[..., None], T_norm_broad[..., None], features_norm], axis=-1)
        init_hidden = jnp.hstack(
            [jnp.zeros((H_past_norm.shape[0], self.model.gru.hidden_size - 1)), H_past_norm[:, -1, None]]
        )
        batch_H_diff_pred = jax.vmap(self.model.gru)(batch_x, init_hidden)

        return batch_H_pred_norm_ja + batch_H_diff_pred[:, :, 0]

    def normalized_warmup_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):

        B_all_norm = jnp.concatenate([B_past_norm, B_future_norm], axis=1)
        B_all, H_past, T = self.normalizer.denormalize(B_all_norm, H_past_norm, T_norm)
        B_past = B_all[:, : B_past_norm.shape[1]]
        B_curr_and_future = B_all[:, B_past_norm.shape[1] - 1 :]
        H0_past = H_past[:, 0]

        def single_batch_warmup(H0_i, B_future_i):
            H_seq_i = self.model.ja(H0_i, B_future_i)
            return H_seq_i

        batch_H_pred = jax.vmap(single_batch_warmup)(H0_past, B_past)
        batch_H_pred_norm_ja = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_past_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_past_norm.shape)

        batch_x = jnp.concatenate(
            [batch_H_pred_norm_ja[..., None], T_norm_broad[:, 1:, None], features_norm[:, 1:]], axis=-1
        )
        init_hidden = jnp.zeros((H_past_norm.shape[0], self.model.gru.hidden_size))

        _, final_hidden_warmup = jax.vmap(self.model.gru.warmup_call)(
            batch_x, init_hidden, H_past_norm[:, 1:] - batch_H_pred_norm_ja
        )

        B_all_norm = jnp.concatenate([B_past_norm, B_future_norm], axis=1)
        B_all, H_past, T = self.normalizer.denormalize(B_all_norm, H_past_norm, T_norm)
        B_past = B_all[:, : B_past_norm.shape[1]]
        B_curr_and_future = B_all[:, B_past_norm.shape[1] - 1 :]
        H0 = H_past[:, -1]

        def single_batch(H0_i, B_future_i):
            H_seq_i = self.model.ja(H0_i, B_future_i)
            return H_seq_i

        batch_H_pred = jax.vmap(single_batch)(H0, B_curr_and_future)
        batch_H_pred_norm_ja = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_future_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)

        batch_x = jnp.concatenate([batch_H_pred_norm_ja[..., None], T_norm_broad[..., None], features_norm], axis=-1)
        init_hidden = final_hidden_warmup
        batch_H_diff_pred = jax.vmap(self.model.gru)(batch_x, init_hidden)

        return batch_H_pred_norm_ja + batch_H_diff_pred[:, :, 0]


class JAGRUwInterface(ModelInterface):
    model: eqx.Module
    normalizer: eqx.Module
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
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)
        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred_norm = self.normalized_warmup_call(B_past_norm, H_past_norm, B_future_norm, T_norm)

        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred_norm)
        return batch_H_pred_denorm

    def normalized_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        B_all_norm = jnp.concatenate([B_past_norm, B_future_norm], axis=1)
        B_all, H_past, T = self.normalizer.denormalize(B_all_norm, H_past_norm, T_norm)

        B_past = B_all[:, : B_past_norm.shape[1]]
        B_curr_and_future = B_all[:, B_past_norm.shape[1] - 1 :]  # need last B value of past
        H0 = H_past[:, -1]

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_future_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)
        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)

        batch_x = jnp.concatenate([T_norm_broad[..., None], features_norm], axis=-1)
        init_hidden = jnp.zeros((H_past_norm.shape[0], self.model.gru.hidden_size))

        def single_batch(H0_i, B_future_i, features_i, init_hidden_i):
            return self.model(H0_i, B_future_i, features_i, init_hidden_i)

        batch_H_pred = jax.vmap(single_batch)(H0, B_curr_and_future, batch_x, init_hidden)
        batch_H_pred_norm = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)
        return batch_H_pred_norm

    def normalized_warmup_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        B_all_norm = jnp.concatenate([B_past_norm, B_future_norm], axis=1)
        B_all, H_past, T = self.normalizer.denormalize(B_all_norm, H_past_norm, T_norm)

        B_past = B_all[:, : B_past_norm.shape[1]]
        B_future = B_all[:, B_past_norm.shape[1] - 1 :]
        H0_past = H_past[:, 0]

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_past_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)
        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_past_norm.shape)

        batch_x = jnp.concatenate([T_norm_broad[:, 1:, None], features_norm[:, 1:]], axis=-1)
        init_hidden = jnp.zeros((H_past_norm.shape[0], self.model.gru.hidden_size))

        def single_batch_warmup(H0_i, B_past_i, features_i, init_hidden_i, H_true_i):
            return self.model.warmup_call(H0_i, B_past_i, features_i, init_hidden_i, H_true_i)

        _, final_hidden_warmup = jax.vmap(single_batch_warmup)(H0_past, B_past, batch_x, init_hidden, H_past[:, 1:])

        H0 = H_past[:, -1]

        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_future_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)
        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)

        batch_x = jnp.concatenate([T_norm_broad[..., None], features_norm], axis=-1)
        init_hidden = final_hidden_warmup

        def single_batch(H0_i, B_future_i, features_i, init_hidden_i):
            return self.model(H0_i, B_future_i, features_i, init_hidden_i)

        batch_H_pred = jax.vmap(single_batch)(H0, B_future, batch_x, init_hidden)
        batch_H_pred_norm = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)
        return batch_H_pred_norm
