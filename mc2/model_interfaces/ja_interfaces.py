from typing import Callable, Type
from mc2.data_management import Normalizer

import jax
import jax.numpy as jnp
import equinox as eqx


from mc2.model_interfaces.model_interface import ModelInterface


class JAwInterface(ModelInterface):
    model: eqx.Module
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


class JAWithExternGRUwInterface(ModelInterface):
    model: eqx.Module
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
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)  #  ,f_norm , f

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred = self.normalized_call(
            B_past_norm, H_past_norm, B_future_norm, T_norm
        )  # normalized_warmup_call -> not working well
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
        # init_hidden = jnp.hstack(
        #     [jnp.zeros((H_past_norm.shape[0], self.model.gru.hidden_size - 1)), H_past_norm[:, -1, None]]
        # )
        init_hidden = jnp.zeros((H_past_norm.shape[0], self.model.gru.hidden_size))
        batch_H_diff_pred = jax.vmap(self.model.gru)(batch_x, init_hidden)

        return batch_H_pred_norm_ja + batch_H_diff_pred[:, :, 0]

    def normalized_warmup_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):

        B_all_norm = jnp.concatenate([B_past_norm, B_future_norm], axis=1)
        B_all, H_past, T = self.normalizer.denormalize(B_all_norm, H_past_norm, T_norm)

        B_past_norm_plus = B_all_norm[:, : B_past_norm.shape[1] + 1]
        B_past = B_all[:, : B_past_norm.shape[1]]
        B_curr_and_future = B_all[:, B_past_norm.shape[1] - 1 :]
        H0_past = H_past[:, 0]

        def single_batch_warmup(H0_i, B_future_i):
            H_seq_i = self.model.ja(H0_i, B_future_i)
            return H_seq_i

        batch_H_pred = jax.vmap(single_batch_warmup)(H0_past, B_past)
        batch_H_pred_norm_ja = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)

        features_plus = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(
            B_past_norm, H_past_norm, B_past_norm_plus, T_norm
        )
        features = features_plus[:, :-1]
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_past_norm.shape)

        batch_x = jnp.concatenate(
            [batch_H_pred_norm_ja[..., None], T_norm_broad[:, 1:, None], features_norm[:, 1:]], axis=-1
        )
        init_hidden = jnp.zeros((H_past_norm.shape[0], self.model.gru.hidden_size))

        _, final_hidden_warmup = jax.vmap(self.model.gru.warmup_call)(
            batch_x, init_hidden, H_past_norm[:, 1:] - batch_H_pred_norm_ja
        )
        # _, final_hidden_warmup = jax.vmap(self.model.gru.warmup_call)(
        #     batch_x, init_hidden, jnp.zeros_like(H_past_norm[:, 1:] - batch_H_pred_norm_ja)
        # )

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


class JAWithGRUwInterface(ModelInterface):
    model: eqx.Module
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
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)
        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred_norm = self.normalized_warmup_call(
            B_past_norm, H_past_norm, B_future_norm, T_norm
        )  # normalized_warmup_call seems to be not benficial here -> Bug?

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

        B_past_norm_plus = B_all_norm[:, : B_past_norm.shape[1] + 1]
        B_past = B_all[:, : B_past_norm.shape[1]]
        B_curr_and_future = B_all[:, B_past_norm.shape[1] - 1 :]
        H0_past = H_past[:, 0]

        features_plus = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(
            B_past_norm, H_past_norm, B_past_norm_plus, T_norm
        )
        features = features_plus[:, :-1]
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

        batch_H_pred = jax.vmap(single_batch)(H0, B_curr_and_future, batch_x, init_hidden)
        batch_H_pred_norm = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)
        return batch_H_pred_norm


class JAParamMLPwInterface(ModelInterface):
    model: eqx.Module
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
        return self.__call__(B_past, H_past, B_future, T)

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

        def single_batch(H0_i, B_future_i, features_i):
            return self.model(H0_i, B_future_i, features_i)

        batch_H_pred = jax.vmap(single_batch)(H0, B_curr_and_future, batch_x)
        batch_H_pred_norm = jax.vmap(jax.vmap(self.normalizer.normalize_H))(batch_H_pred)
        return batch_H_pred_norm

    def normalized_warmup_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        return self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm)


class GRUWithPINNInterface(ModelInterface):
    model: eqx.Module
    normalizer: Normalizer
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

        batch_H_pred = jax.vmap(self.model)(batch_x)
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
