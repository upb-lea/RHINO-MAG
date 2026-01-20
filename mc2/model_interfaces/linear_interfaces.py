from typing import Callable, Type
from mc2.data_management import Normalizer

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.model_interfaces.model_interface import ModelInterface


class LinearInterface(ModelInterface):
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
        B_all = jnp.concatenate([B_past_norm, B_future_norm], axis=1)

        past_size = B_past_norm.shape[1]
        M_per_side = int((self.model.in_size - 1) / 2)

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
            B_in = B_all[:, past_size:, None]

        # features = eqx.filter_vmap(
        #     eqx.filter_vmap(self.featurize, in_axes=(0, 0, 0, 0)),
        #     in_axes=(None, None, 2, None),
        #     out_axes=2,
        # )(B_past_norm, H_past_norm, B_in, T_norm)
        # features_norm = eqx.filter_vmap(eqx.filter_vmap(eqx.filter_vmap(self.normalizer.normalize_fe)))(features)
        # features_norm = features_norm.reshape((features.shape[0], features.shape[1], -1))

        # model_in = jnp.concatenate([features_norm, B_in], axis=-1)

        model_in = B_in

        batch_H_pred_norm = eqx.filter_vmap(self.model)(model_in)

        return batch_H_pred_norm[..., 0]

    def normalized_warmup_call(self, B_past_norm, H_past_norm, B_future_norm, T_norm):
        raise NotImplementedError()
