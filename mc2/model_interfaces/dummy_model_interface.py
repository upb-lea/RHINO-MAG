
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import Normalizer
from mc2.model_interfaces.model_interface import ModelInterface
from mc2.models.dummy_model import DummyModel

class DummyModelInterface(ModelInterface):
    model: DummyModel
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    def __call__(
        self,
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        T: jax.Array,
    ) -> jax.Array:

        # concatenating and normalizing the data
        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        # performing prediction
        batch_H_pred = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm)

        # denormalizing predicted value
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred)

        return batch_H_pred_denorm

    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:
        batch_H_pred = jax.vmap(self.model)(B_future_norm)
        return batch_H_pred