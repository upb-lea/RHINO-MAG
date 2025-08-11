from typing import Callable

import jax
import jax.numpy as jnp
import optax

from mc2.features.features_jax import compute_fe_single
from mc2.data_management import MaterialSet, FrequencySet, load_data_into_pandas_df
from mc2.models.model_interface import ModelInterface, NODEwInterface, RNNwInterface
from mc2.models.NODE import HiddenStateNeuralEulerODE
from mc2.models.RNN import GRU


def get_normalizer(material_name: str, featurize: Callable, subsampling_freq: int):
    data_dict = load_data_into_pandas_df(material=material_name)
    mat_set = MaterialSet.from_pandas_dict(data_dict)

    mat_set = mat_set.subsample(sampling_freq=subsampling_freq)

    train_set, val_set, test_set = mat_set.split_into_train_val_test(
        train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=12
    )
    train_set_norm = train_set.normalize(transform_H=True, featurize=featurize)
    normalizer = train_set_norm.normalizer
    return normalizer


def get_GRU_setup(
    material_name: str, model_key: jax.random.PRNGKey
) -> tuple[ModelInterface, optax.GradientTransformation, dict]:
    params = dict(
        training_params=dict(
            n_steps=100,  # 10_000
            val_every=500,
            tbptt_size=512,
            past_size=10,
            batch_size=64,
        ),
        model_params=dict(
            hidden_size=8,
            in_size=7,
            out_size=1,
            key=model_key,
        ),
        lr=1e-3,
    )
    optimizer = optax.adam(params["lr"])
    model = GRU(**params["model_params"])

    #####
    # TODO: How to do this properly? Will this always be the featurize function? What about the n_s parameter?
    def featurize(norm_B_past, norm_H_past, norm_B_future, temperature):
        past_length = norm_B_past.shape[0]
        future_length = norm_B_future.shape[0]

        featurized_B = compute_fe_single(jnp.hstack([norm_B_past, norm_B_future]), n_s=10)

        return featurized_B[past_length:]

    # TODO: Store normalizer objects, somewhat weird as it is?
    normalizer = get_normalizer(material_name, featurize)
    #####

    wrapped_model = RNNwInterface(model=model, normalizer=normalizer, featurize=featurize)

    return wrapped_model, optimizer, params


def get_HNODE_setup(material_name: str, model_key: jax.random.PRNGKey):
    params = dict(
        training_params=dict(
            n_steps=1_000_000,
            val_every=1_000,
            tbptt_size=64,
            past_size=1,
            batch_size=512,
            subsampling_freq=5,
        ),
        model_params=dict(
            obs_dim=1,
            state_dim=20,
            action_dim=5,
            width_size=64,
            depth=2,
            obs_func_type="identity",
            key=model_key,
        ),
        lr_params=dict(
            init_value=1e-3,
            transition_steps=1_000_000,
            transition_begin=2_000,
            decay_rate=0.1,
            end_value=1e-4,
        ),
    )

    lr_schedule = optax.schedules.exponential_decay(**params["lr_params"])
    optimizer = optax.adam(lr_schedule)
    model = HiddenStateNeuralEulerODE(**params["model_params"])

    def featurize(norm_B_past, norm_H_past, norm_B_future, temperature):
        past_length = norm_B_past.shape[0]
        future_length = norm_B_future.shape[0]

        featurized_B = compute_fe_single(jnp.hstack([norm_B_past, norm_B_future]), n_s=10)

        return featurized_B[past_length:]

    normalizer = get_normalizer(material_name, featurize, params["training_params"]["subsampling_freq"])
    wrapped_model = NODEwInterface(
        model=model,
        normalizer=normalizer,
        featurize=featurize,
    )

    params["model_params"]["key"] = params["model_params"]["key"].tolist()

    return wrapped_model, optimizer, params
