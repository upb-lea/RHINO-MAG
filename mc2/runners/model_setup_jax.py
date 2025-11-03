from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from mc2.losses import MSE_loss, adapted_RMS_loss
from mc2.features.features_jax import compute_fe_single
from mc2.data_management import MaterialSet, load_data_into_pandas_df
from mc2.models.model_interface import (
    NODEwInterface,
    RNNwInterface,
    JAwInterface,
    JAParamMLPwInterface,
    JAWithGRUwInterface,
    JAWithExternGRUwInterface,
)
from mc2.models.NODE import HiddenStateNeuralEulerODE
from mc2.models.RNN import GRU
from mc2.models.jiles_atherton import (
    JAStatic,
    JAStatic2,
    JAParamGRUlin,
    JAParamMLP,
    JAWithExternGRU,
    JAWithGRU,
    JAWithGRUlin,
    JAWithGRUlinFinal,
)
from mc2.data_management import Normalizer


SUPPORTED_MODELS = ["GRU", "HNODE", "JA"]

SUPPORTED_LOSSES = ["MSE", "adapted_RMS"]


def get_normalizer(material_name: str, featurize: Callable, subsampling_freq: int, do_normalization: bool):
    if do_normalization:
        data_dict = load_data_into_pandas_df(material=material_name)
        mat_set = MaterialSet.from_pandas_dict(data_dict)

        mat_set = mat_set.subsample(sampling_freq=subsampling_freq)

        train_set, val_set, test_set = mat_set.split_into_train_val_test(
            train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0
        )
        train_set_norm = train_set.normalize(transform_H=True, featurize=featurize)
        normalizer = train_set_norm.normalizer
    else:
        normalizer = Normalizer(
            B_max=1.0,
            H_max=1.0,
            T_max=1.0,
            norm_fe_max=jnp.ones(5).tolist(),  # TODO: adapt to number of features?
            H_transform=lambda h: h,
            H_inverse_transform=lambda h: h,
        )
        train_set, val_set, test_set = None, None, None
    return normalizer, (train_set, val_set, test_set)


def setup_model(
    model_label: str,
    material_name: str,
    model_key: jax.random.PRNGKey,
    subsample_freq=1,
    n_epochs=100,
    tbptt_size=1024,
    batch_size=256,
    tbptt_size_start=None,  # (size, n_epochs_steps)
):
    def featurize(norm_B_past, norm_H_past, norm_B_future, temperature):
        past_length = norm_B_past.shape[0]
        future_length = norm_B_future.shape[0]

        featurized_B = compute_fe_single(jnp.hstack([norm_B_past, norm_B_future]), n_s=10)

        return featurized_B[past_length:]

    normalizer, data_tuple = get_normalizer(
        material_name,
        featurize,
        subsample_freq,
        True,
    )

    match model_label:
        case "HNODE":
            model_params_d = dict(
                obs_dim=1,
                state_dim=5,
                action_dim=5,
                width_size=64,
                depth=2,
                obs_func_type="identity",
                key=model_key,
            )
            model = HiddenStateNeuralEulerODE(**model_params_d)
            mdl_interface_cls = NODEwInterface
        case "GRU":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = GRU(**model_params_d)
            mdl_interface_cls = RNNwInterface
        case "JAWithExternGRU":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = JAWithExternGRU(**model_params_d)
            mdl_interface_cls = JAWithExternGRUwInterface
        case "JAWithGRUlin":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = JAWithGRUlin(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAWithGRUlinFinal":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = JAWithGRUlinFinal(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAWithGRU":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = JAWithGRU(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAParamGRUlin":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = JAParamGRUlin(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAParamMLP":
            model_params_d = dict(hidden_size=32, depth=2, in_size=7, key=model_key)
            model = JAParamMLP(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAParamMLPwInterface
        case "JA":
            model_params_d = dict(key=model_key)
            model = JAStatic(key=model_key)
            mdl_interface_cls = JAwInterface
        case "JA2":
            model_params_d = dict(key=model_key)
            model = JAStatic2(key=model_key)
            mdl_interface_cls = JAwInterface
        case _:
            raise ValueError(f"Unknown model type: {model_label}. Choose on of {SUPPORTED_MODELS}")

    params = dict(
        training_params=dict(
            n_epochs=n_epochs,
            n_steps=0,  # 10_000
            val_every=1,
            tbptt_size=tbptt_size,
            past_size=1,
            batch_size=batch_size,
            tbptt_size_start=tbptt_size_start,
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

    wrapped_model = mdl_interface_cls(
        model=model,
        normalizer=normalizer,
        featurize=featurize,
    )

    params["model_params"] = model_params_d  # defined from outside
    params["model_params"]["key"] = params["model_params"]["key"].tolist()

    return wrapped_model, optimizer, params, data_tuple


def setup_loss(loss_label: str) -> Callable:

    match loss_label:
        case "MSE":
            loss_function = MSE_loss
        case "adapted_RMS":
            loss_function = adapted_RMS_loss
        case _:
            raise ValueError(f"Unknown loss type: {loss_label}. Choose on of {SUPPORTED_LOSSES}")

    # loss function is expected to return the value and the gradient w.r.t. to the model parameters
    loss_function = eqx.filter_value_and_grad(loss_function)

    return loss_function
