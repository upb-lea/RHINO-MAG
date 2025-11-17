from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from mc2.losses import MSE_loss, adapted_RMS_loss, pinn_gru_loss
from mc2.features.features_jax import compute_fe_single, shift_signal
from mc2.data_management import MaterialSet, load_data_into_pandas_df, Normalizer

# Models
from mc2.models.NODE import HiddenStateNeuralEulerODE
from mc2.models.RNN import GRU, GRUwLinearModel
from mc2.models.jiles_atherton import (
    JAStatic,
    JAStatic2,
    JAStatic3,
    JAParamGRUlin,
    JAParamMLP,
    JAWithExternGRU,
    JAWithGRU,
    JAWithGRUlin,
    JAWithGRUlinFinal,
)
from mc2.models.linear import LinearStatic

# Interfaces
from mc2.model_interfaces.rnn_Interfaces import (
    NODEwInterface,
    RNNwInterface,
    GRUwLinearModelInterface,
    MagnetizationRNNwInterface,
)
from mc2.model_interfaces.ja_interfaces import (
    JAwInterface,
    JAParamMLPwInterface,
    JAWithGRUwInterface,
    JAWithExternGRUwInterface,
)
from mc2.model_interfaces.linear_interfaces import LinearInterface

SUPPORTED_MODELS = ["GRU", "HNODE", "JA", "PinnWithGRU"]

SUPPORTED_LOSSES = ["MSE", "adapted_RMS", "PINN_GRU"]


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
    past_size: int = 10,
    time_shift: int = 0,
    tbptt_size_start=None,  # (size, n_epochs_steps)
    **kwargs,
):
    def featurize(norm_B_past, norm_H_past, norm_B_future, temperature, time_shift):
        past_length = norm_B_past.shape[0]
        B_all = jnp.hstack([norm_B_past, norm_B_future])

        featurized_B = compute_fe_single(B_all, n_s=11, time_shift=time_shift)

        return featurized_B[past_length:]

    featurize = partial(featurize, time_shift=time_shift)

    normalizer, data_tuple = get_normalizer(
        material_name,
        featurize,
        subsample_freq,
        True,
    )

    # dynamically choose model input size:
    test_seq_length = 100
    test_out = featurize(
        norm_B_past=jnp.ones(test_seq_length),
        norm_H_past=jnp.ones(test_seq_length),
        norm_B_future=jnp.ones(test_seq_length),
        temperature=jnp.ones(1),
        time_shift=time_shift,
    )
    assert test_out.shape[0] == test_seq_length
    model_in_size = test_out.shape[-1] + 2  # (+2) due to: flux density B and temperature T

    match model_label:
        case "HNODE":
            model_params_d = dict(
                obs_dim=1,
                state_dim=8,
                action_dim=model_in_size,
                width_size=8,
                depth=2,
                obs_func_type="identity",
                key=model_key,
            )
            model = HiddenStateNeuralEulerODE(**model_params_d)
            mdl_interface_cls = NODEwInterface
        case "GRU":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = GRU(**model_params_d)
            mdl_interface_cls = RNNwInterface
        case "MagnetizationGRU":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = GRU(**model_params_d)
            mdl_interface_cls = MagnetizationRNNwInterface
        case "JAWithExternGRU":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = JAWithExternGRU(**model_params_d)
            mdl_interface_cls = JAWithExternGRUwInterface
        case "JAWithGRUlin":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = JAWithGRUlin(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAWithGRUlinFinal":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = JAWithGRUlinFinal(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAWithGRU":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = JAWithGRU(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAParamGRUlin":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = JAParamGRUlin(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "JAParamMLP":
            model_params_d = dict(hidden_size=32, depth=2, in_size=model_in_size, key=model_key)
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
        case "JA3":
            model_params_d = dict(key=model_key)
            model = JAStatic3(key=model_key)
            mdl_interface_cls = JAwInterface
        case "Linear":
            model_params_d = dict(in_size=9, out_size=1, key=model_key)
            model = LinearStatic(**model_params_d)
            mdl_interface_cls = LinearInterface
        case "GRUwLinearModel":
            # model_params_d = dict(in_size=7, hidden_size=8, linear_in_size=7, key=model_key)
            model_params_d = dict(in_size=model_in_size, hidden_size=8, linear_in_size=1, key=model_key)
            model = GRUwLinearModel(**model_params_d)
            mdl_interface_cls = GRUwLinearModelInterface
        case "PinnWithGRU":
            model_params_d = dict(hidden_size=8, input_size=7, key=model_key)
            model = PinnWithGRU(**model_params_d)
            mdl_interface_cls = GRUWithPINNInterface
        case _:
            raise ValueError(f"Unknown model type: {model_label}. Choose on of {SUPPORTED_MODELS}")

    assert (
        past_size < tbptt_size
    ), f"The trajectory is too short for the specified warm-up time. {past_size} < {tbptt_size}."
    if tbptt_size_start is not None:
        assert past_size < tbptt_size_start[0], (
            f"The initial trajectories are too short for "
            + f"the specified warm-up time. {past_size} < {tbptt_size_start[0]}."
        )

    params = dict(
        training_params=dict(
            n_epochs=n_epochs,
            n_steps=0,  # 10_000
            val_every=1,
            tbptt_size=tbptt_size,
            past_size=past_size,
            time_shift=time_shift,
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
        case "PINN_GRU":
            loss_function = pinn_gru_loss
        case _:
            raise ValueError(f"Unknown loss type: {loss_label}. Choose on of {SUPPORTED_LOSSES}")

    # loss function is expected to return the value and the gradient w.r.t. to the model parameters
    loss_function = eqx.filter_value_and_grad(loss_function)

    return loss_function
