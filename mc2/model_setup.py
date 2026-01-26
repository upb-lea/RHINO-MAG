"""Setup of models based on specified parameterization.

This code is generally used to create the model based on the parameterization of training
scripts and to reconstruct models that have been stored to disk.
"""

from typing import Callable
from functools import partial
from copy import deepcopy

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from mc2.losses import MSE_loss, adapted_RMS_loss
from mc2.features.features_jax import compute_fe_single, db_dt, d2b_dt2
from mc2.data_management import MaterialSet, Normalizer

# Models
from mc2.features.features_jax import compute_fe_single
from mc2.data_management import MaterialSet
from mc2.model_interfaces.rnn_interfaces import (
    NODEwInterface,
    RNNwInterface,
)
from mc2.models.NODE import HiddenStateNeuralEulerODE
from mc2.models.RNN import GRU, GRUwLinearModel, VectorfieldGRU, GRUaroundLinearModel, ExpGRU
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
    JADirectParamGRU,
    JAEnsemble,
    GRUWithJA,
    LFRWithGRUJA,
)
from mc2.models.linear import LinearStatic
from mc2.models.dummy_model import DummyModel

# Interfaces
from mc2.model_interfaces.model_interface import ModelInterface
from mc2.model_interfaces.rnn_interfaces import (
    NODEwInterface,
    RNNwInterface,
    GRUwLinearModelInterface,
    MagnetizationRNNwInterface,
    VectorfieldGRUInterface,
    GRUaroundLinearModelInterface,
)
from mc2.model_interfaces.ja_interfaces import (
    JAwInterface,
    JAParamMLPwInterface,
    JAWithGRUwInterface,
    JAWithExternGRUwInterface,
    GRUWithJAwInterface,
    LFRWithGRUJAwInterface,
)
from mc2.model_interfaces.linear_interfaces import LinearInterface
from mc2.model_interfaces.dummy_model_interface import DummyModelInterface

SUPPORTED_MODELS = ["GRU{hidden-size}", "HNODE", "JA"]
SUPPORTED_LOSSES = ["MSE", "adapted_RMS"]


def setup_dataset(
    material_name: str,
    subsampling_freq: int,
    use_all_data: bool = False,
) -> tuple[MaterialSet, MaterialSet, MaterialSet]:
    """Loads the material data from disk and splits it into a train, eval, and test set

    Args:
        material_name (str): The name of the material. See `mc2.datamanagement.AVAILABLE_MATERIALS`.
        subsampling_freq (int): The frequency with which the data set should be subsampled after loading.
            `1` returns the data set as is, `2` only returns every other element, etc.
        use_all_data (bool): Whether all data should be used in the training script or a (70/15/15)
            training/eval/test split should be done

    Returns:
        The data tuple of `(training_set, eval_set, test_set)`
    """

    mat_set = MaterialSet.from_material_name(material_name)
    mat_set = mat_set.subsample(sampling_freq=subsampling_freq)
    if use_all_data:
        train_set = deepcopy(mat_set)
        val_set = None
        test_set = None
    else:
        train_set, val_set, test_set = mat_set.split_into_train_val_test(
            train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0
        )
    return train_set, val_set, test_set


def setup_normalizer(
    train_set: MaterialSet,
    featurize: Callable,
    transform_H: bool,
) -> tuple[Normalizer, tuple[MaterialSet | None, MaterialSet | None, MaterialSet | None]]:
    """Builds a normalizer for the given train set

    Args:
        featurize (Callable): A callable function that takes (B_past, H_past, B_future, T, timeshift)
            and returns a feature vector
        transform_H (bool): Whether a `tanh` transformation is to be applied to the data before it is
            fed into the model

    Returns:
        The `Normalizer` object
    """

    train_set_norm = train_set.normalize(transform_H=transform_H, featurize=featurize)
    normalizer = train_set_norm.normalizer
    return normalizer


def setup_featurize(
    disable_features: bool,
    dyn_avg_kernel_size: int,
    time_shift: int,
) -> Callable:
    """Setup the `featurize` function.

    Args:
        disable_features (bool | str): One of (True, False, "reduce"), True uses no features, False uses all default features,
            "reduce" uses the dB/dt and d^2 B / dt^2 as features.
        dyn_avg_kernel_size (int): The kernel size of the dynamic average feature.
        time_shift (int): When specifying a value `!=0`, a feature is added where the `B` trajectory is shifted by that
            number of time steps

    Returns:
        Function that adds features to the input material data

    """
    if disable_features == True:
        raise NotImplementedError("This is likely not working as intended")

        def featurize(norm_B_past, norm_H_past, norm_B_future, temperature, time_shift):
            return norm_B_future[..., None]

    elif disable_features == "reduce":

        def featurize(norm_B_past, norm_H_past, norm_B_future, temperature, time_shift):
            past_length = norm_B_past.shape[0]
            B_all = jnp.hstack([norm_B_past, norm_B_future])
            db = db_dt(B_all)
            d2b = d2b_dt2(B_all)
            featurized_B = jnp.stack((db, d2b), axis=-1)
            return featurized_B[past_length:]

    elif not disable_features:

        def featurize(norm_B_past, norm_H_past, norm_B_future, temperature, time_shift):
            past_length = norm_B_past.shape[0]
            B_all = jnp.hstack([norm_B_past, norm_B_future])
            featurized_B = compute_fe_single(B_all, n_s=dyn_avg_kernel_size, time_shift=time_shift)
            return featurized_B[past_length:]

    else:
        raise ValueError("Option 'disable_features' with value '{disable_features}' cannot be processed.")
    featurize = partial(featurize, time_shift=time_shift)
    return featurize


def setup_model(
    model_label: str,
    model_key: jax.random.PRNGKey,
    normalizer: Normalizer,
    featurize: Callable,
    time_shift: int = 0.0,
) -> tuple[ModelInterface, dict]:
    """
    Creates the model wrapped into its model interface from the provided parameterization.

    Args:
        model_label (str): Identifier of the model types to be created.
        model_key (jax.random.PRNGKey): Pseudo random number generation key for the creation of the model.
            This key is derived from the key initially given into the algorithm, if `setup_model` is used
            from within the training script.
        normalizer (Normalizer): Normalizer object used to normalize the material data.
        featurize (Callable): Featurize function used to add features to the input of the model.
        time_shift (int): When specifying a value `!=0`, a feature is added where the `B` trajectory is shifted by that
            number of time steps

    Returns:
        The ModelInterface (i.e. the wrapped model) and the model parameterization as a dict
    """

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
        case label if label.startswith("GRU") and label[3:].isdigit():
            hidden_size = int(label[3:])
            model_params_d = dict(hidden_size=hidden_size, in_size=model_in_size, key=model_key)
            model = GRU(**model_params_d)
            mdl_interface_cls = RNNwInterface
        case label if label.startswith("ExpGRU") and label[6:].isdigit():
            hidden_size = int(label[6:])
            model_params_d = dict(hidden_size=hidden_size, in_size=model_in_size, key=model_key)
            model = ExpGRU(**model_params_d)
            mdl_interface_cls = RNNwInterface
        case "MagnetizationGRU":
            model_params_d = dict(hidden_size=8, in_size=model_in_size, key=model_key)
            model = GRU(**model_params_d)
            mdl_interface_cls = MagnetizationRNNwInterface
        case "VectorfieldGRU":
            model_params_d = dict(n_locs=9, in_size=model_in_size, key=model_key)
            model = VectorfieldGRU(**model_params_d)
            mdl_interface_cls = VectorfieldGRUInterface
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
        case "GRUWithJA":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = GRUWithJA(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = GRUWithJAwInterface
        case "LFRWithGRUJA":
            model_params_d = dict(hidden_size=8, in_size=7, key=model_key)
            model = LFRWithGRUJA(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = LFRWithGRUJAwInterface
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
            # lr_params = dict(
            #     init_value=1e-1,
            #     transition_steps=1_000_000,
            #     transition_begin=2_000,
            #     decay_rate=0.1,
            #     end_value=1e-4,
            # )
        case "JA2":
            model_params_d = dict(key=model_key)
            model = JAStatic2(key=model_key)
            mdl_interface_cls = JAwInterface
        case "JA3":
            model_params_d = dict(key=model_key)
            model = JAStatic3(key=model_key)
            mdl_interface_cls = JAwInterface
        case "JAEnsemble":
            model_params_d = dict(
                key=model_key,
                n_models=100,
            )
            model = JAEnsemble(**model_params_d)
            mdl_interface_cls = JAwInterface
        case "Linear":
            in_size = 3
            # feat_in_size = 3 * (test_out.shape[-1] + 1)

            # model_params_d = dict(in_size=in_size, feat_in_size=feat_in_size, out_size=1, key=model_key)
            model_params_d = dict(in_size=in_size, out_size=1, key=model_key)
            model = LinearStatic(**model_params_d)
            mdl_interface_cls = LinearInterface
        case "GRUwLinearModel":
            # model_params_d = dict(in_size=7, hidden_size=8, linear_in_size=7, key=model_key)
            model_params_d = dict(in_size=model_in_size, hidden_size=8, linear_in_size=1, key=model_key)
            model = GRUwLinearModel(**model_params_d)
            mdl_interface_cls = GRUwLinearModelInterface
        case "GRUaroundLinearModel":
            model_params_d = dict(in_size=model_in_size, hidden_size=3, linear_in_size=3, key=model_key)
            model = GRUaroundLinearModel(**model_params_d)
            mdl_interface_cls = GRUaroundLinearModelInterface
        case "JADirectParamGRU":
            model_params_d = dict(in_size=model_in_size + 1, hidden_size=8, key=model_key)
            model = JADirectParamGRU(normalizer=normalizer, **model_params_d)
            mdl_interface_cls = JAWithGRUwInterface
        case "DummyModel":
            model_params_d = dict(key=model_key)
            model = DummyModel(key=model_key)
            mdl_interface_cls = DummyModelInterface
        case _:
            raise ValueError(f"Unknown model type: {model_label}. Choose on of {SUPPORTED_MODELS}")

    wrapped_model = mdl_interface_cls(
        model=model,
        normalizer=normalizer,
        featurize=featurize,
    )

    return wrapped_model, model_params_d


def setup_loss(loss_label: str) -> Callable:
    """Returns the callable loss function based on the provided string identifier."""

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


def setup_experiment(
    model_label: str,
    material_name: str,
    loss_type: str,
    model_key: jax.random.PRNGKey,
    subsample_freq: int = 1,
    n_epochs: int = 100,
    tbptt_size: int = 1024,
    batch_size: int = 256,
    past_size: int = 10,
    time_shift: int = 0,
    tbptt_size_start: tuple[int, int] | None = None,  # (size, n_epochs_steps)
    disable_features: bool | str = False,
    transform_H: bool = False,
    noise_on_data: float = 0.0,
    dyn_avg_kernel_size: int = 11,
    use_all_data: bool = False,
    val_every: int = 1,
    **kwargs,
) -> tuple[ModelInterface, optax.GradientTransformation, dict, tuple[MaterialSet, MaterialSet, MaterialSet]]:
    """Create everything necessary for a training experiment from the given parameterization.

    Args:
        model_label (str): Identifier of the model types to be created.
        material_name (str): The name of the material. See `mc2.datamanagement.AVAILABLE_MATERIALS`.
        loss_type (str): Idenfier for the type of training loss to use.
        model_key (jax.random.PRNGKey): Pseudo random number generation key for the creation of the model.
            This key is derived from the key initially given into the algorithm, if `setup_model` is used
            from within the training script.
        subsample_freq (int): The frequency with which the data set should be subsampled after loading.
            `1` returns the data set as is, `2` only returns every other element, etc.
        n_epochs (int): The number of epochs to train for.
        tbptt_size (int): Length of the sequences to process per parameter update (i.e., per gradient calculation).
        batch_size (int): Number of parallel sequences to process per parameter update (i.e., per gradient calculation).
        past_size (int): Number of warmup steps before the prediction starts.
        time_shift (int): When specifying a value `!=0`, a feature is added where the `B` trajectory is shifted by that
            number of time steps
        tbptt_size_start (tuple[int, int]): Optional training with specified sequence length (first element of tuple)
            and the number of epochs to train with this sequence length (second element of tuple). This might be helpful when
            the model diverges on the full sequence length and needs to start training with shorter sequences to stabilize
            first.
        disable_features (bool | str): One of (True, False, "reduce"), True uses no features, False uses all default features,
            "reduce" uses the dB/dt and d^2 B / dt^2 as features.
        transform_H (bool): Whether a tanh transform for H should be utilized.
        noise_on_data (float): The standard deviation of noise added to the `B` trajectories.
        dyn_avg_kernel_size (int): The kernel size of the dynamic average feature.
        use_all_data (bool): Whether all data should be used for training or if instead a train, eval, test split should be performed.
        val_every (int): How often the a validation loss should be computed.

    Returns:
        The model with the proper interface, the parameter optimizer, the loss function, a dict of parameters,
            and the data tuple containing (train_set, eval_set, test_set).
    """
    if "data_tuple" in kwargs:
        data_tuple = kwargs.pop("data_tuple")
    else:
        data_tuple = setup_dataset(
            material_name=material_name,
            subsampling_freq=subsample_freq,
            use_all_data=use_all_data,
        )

    featurize = setup_featurize(
        disable_features=disable_features,
        dyn_avg_kernel_size=dyn_avg_kernel_size,
        time_shift=time_shift,
    )

    if "normalizer" in kwargs:
        normalizer = kwargs.pop("normalizer")
    else:
        # get_normalizer nur aufrufen, wenn sie nicht in kwargs sind
        normalizer = setup_normalizer(
            data_tuple[0],
            featurize,
            transform_H,
        )

    wrapped_model, model_params_d = setup_model(
        model_label=model_label,
        model_key=model_key,
        normalizer=normalizer,
        featurize=featurize,
        time_shift=time_shift,
    )

    loss_function = setup_loss(loss_label=loss_type)

    # validate parameterization
    assert (
        past_size < tbptt_size
    ), f"The trajectory is too short for the specified warm-up time. {past_size} < {tbptt_size}."
    if tbptt_size_start is not None:
        assert past_size < tbptt_size_start[0], (
            f"The initial trajectories are too short for "
            + f"the specified warm-up time. {past_size} < {tbptt_size_start[0]}."
        )

    lr_params = None
    params = dict(
        training_params=dict(
            n_epochs=n_epochs,
            n_steps=0,  # 10_000
            val_every=val_every,
            tbptt_size=tbptt_size,
            past_size=past_size,
            time_shift=time_shift,
            batch_size=batch_size,
            tbptt_size_start=tbptt_size_start,
            noise_on_data=noise_on_data,
            dyn_avg_kernel_size=dyn_avg_kernel_size,
            transform_H=transform_H,
            disable_features=disable_features,
            use_all_data=use_all_data,
        ),
    )

    if lr_params == None:
        params["lr_params"] = dict(
            init_value=1e-3,
            transition_steps=1_000_000,
            transition_begin=2_000,
            decay_rate=0.1,
            end_value=1e-4,
        )
    else:
        params["lr_params"] = lr_params

    lr_schedule = optax.schedules.exponential_decay(**params["lr_params"])
    optimizer = optax.adam(lr_schedule)

    params["model_params"] = model_params_d  # defined from outside
    params["model_params"]["key"] = params["model_params"]["key"].tolist()
    params["loss_type"] = loss_type
    params["material_name"] = material_name
    params["model_type"] = model_label

    return wrapped_model, optimizer, loss_function, params, data_tuple
