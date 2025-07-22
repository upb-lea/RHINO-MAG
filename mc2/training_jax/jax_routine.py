import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pandas as pd
import torch
import json
import random
import logging as log
from tqdm.notebook import trange
from joblib import Parallel, delayed
from typing import Dict
from torchinfo import summary
from mc2.data_management import (
    AVAILABLE_MATERIALS,
    book_keeping,
    get_train_val_test_pandas_dicts,
    load_data_into_pandas_df,
    setup_package_logging,
    MaterialSet,
    FrequencySet,
)
from mc2.features.features_jax import add_fe
from mc2.models.RNN import BaseRNN
from mc2.training.data_sampling import draw_data_uniformly
from mc2.training.optimization import make_step

from IPython.display import clear_output, display

DO_TRANSFORM_H = True
H_FACTOR = 1.2
VAL_EVERY = 100


@torch.no_grad()
def evaluate_recursively(mdl, tensors_d, loss, device, max_H, n_states, set_lbl="val"):
    target_lbl = "H_traf" if DO_TRANSFORM_H else "H"
    mdl.eval()
    val_loss = 0.0
    preds_MS_l = []
    for model_in_MSP, h_MS in zip(tensors_d[set_lbl]["model_in_ASP"], tensors_d[set_lbl][target_lbl]):
        M, S, P = model_in_MSP.shape
        hidden_IMI = h_MS[:, 0].reshape(1, M, 1)
        """hidden_IMR = torch.cat(
            [hidden_IMI, torch.zeros((1, M, n_states - 1), dtype=torch.float32, device=device)], dim=2
        )"""
        hidden_IMR = torch.tile(hidden_IMI, (1, 1, n_states))  # init hidden state with first H
        val_pred_MSR, hidden_IMI = mdl(model_in_MSP, hidden_IMR)
        if DO_TRANSFORM_H:
            val_pred_untransformed_MS = torch.atanh(torch.clip(val_pred_MSR[:, :, 0], -0.999, 0.999)) / H_FACTOR * max_H
        else:
            val_pred_untransformed_MS = val_pred_MSR[:, :, 0] * max_H
        val_loss += loss(val_pred_untransformed_MS, h_MS * max_H).cpu().numpy()
        preds_MS_l += [val_pred_untransformed_MS]
    return val_loss / len(tensors_d[set_lbl]["model_in_ASP"]), preds_MS_l


def normalize_material_set(mat_set: MaterialSet, max_d: dict[str, float], reduce_train=False) -> MaterialSet:
    normalized_freq_sets = []
    for freq_set in mat_set.frequency_sets:

        B_norm = freq_set.B / max_d["B"]
        H_norm = freq_set.H / max_d["H"]
        T_norm = freq_set.T / max_d["T"]

        if reduce_train:
            norm_freq_set = FrequencySet(
                material_name=freq_set.material_name,
                frequency=freq_set.frequency,
                B=B_norm[::10],
                H=H_norm[::10],
                T=T_norm[::10],
            )
        else:
            norm_freq_set = FrequencySet(
                material_name=freq_set.material_name,
                frequency=freq_set.frequency,
                B=B_norm,
                H=H_norm,
                T=T_norm,
            )
        normalized_freq_sets.append(norm_freq_set)

    return MaterialSet(
        material_name=mat_set.material_name,
        frequency_sets=normalized_freq_sets,
        frequencies=mat_set.frequencies,
    )


def train_recursive_nn(
    material=None,
    debug: bool = False,
    n_steps: int = 100,
    n_seeds: int = 5,
    n_jobs: int = 5,
    tbptt_size: int = 64,
    batch_size: int = 1024,
):

    n_steps = 3 if debug else n_steps

    if material is None:
        material = "N87"
    data_dict = load_data_into_pandas_df(material=material)
    mat_set = MaterialSet.from_pandas_dict(data_dict)

    train_set, val_set, test_set = mat_set.split_into_train_val_test(
        train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=12
    )
    max_l_d = {"B": [], "H": []}
    max_d = {"B": None, "H": None, "T": 100}  # max temperature is 100°C

    for freq_set in train_set.frequency_sets:
        max_B = jnp.abs(freq_set.B).max()
        max_H = jnp.abs(freq_set.H).max()
        max_l_d["B"].append(float(max_B))
        max_l_d["H"].append(float(max_H))

    for k, max_l in max_l_d.items():
        max_v = max(max_l)
        log.info(f"max {k} value in training set: {max_v:.2f} T or A/m or °C")
        max_d[k] = max_v

    # normalize
    train_set_norm = normalize_material_set(train_set, max_d, reduce_train=debug)
    val_set_norm = normalize_material_set(val_set, max_d)
    test_set_norm = normalize_material_set(test_set, max_d)
    del train_set, val_set, test_set  # free memory

    log.info(
        f"train size: {sum(freq_set.H.shape[0] for freq_set in train_set_norm.frequency_sets)}, "
        f"val size: {sum(freq_set.H.shape[0] for freq_set in val_set_norm.frequency_sets)}, "
        f"test size: {sum(freq_set.H.shape[0] for freq_set in test_set_norm.frequency_sets)}"
    )

    # FREQ_CATEGORIES = jnp.array([50000.0, 80000.0, 125000.0, 200000.0, 320000.0, 500000.0, 800000.0])

    def featurize(B, H, T, f):
        # fes = add_fe(jnp.reshape(B, (1, -1)), n_s=tbptt_size)
        return jnp.concatenate([B, T, f], axis=-1), jnp.tanh(H_FACTOR * H)  # fes[0],

    # @eqx.filter_jit
    # def train_step(model, opt_state, train_set_norm, optimizer, featurize, key):
    #     train_loss = 0.0
    #     for freq_set in train_set_norm.frequency_sets:
    #         key, subkey = jax.random.split(key)
    #         batch_H, batch_B, batch_T, _ = draw_data_uniformly(
    #             freq_set, training_sequence_length=tbptt_size, training_batch_size=batch_size, loader_key=subkey
    #         )
    #         batch_f = freq_set.frequency * jnp.ones_like(batch_B) / 800_000  # do normlization already in advance TODO

    #         batch_x, batch_y = jax.vmap(featurize)(batch_B, batch_H, batch_T, batch_f)
    #         # print(batch_x.shape, batch_y.shape, "Training batch shape")
    #         loss, model, opt_state = make_step(model, batch_x, batch_y, optimizer, opt_state)

    #         train_loss += loss / len(train_set_norm.frequencies)
    #     return train_loss, model, opt_state

    @eqx.filter_jit
    def process_freq_set(model, opt_state, key, freq_set, optimizer):
        key, subkey = jax.random.split(key)
        batch_H, batch_B, batch_T, _ = draw_data_uniformly(
            freq_set,
            training_sequence_length=tbptt_size,
            training_batch_size=batch_size,
            loader_key=subkey,
        )
        batch_f = freq_set.frequency * jnp.ones_like(batch_B) / 800_000

        batch_x, batch_y = jax.vmap(featurize)(batch_B, batch_H, batch_T, batch_f)
        loss, model, opt_state = make_step(model, batch_x, batch_y, optimizer, opt_state)

        return loss, model, opt_state, key

    def train_step(model, opt_state, train_set_norm, optimizer, featurize, key):

        train_loss = 0.0
        for freq_set in train_set_norm.frequency_sets:
            loss, model, opt_state, key = process_freq_set(model, opt_state, key, freq_set, optimizer)
            train_loss += loss / len(train_set_norm.frequencies)

        return train_loss, model, opt_state

    # @eqx.filter_jit
    # def val_test(set, model, featurize):
    #     val_loss = 0
    #     for freq_set in set.frequency_sets:
    #         f = freq_set.frequency / 800_000
    #         batch_x, batch_y = jax.vmap(featurize)(
    #             freq_set.B[..., None],
    #             freq_set.H[..., None],
    #             jnp.broadcast_to(freq_set.T[:, None, None], freq_set.B[..., None].shape),
    #             f * jnp.ones_like(freq_set.B[..., None]),
    #         )
    #         # print(batch_x.shape, batch_y.shape)
    #         pred_y = jax.vmap(model)(batch_x)
    #         val_loss += jnp.mean((pred_y - batch_y) ** 2) / len(set.frequencies)
    #     return val_loss

    @eqx.filter_jit
    def process_freq_set_val(freq_set, model, featurize):
        f = freq_set.frequency / 800_000
        B = freq_set.B[..., None]
        H = freq_set.H[..., None]
        T = jnp.broadcast_to(freq_set.T[:, None, None], B.shape)
        F = f * jnp.ones_like(B)

        batch_x, batch_y = jax.vmap(featurize)(B, H, T, F)
        pred_y = jax.vmap(model)(batch_x)
        loss = jnp.mean((pred_y - batch_y) ** 2)
        return loss

    def val_test(set, model, featurize):

        val_loss = 0.0
        for freq_set in set.frequency_sets:
            loss = process_freq_set_val(freq_set, model, featurize)
            val_loss += loss / len(set.frequencies)

        return val_loss

    def run_seeded_training(seed):
        key = jax.random.PRNGKey(seed)
        logs = {
            "material": material,
            "loss_trends_train": [],
            "loss_trends_val": [],
            "models_state_dict": [],
            "start_time": pd.Timestamp.now().round(freq="s"),
        }

        hidden_size = 8
        in_size = 3
        out_size = 1
        model = BaseRNN(in_size, out_size, hidden_size, key=key)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(model)

        pbar = trange(n_steps, desc=f"Seed {seed}", position=seed, unit="step")
        test_loss = val_test(test_set_norm, model, featurize)
        log.info(f"Test loss seed {seed}: {test_loss:.3f} A/m")
        for step in pbar:
            train_loss = 0
            # train_data: (N, S, features), (N, S, output_features)
            key, subkey = jax.random.split(key)
            train_loss, model, opt_state = train_step(model, opt_state, train_set_norm, optimizer, featurize, subkey)
            pbar_str = f"Loss {train_loss:.2e}"
            if step % VAL_EVERY == 0:
                val_loss = val_test(val_set_norm, model, featurize)
                logs["loss_trends_val"].append(val_loss)
            pbar_str += f"| val loss {val_loss:.2e}"
            logs["loss_trends_train"].append(train_loss)
            pbar.set_postfix_str(pbar_str)

        test_loss = val_test(test_set_norm, model, featurize)
        log.info(f"Test loss seed {seed}: {test_loss:.3f} A/m")

        logs["end_time"] = pd.Timestamp.now().round(freq="s")
        logs["seed"] = seed
        return logs, model

    # n_seeds = 1 if debug else n_seeds
    # # log.info(f"Parallelize over {n_seeds} seeds with {n_jobs} processes..")
    # # with Parallel(n_jobs=n_jobs) as prll:
    # #     # list of dicts
    # #     experiment_logs_l = prll(delayed(run_seeded_training)(s) for s in range(n_seeds))
    # #experiment_logs_l = [run_seeded_training(0)]

    # import matplotlib.pyplot as plt

    # plt.plot(experiment_logs_l[0]["loss_trends_train"], label="train loss")
    # plt.plot(experiment_logs_l[0]["loss_trends_val"], label="val loss")
    # plt.show()
    # best_idx = np.argmin([l["loss_trends_val"][-1] for l in experiment_logs_l])
    # best_log = experiment_logs_l[best_idx]
    # book_keeping(best_log)

    return run_seeded_training(1)


# TODO
# wie schnell ist ein GRU in JAX? (OJS)
# Kleinere Abtastraten untersuchen (Hendrik)
# NODEs (Hendrik)
# FE function implementieren (Till)
# Target-Trafos untersuchen (Wilhelm)

# k-fold CV für 1. Nov submission muss implementiert werden
# train-val-test split mit reduzierter Datenmenge für prototyping muss implementiert werden
