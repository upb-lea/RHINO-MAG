import numpy as np
import pandas as pd
import torch
import json
import random
import logging as log
from tqdm import trange
from joblib import Parallel, delayed
from typing import Dict
from torchinfo import summary
from mc2.data_management import (
    AVAILABLE_MATERIALS,
    book_keeping,
    get_train_val_test_pandas_dicts,
    setup_package_logging,
)

SUPPORTED_ARCHS = ["gru"]
DO_TRANSFORM_H = True
H_FACTOR = 1.2


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


def train_recursive_nn(
    material=None,
    model_arch: str = "gru",
    debug: bool = False,
    n_epochs: int = 100,
    n_seeds: int = 5,
    n_jobs: int = 5,
    tbptt_size: int = 64,
    batch_size: int = 1024,
):
    assert model_arch in SUPPORTED_ARCHS, f"model arch {model_arch} not in {SUPPORTED_ARCHS}"
    assert material is None or material in AVAILABLE_MATERIALS, f"mat {material} not in {AVAILABLE_MATERIALS}"
    device = torch.device("cuda:0")
    n_epochs = 3 if debug else n_epochs

    if material is None:
        material = "N87"
    train_d, val_d, test_d = get_train_val_test_pandas_dicts(material_name=material, seed=12)

    # determine max_B and max_H, and max_T
    max_l_d = {"B": [], "H": []}
    max_d = {"B": None, "H": None, "T": 100}  # max temperature is 100°C
    for num_train_k, num_train_df in train_d.items():
        quant_lbl = num_train_k[-1].upper()
        if quant_lbl in "BH":
            max_l_d[quant_lbl] += [np.abs(num_train_df.to_numpy()).max()]
    for k, max_l in max_l_d.items():
        max_v = np.max(max_l)
        log.info(f"max {k} value in training set: {max_v:.2f} T or A/m or °C")
        max_d[k] = max_v

    # convert to torch tensor, normalize
    tensors_d = {}
    for set_d, set_lbl in zip((train_d, val_d, test_d), ("train", "val", "test")):
        for num_set_k, num_set_df in set_d.items():
            mat, set_num, quant_lbl = num_set_k.upper().split("_")
            num_set_AS = num_set_df.to_numpy() / max_d[quant_lbl]  # FE, normalize
            if debug and set_lbl == "train":
                num_set_AS = num_set_AS[::10, :]
            if set_lbl not in tensors_d:
                tensors_d[set_lbl] = {"B": [], "H": [], "H_traf": [], "T": []}
            tensors_d[set_lbl][quant_lbl].append(torch.from_numpy(num_set_AS).to(device, torch.float32))
            if DO_TRANSFORM_H and quant_lbl == "H":
                # experimental target transform
                tensors_d[set_lbl]["H_traf"].append(
                    torch.from_numpy(np.tanh(H_FACTOR * num_set_AS)).to(device, torch.float32)
                )

        # concatenate B and T into a model_in tensor
        if "model_in_ASP" not in tensors_d[set_lbl]:
            tensors_d[set_lbl]["model_in_ASP"] = []
        for b_AS, t_AI in zip(tensors_d[set_lbl]["B"], tensors_d[set_lbl]["T"]):
            A, S = b_AS.shape
            tensors_d[set_lbl]["model_in_ASP"].append(torch.cat([b_AS[..., None], t_AI.repeat(1, S)[..., None]], dim=2))
            # TODO: do FE here

        del tensors_d[set_lbl]["B"]
        del tensors_d[set_lbl]["T"]
    del train_d, val_d, test_d  # free memory
    n_inputs = tensors_d["train"]["model_in_ASP"][0].shape[-1]
    log.info(
        f"train size: {np.sum([len(a) for a in tensors_d['train']])}, "
        f"val size: {np.sum([len(a) for a in tensors_d['val']])}, "
        f"test size: {np.sum([len(a) for a in tensors_d['test']])}"
    )

    def run_seeded_training(seed=0):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        logs = {
            "material": material,
            "loss_trends_train": [],
            "loss_trends_val": [],
            "models_state_dict": [],
            "start_time": pd.Timestamp.now().round(freq="s"),
        }
        match model_arch:
            case "gru":
                # TODO configurize
                n_units = 3
                mdl = torch.nn.GRU(n_inputs, n_units, batch_first=True)

        if seed == 0:
            mdl_info = summary(
                mdl,
                input_size=((1, 1, n_inputs), (1, 1, n_units)),
            )
            # log.info(f"\n{mdl_info}")
        R = n_units
        # mdl = torch.jit.script(mdl)  # does not work for GRU
        mdl = mdl.to(device)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        loss = torch.nn.MSELoss()

        pbar = trange(n_epochs, desc=f"Seed {seed}", position=seed, unit="epoch")

        target_lbl = "H_traf" if DO_TRANSFORM_H else "H"

        for epoch_i in pbar:
            mdl.train()
            # iterate over frequency sets
            train_loss_avg_per_epoch = 0
            for model_in_MSP, h_MS in zip(tensors_d["train"]["model_in_ASP"], tensors_d["train"][target_lbl]):
                A, S, P = model_in_MSP.shape
                # calculate amount of tbptt-len subsequences within a chunk
                n_seqs = np.ceil(S / tbptt_size).astype(int)
                n_batches = np.ceil(A / batch_size).astype(int)
                # shuffle idx continuously
                shuffle_idx = np.arange(A)
                np.random.shuffle(shuffle_idx)
                train_shuff_NSP = model_in_MSP[shuffle_idx]
                train_shuff_h_NS = h_MS[shuffle_idx]
                avg_loss = 0.0
                for batch_i in range(n_batches):
                    batch_start, batch_end = batch_i * batch_size, min((batch_i + 1) * batch_size, A)
                    train_BSP = train_shuff_NSP[batch_start:batch_end]
                    train_h_BS = train_shuff_h_NS[batch_start:batch_end]
                    B = len(train_BSP)
                    hidden_IBI = train_h_BS[:, 0].reshape(1, B, 1)
                    """hidden_IBR = torch.cat(
                        [hidden_IBI, torch.zeros((1, B, R - 1), dtype=torch.float32, device=device)], dim=2
                    )"""
                    hidden_IBR = torch.tile(hidden_IBI, (1, 1, R))  # init hidden state with first H

                    for seq_i in range(n_seqs):
                        seq_start, seq_end = seq_i * tbptt_size, min((seq_i + 1) * tbptt_size, S)
                        mdl.zero_grad()
                        hidden_IBR = hidden_IBR.detach()

                        train_BQP = train_BSP[:, seq_start:seq_end, :]
                        train_h_BQ = train_h_BS[:, seq_start:seq_end]
                        output_BQR, hidden_IBR = mdl(train_BQP, hidden_IBR)
                        train_loss = loss(output_BQR[:, :, 0], train_h_BQ)
                        train_loss.backward()
                        opt.step()
                        with torch.no_grad():
                            avg_loss += train_loss.cpu().numpy().item()

                avg_loss /= n_seqs * n_batches
                train_loss_avg_per_epoch += avg_loss / len(tensors_d["train"]["model_in_ASP"])
            logs["loss_trends_train"].append(train_loss_avg_per_epoch)
            pbar_str = f"Loss {train_loss_avg_per_epoch:.2e}"

            val_loss, _ = evaluate_recursively(mdl, tensors_d, loss, device, max_d["H"], R, set_lbl="val")
            logs["loss_trends_val"].append(val_loss)
            pbar_str += f"| val loss {val_loss:.2e}"
            pbar.set_postfix_str(pbar_str)
            if np.isnan(val_loss):
                break

        test_loss, test_pred_MS_l = evaluate_recursively(mdl, tensors_d, loss, device, max_d["H"], R, set_lbl="test")
        log.info(f"Test loss seed {seed}: {test_loss:.3f} A/m")
        # book keeping
        if "models_arch" not in logs:
            # TODO implement configurized topology
            logs["models_arch"] = json.dumps({})
        with torch.no_grad():
            for i, (gt, pred) in enumerate(zip(tensors_d[set_lbl][target_lbl], test_pred_MS_l)):
                logs[f"predictions_MS_{i}"] = pred.cpu().numpy()
                logs[f"ground_truth_MS_{i}"] = gt.cpu().numpy() * max_d["H"]
        logs["end_time"] = pd.Timestamp.now().round(freq="s")
        logs["seed"] = seed
        return logs

    n_seeds = 1 if debug else n_seeds
    log.info(f"Parallelize over {n_seeds} seeds with {n_jobs} processes..")
    with Parallel(n_jobs=n_jobs) as prll:
        # list of dicts
        experiment_logs_l = prll(delayed(run_seeded_training)(s) for s in range(n_seeds))

    best_idx = np.argmin([l["loss_trends_val"][-1] for l in experiment_logs_l])
    best_log = experiment_logs_l[best_idx]
    book_keeping(best_log)

    # TODO DB logging
    return experiment_logs_l


# TODO
# wie schnell ist ein GRU in JAX? (OJS)
# Kleinere Abtastraten untersuchen (Hendrik)
# NODEs (Hendrik)
# FE function implementieren (Till)
# Target-Trafos untersuchen (Wilhelm)

# k-fold CV für 1. Nov submission muss implementiert werden
# train-val-test split mit reduzierter Datenmenge für prototyping muss implementiert werden
