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
from mc2.features.features_torch import Featurizer
from mc2.models.topologies_torch import DifferenceEqLayer, ExplEulerCell

SUPPORTED_ARCHS = ["gru", "expleuler"]
DO_TRANSFORM_H = False
H_FACTOR = 1.2


@torch.no_grad()
def evaluate_recursively(
    mdl: torch.nn.Module,
    tensors_d: Dict[str, Dict[str, torch.Tensor]],
    loss: torch.nn.Module,
    featurizer: Featurizer,
    max_H: float,
    n_states: int,
    set_lbl: str = "val",
    model_arch: str = "gru",
):
    target_lbl = "H_traf" if DO_TRANSFORM_H else "H"
    mdl.eval()
    val_loss = 0.0
    preds_MS_l = []
    for (model_in_MS, temp_MI), h_MS in zip(tensors_d[set_lbl]["model_in_AS_l"], tensors_d[set_lbl][target_lbl]):
        # featurize
        model_in_MSP = torch.dstack(
            featurizer.normalize(featurizer.add_fe(model_in_MS, with_original=True, temperature_MI=temp_MI))
        )
        M, S, P = model_in_MSP.shape
        hidden = h_MS[:, 0].reshape(1, M, 1)
        match model_arch:
            case "gru":
                # init hidden state with first H, shape (I, M, I)
                hidden = torch.tile(hidden, (1, 1, n_states))
            case "expleuler":
                hidden = hidden.squeeze(0)  # init hidden state with first H, shape: (M, I)

        val_pred_MSR, hidden = mdl(model_in_MSP, hidden)
        if DO_TRANSFORM_H:
            val_pred_untransformed_MS = torch.atanh(torch.clip(val_pred_MSR[:, :, 0], -0.999, 0.999)) / H_FACTOR * max_H
        else:
            val_pred_untransformed_MS = val_pred_MSR[:, :, 0] * max_H
        val_loss += loss(val_pred_untransformed_MS, h_MS * max_H).cpu().numpy()
        preds_MS_l += [val_pred_untransformed_MS]
    return val_loss / len(tensors_d[set_lbl]["model_in_AS_l"][0]), preds_MS_l


def train_recursive_nn(
    material=None,
    model_arch: str = "gru",
    debug: bool = False,
    n_epochs: int = 100,
    n_seeds: int = 5,
    n_jobs: int = 5,
    tbptt_size: int = 1024,
    batch_size: int = 256,
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

        # filter input features
        if "model_in_AS_l" not in tensors_d[set_lbl]:
            tensors_d[set_lbl]["model_in_AS_l"] = []
        tensors_d[set_lbl]["model_in_AS_l"] = list(zip(tensors_d[set_lbl]["B"], tensors_d[set_lbl]["T"]))

        del tensors_d[set_lbl]["B"]
        del tensors_d[set_lbl]["T"]
    del train_d, val_d, test_d  # free memory
    # extract number of features from add_fe function
    featurizer = Featurizer(mat_lbl=material, device=device)
    featurizer.extract_normalization_constants(tensors_d["train"]["model_in_AS_l"])

    log.info(
        f"train size: {np.sum([a.numel() for a in tensors_d['train']['H']])}, "
        f"val size: {np.sum([a.numel() for a in tensors_d['val']['H']])}, "
        f"test size: {np.sum([a.numel() for a in tensors_d['test']['H']])}"
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
                n_units = 8
                mdl = torch.nn.GRU(featurizer.n_inputs, n_units, batch_first=True)
                mdl_info_input_size = ((1, 1, featurizer.n_inputs), (1, 1, n_units))
                R = n_units
                # torch.jit.script does not work with GRU

            case "expleuler":
                mdl = DifferenceEqLayer(ExplEulerCell, n_inputs=featurizer.n_inputs)
                mdl = torch.jit.script(mdl)  # new syntax as of pytorch 1.2
                mdl_info_input_size = ((1, 1, featurizer.n_inputs), (1, 1))
                R = 1  # won't be used
        if seed == 0:
            mdl_info = summary(mdl, input_size=mdl_info_input_size)

        mdl = mdl.to(device)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        loss = torch.nn.MSELoss()

        pbar = trange(n_epochs, desc=f"Seed {seed}", position=seed, unit="epoch")

        target_lbl = "H_traf" if DO_TRANSFORM_H else "H"

        for epoch_i in pbar:
            mdl.train()
            # iterate over frequency sets
            train_loss_avg_per_epoch = 0
            for (model_in_AS, temp_AI), h_AS in zip(
                tensors_d["train"]["model_in_AS_l"], tensors_d["train"][target_lbl]
            ):
                A, S = model_in_AS.shape
                # calculate amount of tbptt-len subsequences within a chunk
                n_seqs = np.ceil(S / tbptt_size).astype(int)
                n_batches = np.ceil(A / batch_size).astype(int)
                # shuffle idx continuously
                shuffle_idx = np.arange(A)
                np.random.shuffle(shuffle_idx)
                train_shuff_AS = model_in_AS[shuffle_idx]
                train_shuff_h_AS = h_AS[shuffle_idx]
                train_shuff_temp_AI = temp_AI[shuffle_idx]
                avg_loss = 0.0
                for batch_i in range(n_batches):
                    batch_start, batch_end = batch_i * batch_size, min((batch_i + 1) * batch_size, A)
                    train_BS = train_shuff_AS[batch_start:batch_end]
                    temp_BI = train_shuff_temp_AI[batch_start:batch_end]
                    train_h_BS = train_shuff_h_AS[batch_start:batch_end]

                    # featurize
                    train_BSP = torch.dstack(
                        featurizer.normalize(featurizer.add_fe(train_BS, with_original=True, temperature_MI=temp_BI))
                    )

                    B = len(train_BSP)
                    hidden_IBI = train_h_BS[:, 0].reshape(1, B, 1)
                    """hidden_IBR = torch.cat(
                        [hidden_IBI, torch.zeros((1, B, R - 1), dtype=torch.float32, device=device)], dim=2
                    )"""
                    match model_arch:
                        case "gru":
                            hidden = torch.tile(
                                hidden_IBI, (1, 1, R)
                            )  # init hidden state with first H, shape (I, B, I)
                        case "expleuler":
                            hidden = hidden_IBI.squeeze(0)  # init hidden state with first H, shape: (B, I)

                    for seq_i in range(n_seqs):
                        seq_start, seq_end = seq_i * tbptt_size, min((seq_i + 1) * tbptt_size, S)
                        mdl.zero_grad()
                        hidden = hidden.detach()

                        train_BQP = train_BSP[:, seq_start:seq_end, :]
                        train_h_BQ = train_h_BS[:, seq_start:seq_end]
                        output_BQR, hidden = mdl(train_BQP, hidden)
                        train_loss = loss(output_BQR[:, :, 0], train_h_BQ)
                        train_loss.backward()
                        opt.step()
                        with torch.no_grad():
                            avg_loss += train_loss.cpu().numpy().item()

                avg_loss /= n_seqs * n_batches
                train_loss_avg_per_epoch += avg_loss / len(tensors_d["train"]["model_in_AS_l"][0])
            logs["loss_trends_train"].append(train_loss_avg_per_epoch)
            pbar_str = f"Loss {train_loss_avg_per_epoch:.2e}"

            val_loss, _ = evaluate_recursively(
                mdl, tensors_d, loss, featurizer, max_d["H"], R, set_lbl="val", model_arch=model_arch
            )
            logs["loss_trends_val"].append(val_loss)
            pbar_str += f"| val loss {val_loss:.2e}"
            pbar.set_postfix_str(pbar_str)
            if np.isnan(val_loss):
                break

        test_loss, test_pred_MS_l = evaluate_recursively(
            mdl, tensors_d, loss, featurizer, max_d["H"], R, set_lbl="test", model_arch=model_arch
        )
        log.info(f"Test loss seed {seed}: {test_loss:.3f} A/m")
        # book keeping
        if "models_arch" not in logs:
            # TODO implement configurized topology
            logs["models_arch"] = json.dumps({})
        logs["model_state_dict"] = mdl.cpu().state_dict()
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

# TODO
# add output layer that averages cell states to single output as alternative to always taking first cell
