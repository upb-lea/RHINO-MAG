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
from mc2.data_management import AVAILABLE_MATERIALS, load_data_into_pandas_df

SUPPORTED_ARCHS = ["gru"]


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

    # TODO call func from data_sampling for train-val-test splits
    #  For now, just take specific material and specific frequency
    if material is None:
        material = "N87"
    data_d = load_data_into_pandas_df(material=material, number=1)

    train_b_NS = data_d[f"{material}_1_B"].to_numpy()
    train_h_NS = data_d[f"{material}_1_H"].to_numpy()
    train_t_NI = data_d[f"{material}_1_T"].to_numpy().astype(float)

    # TODO FE and normalization
    max_H = np.abs(train_b_NS).max()
    max_B = np.abs(train_h_NS).max()
    train_b_NS /= max_B
    train_h_NS /= max_H
    train_t_NI /= 100.0  # in °C

    # convert to torch tensor
    train_b_NS = torch.from_numpy(train_b_NS).to(device, torch.float32)
    train_h_NS = torch.from_numpy(train_h_NS).to(device, torch.float32)
    train_t_NI = torch.from_numpy(train_t_NI).to(device, torch.float32)

    # experimental target transform
    with torch.no_grad():
        train_h_NS = torch.tanh(1.5 * train_h_NS)

    total_len = len(train_b_NS)
    test_portion = total_len // 10

    test_b_MS = train_b_NS[-test_portion:]
    test_h_MS = train_h_NS[-test_portion:]
    test_t_MI = train_t_NI[-test_portion:]
    train_b_NS = train_b_NS[:-test_portion]
    train_h_NS = train_h_NS[:-test_portion]
    train_t_NI = train_t_NI[:-test_portion]

    N, S = train_b_NS.shape
    M = test_b_MS.shape[0]
    train_NSP = torch.cat([train_b_NS[..., None], train_t_NI.repeat(1, S)[..., None]], dim=2)
    test_MSP = torch.cat([test_b_MS[..., None], test_t_MI.repeat(1, S)[..., None]], dim=2)

    log.info(f"train size: {N}, test size: {test_portion}")

    def run_seeded_training(seed=0):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        logs = {
            "loss_trends_train": [],
            "loss_trends_val": [],
            "models_state_dict": [],
            "start_time": pd.Timestamp.now().round(freq="s"),
        }
        match model_arch:
            case "gru":
                # TODO configurize
                n_units = 5
                n_inputs = 2
                mdl = torch.nn.GRU(n_inputs, n_units, batch_first=True)

        if seed == 0:
            mdl_info = summary(
                mdl,
                input_size=((1, 1, n_inputs), (1, 1, n_units)),
            )
            log.info(f"\n{mdl_info}")
        R = n_units
        # mdl = torch.jit.script(mdl)  # does not work for GRU
        mdl = mdl.to(device)
        opt = torch.optim.NAdam(mdl.parameters(), lr=1e-3)
        loss = torch.nn.MSELoss()

        pbar = trange(n_epochs, desc=f"Seed {seed}", position=seed, unit="epoch")

        # calculate amount of tbptt-len subsequences within a chunk
        n_seqs = np.ceil(S / tbptt_size).astype(int)
        n_batches = np.ceil(N / batch_size).astype(int)
        # shuffle idx continuously
        shuffle_idx = np.arange(N)

        for epoch_i in pbar:
            mdl.train()
            np.random.shuffle(shuffle_idx)
            train_shuff_NSP = train_NSP[shuffle_idx]
            train_shuff_h_NS = train_h_NS[shuffle_idx]

            for batch_i in range(n_batches):
                batch_start, batch_end = batch_i * batch_size, min((batch_i + 1) * batch_size, N)
                train_BSP = train_shuff_NSP[batch_start:batch_end]
                train_h_BS = train_shuff_h_NS[batch_start:batch_end]
                B = len(train_BSP)
                hidden_IBI = train_h_BS[:, 0].reshape(1, B, 1)
                hidden_IBR = torch.cat(
                    [hidden_IBI, torch.zeros((1, B, R - 1), dtype=torch.float32, device=device)], dim=2
                )

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
                logs["loss_trends_train"].append(train_loss.item())
                pbar_str = f"Loss {train_loss.item():.2e}"
                mdl.eval()

                hidden_IMI = test_h_MS[:, 0].reshape(1, M, 1)
                hidden_IMR = torch.cat(
                    [hidden_IMI, torch.zeros((1, M, R - 1), dtype=torch.float32, device=device)], dim=2
                )
                val_pred_MSR, hidden_IMI = mdl(test_MSP, hidden_IMR)
                val_loss = loss(val_pred_MSR[:, :, 0], test_h_MS).cpu().numpy()
            logs["loss_trends_val"].append(val_loss)
            pbar_str += f"| val loss {val_loss:.2e}"
            pbar.set_postfix_str(pbar_str)
            if np.isnan(val_loss):
                break

        # book keeping
        if "models_arch" not in logs:
            # TODO implement configurized topology
            logs["models_arch"] = json.dumps({})
        with torch.no_grad():
            logs["all_predictions_MS"] = np.arctanh(np.clip(val_pred_MSR[:, :, 0].cpu().numpy(), -1, 1)) / 1.5 * max_H
            logs["ground_truth"] = np.arctanh(np.clip(test_h_MS.cpu().numpy(), -1, 1)) / 1.5 * max_H
        logs["end_time"] = pd.Timestamp.now().round(freq="s")
        logs["seed"] = seed
        return logs

    n_seeds = 1 if debug else n_seeds
    print(f"Parallelize over {n_seeds} seeds with {n_jobs} processes..")
    with Parallel(n_jobs=n_jobs) as prll:
        # list of dicts
        experiment_logs = prll(delayed(run_seeded_training)(s) for s in range(n_seeds))

    # TODO DB logging
    return experiment_logs
