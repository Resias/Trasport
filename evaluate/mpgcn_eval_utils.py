import math
import os
import re
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from SCIE_Benchmark.MPGCN import MPGCN, get_support_K
from train.trainer import MPGCNLM

TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

from train_mpgcn import load_dynamic_graph_bank, load_static_supports


def _infer_mpgcn_config(state_dict, num_nodes):
    branch_ids = {
        int(match.group(1))
        for key in state_dict
        for match in [re.search(r"model\.branch_models\.(\d+)\.", key)]
        if match is not None
    }
    if not branch_ids:
        raise ValueError("Could not infer MPGCN branch count from checkpoint state_dict.")

    lstm_layer_ids = {
        int(match.group(1))
        for key in state_dict
        for match in [re.search(r"model\.branch_models\.0\.temporal\.weight_ih_l(\d+)", key)]
        if match is not None
    }
    if not lstm_layer_ids:
        raise ValueError("Could not infer MPGCN LSTM layer count from checkpoint state_dict.")

    spatial_layer_ids = {
        int(match.group(1))
        for key in state_dict
        for match in [re.search(r"model\.branch_models\.0\.spatial\.(\d+)\.W", key)]
        if match is not None
    }
    if not spatial_layer_ids:
        raise ValueError("Could not infer MPGCN GCN layer count from checkpoint state_dict.")

    input_dim = state_dict["model.branch_models.0.temporal.weight_ih_l0"].shape[1]
    lstm_hidden_dim = state_dict["model.branch_models.0.temporal.weight_hh_l0"].shape[1]
    gcn_hidden_dim = state_dict["model.branch_models.0.spatial.0.W"].shape[1]

    first_spatial_width = state_dict["model.branch_models.0.spatial.0.W"].shape[0]
    if first_spatial_width % max(lstm_hidden_dim, 1) != 0:
        raise ValueError(
            "Could not infer MPGCN support size K from checkpoint: "
            f"first spatial width={first_spatial_width}, lstm_hidden_dim={lstm_hidden_dim}."
        )

    k_sq = first_spatial_width // lstm_hidden_dim
    support_k = math.isqrt(k_sq)
    if support_k * support_k != k_sq:
        raise ValueError(
            "Could not infer MPGCN support size K from checkpoint: "
            f"derived K^2={k_sq} is not a perfect square."
        )

    return {
        "M": max(branch_ids) + 1,
        "K": support_k,
        "input_dim": input_dim,
        "lstm_hidden_dim": lstm_hidden_dim,
        "lstm_num_layers": max(lstm_layer_ids) + 1,
        "gcn_hidden_dim": gcn_hidden_dim,
        "gcn_num_layers": max(spatial_layer_ids) + 1,
        "num_nodes": num_nodes,
        "use_bias": True,
        "activation": torch.nn.ReLU,
    }


def load_mpgcn_model(
    ckpt_path,
    device,
    train_dataset,
    od_csv,
    dynamic_graph_cache="",
    dynamic_bin_size=60,
    kernel_type="random_walk_diffusion",
    cheby_order=1,
    lr=1e-4,
    weight_decay=0.0,
    train_rollout_steps=1,
):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    sample = train_dataset[0]
    num_nodes = sample["x"].shape[-1]

    model_kwargs = _infer_mpgcn_config(checkpoint["state_dict"], num_nodes)
    expected_k = get_support_K(kernel_type, cheby_order)
    if model_kwargs["K"] != expected_k:
        raise ValueError(
            "MPGCN checkpoint support size does not match evaluation graph settings. "
            f"checkpoint K={model_kwargs['K']}, "
            f"kernel_type={kernel_type}, cheby_order={cheby_order}, expected K={expected_k}."
        )

    static_supports = load_static_supports(od_csv, kernel_type, cheby_order)
    if static_supports.shape[-1] != num_nodes:
        raise ValueError(
            f"Adjacency node count {static_supports.shape[-1]} does not match dataset node count {num_nodes}."
        )

    origin_dynamic_supports, destination_dynamic_supports, _meta = load_dynamic_graph_bank(
        cache_path=dynamic_graph_cache,
        dataset=train_dataset,
        dynamic_bin_size=dynamic_bin_size,
        kernel_type=kernel_type,
        cheby_order=cheby_order,
    )

    model = MPGCN(**model_kwargs)
    lightning_module = MPGCNLM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model=model,
        static_supports=static_supports,
        origin_dynamic_supports=origin_dynamic_supports,
        destination_dynamic_supports=destination_dynamic_supports,
        loss=torch.nn.MSELoss(),
        lr=lr,
        weight_decay=weight_decay,
        train_rollout_steps=train_rollout_steps,
    )
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    return lightning_module


def _build_mpgcn_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _run_mpgcn_batch(lightning_module, batch, device):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    future_keys = batch["future_keys"].to(device)

    with torch.no_grad():
        pred_log = lightning_module._rollout(x, future_keys, y.shape[1])

    pred_full = torch.expm1(torch.clamp(pred_log, min=0.0))
    true_full = torch.expm1(y)
    return pred_full, true_full


def run_mpgcn_local_inference(
    lightning_module,
    dataset,
    device,
    od_i,
    od_j,
    batch_size,
    num_workers,
    progress_desc=None,
):
    loader = _build_mpgcn_loader(dataset, batch_size, num_workers)
    preds = []
    trues = []

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, total=len(loader))

    for batch in iterator:
        pred_full, true_full = _run_mpgcn_batch(lightning_module, batch, device)
        preds.append(pred_full[:, :, od_i, od_j].cpu())
        trues.append(true_full[:, :, od_i, od_j].cpu())

    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def evaluate_mpgcn_full_network(
    lightning_module,
    dataset,
    device,
    batch_size,
    num_workers,
    progress_desc=None,
):
    loader = _build_mpgcn_loader(dataset, batch_size, num_workers)

    total_sq_error = 0.0
    total_abs_error = 0.0
    total_true_abs = 0.0
    total_count = 0

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, total=len(loader))

    for batch in iterator:
        pred_full, true_full = _run_mpgcn_batch(lightning_module, batch, device)
        diff = (true_full - pred_full).reshape(-1)
        total_sq_error += torch.sum(diff ** 2).item()
        total_abs_error += torch.sum(torch.abs(diff)).item()
        total_true_abs += torch.sum(torch.abs(true_full.reshape(-1))).item()
        total_count += diff.numel()

    mse = total_sq_error / total_count
    rmse = mse ** 0.5
    mae = total_abs_error / total_count
    wmape = (total_abs_error / (total_true_abs + 1e-8)) * 100.0
    return mse, rmse, mae, wmape
