import re

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse
from tqdm.auto import tqdm

from dataset import graph_week_collate_fn
from models.GATransformerdecoder import GATTransformerODWeek
from train.trainer import MetroGraphWeekLM


def build_static_edge_index(od_csv):
    od_df = pd.read_csv(od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    static_edge_index, _ = dense_to_sparse(adj)
    return static_edge_index


def _infer_graph_week_config(state_dict):
    num_nodes, node_feat_dim = state_dict["model.node_embed.weight"].shape
    weekday_emb_dim = state_dict["model.weekday_embed.weight"].shape[1]
    gat_hid_dim = state_dict["model.start_token"].shape[-1]
    num_future_steps = state_dict["model.future_step_emb.weight"].shape[0]
    time_enc_dim = state_dict["model.time_enc_linear.weight"].shape[1]
    heads = state_dict["model.short_spatial_encoder.gat1.gat1.att"].shape[1]
    use_gate_head = "model.gate_origin_proj.weight" in state_dict

    layer_ids = {
        int(match.group(1))
        for key in state_dict
        for match in [re.search(r"model\.decoder\.transformer_decoder\.layers\.(\d+)\.", key)]
        if match is not None
    }
    decode_num_layers = max(layer_ids) + 1 if layer_ids else 1

    node_latlon = state_dict.get("model.node_latlon")
    if node_latlon is not None:
        node_latlon = torch.zeros_like(node_latlon)

    return {
        "num_nodes": num_nodes,
        "node_feat_dim": node_feat_dim,
        "gat_hid_dim": gat_hid_dim,
        "heads": heads,
        "decode_num_layers": decode_num_layers,
        "num_future_steps": num_future_steps,
        "weekday_emb_dim": weekday_emb_dim,
        "time_enc_dim": time_enc_dim,
        "node_latlon": node_latlon,
        "use_gate_head": use_gate_head,
    }


def load_graph_week_model(ckpt_path, device, od_csv, gate_tau=0.6, lr=5e-4):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = GATTransformerODWeek(**_infer_graph_week_config(checkpoint["state_dict"]))
    lightning_module = MetroGraphWeekLM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model=model,
        loss=torch.nn.SmoothL1Loss(),
        lr=lr,
        gate_tau=gate_tau,
    )
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    static_edge_index = build_static_edge_index(od_csv)
    return lightning_module, static_edge_index


def _build_graph_week_loader(dataset, static_edge_index, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: graph_week_collate_fn(batch, static_edge_index),
    )


def _run_graph_week_batch(lightning_module, batch, device):
    static_edge_index, batch_graph, batch_size, hist_steps, labels, time_hist, time_fut, weekday = batch
    static_edge_index = static_edge_index.to(device)
    batch_graph = batch_graph.to(device)
    labels = labels.to(device)
    time_hist = time_hist.to(device)
    time_fut = time_fut.to(device)
    weekday = weekday.to(device)

    with torch.no_grad():
        output = lightning_module.model(
            static_edge_index,
            batch_graph,
            batch_size,
            hist_steps,
            time_hist,
            time_fut,
            weekday,
        )
        if isinstance(output, tuple):
            mag_log, gate_logits = output
            mag_log_hard, _ = lightning_module._apply_gate(mag_log, gate_logits)
        else:
            mag_log_hard = output

    pred_full = torch.expm1(torch.clamp(mag_log_hard, min=0.0))
    true_full = torch.expm1(labels)
    return pred_full, true_full


def run_graph_week_local_inference(
    lightning_module,
    dataset,
    static_edge_index,
    device,
    od_i,
    od_j,
    batch_size,
    num_workers,
    progress_desc=None,
):
    loader = _build_graph_week_loader(dataset, static_edge_index, batch_size, num_workers)
    preds = []
    trues = []

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, total=len(loader))

    for batch in iterator:
        pred_full, true_full = _run_graph_week_batch(lightning_module, batch, device)
        preds.append(pred_full[:, :, od_i, od_j].cpu())
        trues.append(true_full[:, :, od_i, od_j].cpu())

    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def evaluate_graph_week_full_network(
    lightning_module,
    dataset,
    static_edge_index,
    device,
    batch_size,
    num_workers,
    progress_desc=None,
):
    loader = _build_graph_week_loader(dataset, static_edge_index, batch_size, num_workers)

    total_sq_error = 0.0
    total_abs_error = 0.0
    total_true_abs = 0.0
    total_count = 0

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, total=len(loader))

    for batch in iterator:
        pred_full, true_full = _run_graph_week_batch(lightning_module, batch, device)
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
