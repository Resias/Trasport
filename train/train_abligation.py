import argparse
import datetime
import gc
import json
import os
import sys
import time
from collections import OrderedDict

import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torch_geometric.utils import dense_to_sparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_dataset
from models.GATransformerdecoder import GATTransformerODWeek, PositionalEncoding, TimeTransformerDecoder
from trainer import balanced_huber, smape


ABLATION_REGISTRY = OrderedDict(
    {
        "no_geo": {
            "description": "Remove geographic lat/lon signal.",
            "use_geo_feature": False,
        },
        "no_weekday": {
            "description": "Remove weekday embedding signal.",
            "use_weekday_feature": False,
        },
        "no_time_enc": {
            "description": "Remove both historical and future time encodings.",
            "use_time_hist_feature": False,
            "use_time_fut_feature": False,
        },
        "no_future_time_enc": {
            "description": "Keep historical time encoding but remove decoder future-time hint.",
            "use_time_fut_feature": False,
        },
        "no_gate": {
            "description": "Disable gate head and train dense magnitude-only regression.",
            "use_gate_head": False,
        },
        "gat_encoder": {
            "description": "Replace GATv2 encoders with vanilla GAT encoders.",
            "encoder_type": "gat",
        },
        "gcn_encoder": {
            "description": "Replace GATv2 encoders with GCN encoders.",
            "encoder_type": "gcn",
        },
        "short_only": {
            "description": "Use only the short static branch.",
            "static_branch_mode": "short_only",
        },
        "long_only": {
            "description": "Use only the long static branch.",
            "static_branch_mode": "long_only",
        },
        "no_dynamic": {
            "description": "Remove the dynamic OD graph encoder.",
            "use_dynamic_encoder": False,
        },
        "no_factorization": {
            "description": "Replace origin-destination factorization with direct row-wise prediction.",
            "factorization_mode": "rowwise",
        },
        "no_reverse_edge": {
            "description": "Do not append reverse edges in dynamic OD graph construction.",
            "use_reverse_edges": False,
        },
        "simple_temporal": {
            "description": "Replace the Transformer decoder with a simpler GRU-based temporal head.",
            "temporal_model": "simple",
        },
        "meta_off": {
            "description": "Remove geo, weekday, and all time encoding signals.",
            "use_geo_feature": False,
            "use_weekday_feature": False,
            "use_time_hist_feature": False,
            "use_time_fut_feature": False,
        },
    }
)

BASE_ABLATION_FLAGS = {
    "use_geo_feature": True,
    "use_weekday_feature": True,
    "use_time_hist_feature": True,
    "use_time_fut_feature": True,
    "use_gate_head": True,
    "encoder_type": "gatv2",
    "static_branch_mode": "dual",
    "use_dynamic_encoder": True,
    "factorization_mode": "factorized",
    "temporal_model": "transformer",
    "use_reverse_edges": True,
}

PROGRESSIVE_STAGE_REGISTRY = OrderedDict(
    {
        "S0_minimal": {
            "description": "Minimal graph-temporal backbone: short-only static branch, no dynamic encoder, simple temporal head, direct prediction, no metadata, no gate.",
            "flags": {
                "use_geo_feature": False,
                "use_weekday_feature": False,
                "use_time_hist_feature": False,
                "use_time_fut_feature": False,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "short_only",
                "use_dynamic_encoder": False,
                "factorization_mode": "rowwise",
                "temporal_model": "simple",
                "use_reverse_edges": True,
            },
        },
        "S1_factorization": {
            "description": "S0 + factorized origin-destination output head.",
            "flags": {
                "use_geo_feature": False,
                "use_weekday_feature": False,
                "use_time_hist_feature": False,
                "use_time_fut_feature": False,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "short_only",
                "use_dynamic_encoder": False,
                "factorization_mode": "factorized",
                "temporal_model": "simple",
                "use_reverse_edges": True,
            },
        },
        "S2_multiscale_static": {
            "description": "S1 + dual short/long static branches.",
            "flags": {
                "use_geo_feature": False,
                "use_weekday_feature": False,
                "use_time_hist_feature": False,
                "use_time_fut_feature": False,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "dual",
                "use_dynamic_encoder": False,
                "factorization_mode": "factorized",
                "temporal_model": "simple",
                "use_reverse_edges": True,
            },
        },
        "S3_dynamic": {
            "description": "S2 + dynamic OD graph encoder.",
            "flags": {
                "use_geo_feature": False,
                "use_weekday_feature": False,
                "use_time_hist_feature": False,
                "use_time_fut_feature": False,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "dual",
                "use_dynamic_encoder": True,
                "factorization_mode": "factorized",
                "temporal_model": "simple",
                "use_reverse_edges": True,
            },
        },
        "S4_transformer": {
            "description": "S3 + Transformer decoder.",
            "flags": {
                "use_geo_feature": False,
                "use_weekday_feature": False,
                "use_time_hist_feature": False,
                "use_time_fut_feature": False,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "dual",
                "use_dynamic_encoder": True,
                "factorization_mode": "factorized",
                "temporal_model": "transformer",
                "use_reverse_edges": True,
            },
        },
        "S5_time": {
            "description": "S4 + historical/future time encoding.",
            "flags": {
                "use_geo_feature": False,
                "use_weekday_feature": False,
                "use_time_hist_feature": True,
                "use_time_fut_feature": True,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "dual",
                "use_dynamic_encoder": True,
                "factorization_mode": "factorized",
                "temporal_model": "transformer",
                "use_reverse_edges": True,
            },
        },
        "S6_weekday": {
            "description": "S5 + weekday embedding.",
            "flags": {
                "use_geo_feature": False,
                "use_weekday_feature": True,
                "use_time_hist_feature": True,
                "use_time_fut_feature": True,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "dual",
                "use_dynamic_encoder": True,
                "factorization_mode": "factorized",
                "temporal_model": "transformer",
                "use_reverse_edges": True,
            },
        },
        "S7_geo": {
            "description": "S6 + station geographic coordinates.",
            "flags": {
                "use_geo_feature": True,
                "use_weekday_feature": True,
                "use_time_hist_feature": True,
                "use_time_fut_feature": True,
                "use_gate_head": False,
                "encoder_type": "gatv2",
                "static_branch_mode": "dual",
                "use_dynamic_encoder": True,
                "factorization_mode": "factorized",
                "temporal_model": "transformer",
                "use_reverse_edges": True,
            },
        },
        "S8_gate": {
            "description": "S7 + sparsity-aware gating (full Metro-GATF).",
            "flags": {
                "use_geo_feature": True,
                "use_weekday_feature": True,
                "use_time_hist_feature": True,
                "use_time_fut_feature": True,
                "use_gate_head": True,
                "encoder_type": "gatv2",
                "static_branch_mode": "dual",
                "use_dynamic_encoder": True,
                "factorization_mode": "factorized",
                "temporal_model": "transformer",
                "use_reverse_edges": True,
            },
        },
    }
)

PROGRESSIVE_STAGE_ALIASES = {
    "S0": "S0_minimal",
    "S1": "S1_factorization",
    "S2": "S2_multiscale_static",
    "S3": "S3_dynamic",
    "S4": "S4_transformer",
    "S5": "S5_time",
    "S6": "S6_weekday",
    "S7": "S7_geo",
    "S8": "S8_gate",
}


def resolve_accelerator():
    return "cuda" if torch.cuda.is_available() else "cpu"


def close_active_wandb_run():
    try:
        import wandb
    except Exception:
        return

    if wandb.run is not None:
        wandb.finish()


def resolve_path(path, root):
    if path in (None, ""):
        return None
    if os.path.isabs(path):
        return path
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(root, path)


def parse_csv_list(raw, cast_fn=str):
    items = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(cast_fn(token))
    return items


def get_experiment_registry(experiment_plan):
    if experiment_plan == "ablation":
        return ABLATION_REGISTRY
    if experiment_plan == "progressive_core":
        return PROGRESSIVE_STAGE_REGISTRY
    raise ValueError(f"Unsupported experiment_plan: {experiment_plan}")


def normalize_experiment_name(experiment_plan, experiment_name):
    if experiment_plan == "progressive_core":
        return PROGRESSIVE_STAGE_ALIASES.get(experiment_name, experiment_name)
    return experiment_name


def build_experiment_flags(experiment_plan, experiment_name):
    if experiment_name == "base":
        return dict(BASE_ABLATION_FLAGS)

    registry = get_experiment_registry(experiment_plan)
    if experiment_name not in registry:
        raise ValueError(f"Unknown experiment name: {experiment_name}")

    if experiment_plan == "ablation":
        flags = dict(BASE_ABLATION_FLAGS)
        flags.update({k: v for k, v in registry[experiment_name].items() if k in flags})
        return flags

    return dict(registry[experiment_name]["flags"])


def describe_experiment(experiment_plan, experiment_name):
    if experiment_name == "base":
        return "Base setting."
    registry = get_experiment_registry(experiment_plan)
    return registry[experiment_name]["description"]


def graph_week_collate_fn_ablation(batch, static_edge_index, add_reverse_edges=True):
    data_list = []
    B = len(batch)
    T = batch[0]["x_tensor"].shape[0]
    N = batch[0]["x_tensor"].shape[1]

    time_hist_list = []
    time_fut_list = []
    weekday_list = []

    for b in range(B):
        x_seq = batch[b]["x_tensor"]
        time_hist_list.append(batch[b]["time_enc_hist"])
        time_fut_list.append(batch[b]["time_enc_fut"])
        weekday_list.append(batch[b]["weekday_tensor"])

        for t in range(T):
            od_t = x_seq[t]
            edge_idx, edge_attr = dense_to_sparse(od_t)
            edge_attr = edge_attr.unsqueeze(-1)

            if add_reverse_edges:
                rev_edge_idx = edge_idx.flip(0)
                edge_idx = torch.cat([edge_idx, rev_edge_idx], dim=1)
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

            x_node = torch.zeros(N, 1)
            data_list.append(Data(x=x_node, edge_index=edge_idx, edge_attr=edge_attr))

    batch_graph = Batch.from_data_list(data_list)
    labels = torch.stack([b["y_tensor"] for b in batch])
    time_enc_hist = torch.stack(time_hist_list)
    time_enc_fut = torch.stack(time_fut_list)
    weekday = torch.stack(weekday_list)
    return static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday


def _edge_weight_from_attr(edge_attr):
    if edge_attr is None:
        return None
    if edge_attr.dim() == 1:
        return edge_attr
    if edge_attr.shape[-1] == 1:
        return edge_attr.view(-1)
    return edge_attr.mean(dim=-1)


class GraphConvBlock(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, conv_type="gatv2", heads=4, edge_attr_dim=None):
        super().__init__()
        self.conv_type = conv_type

        if conv_type == "gatv2":
            self.conv1 = GATv2Conv(
                in_channels,
                hid_channels,
                heads=heads,
                dropout=0.2,
                edge_dim=edge_attr_dim,
            )
            self.conv2 = GATv2Conv(
                hid_channels * heads,
                hid_channels,
                heads=1,
                concat=False,
                edge_dim=edge_attr_dim,
            )
        elif conv_type == "gat":
            self.conv1 = GATConv(
                in_channels,
                hid_channels,
                heads=heads,
                dropout=0.2,
                edge_dim=edge_attr_dim,
            )
            self.conv2 = GATConv(
                hid_channels * heads,
                hid_channels,
                heads=1,
                concat=False,
                edge_dim=edge_attr_dim,
            )
        elif conv_type == "gcn":
            self.conv1 = GCNConv(in_channels, hid_channels)
            self.conv2 = GCNConv(hid_channels, hid_channels)
        else:
            raise ValueError(f"Unsupported encoder_type: {conv_type}")

    def forward(self, x, edge_index, edge_attr=None):
        if self.conv_type == "gcn":
            edge_weight = _edge_weight_from_attr(edge_attr)
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.elu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            return x

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class FlexibleStaticEncoder(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, layers=1, heads=4, conv_type="gatv2"):
        super().__init__()
        self.block1 = GraphConvBlock(
            in_channels,
            hid_channels,
            conv_type=conv_type,
            heads=heads,
            edge_attr_dim=None,
        )
        self.blocks = torch.nn.ModuleList(
            [
                GraphConvBlock(
                    hid_channels,
                    hid_channels,
                    conv_type=conv_type,
                    heads=heads,
                    edge_attr_dim=None,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, edge_index):
        h = self.block1(x, edge_index)
        h = F.elu(h)
        for block in self.blocks:
            h = block(h, edge_index)
            h = F.elu(h)
        return h


class FlexibleDynamicEncoder(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, layers=2, heads=4, conv_type="gatv2", edge_attr_dim=1):
        super().__init__()
        self.block1 = GraphConvBlock(
            in_channels,
            hid_channels,
            conv_type=conv_type,
            heads=heads,
            edge_attr_dim=edge_attr_dim,
        )
        self.blocks = torch.nn.ModuleList(
            [
                GraphConvBlock(
                    hid_channels,
                    hid_channels,
                    conv_type=conv_type,
                    heads=heads,
                    edge_attr_dim=edge_attr_dim,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.block1(x, edge_index, edge_attr)
        h = F.elu(h)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
            h = F.elu(h)
        return h


class SimpleTemporalDecoder(torch.nn.Module):
    def __init__(self, hid_dim, num_layers):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=hid_dim,
            hidden_size=hid_dim,
            num_layers=max(1, num_layers),
            batch_first=True,
        )
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, tgt, memory, tgt_mask=None):
        del tgt_mask
        memory_bf = memory.permute(1, 0, 2)
        _, hidden = self.gru(memory_bf)
        context = hidden[-1]
        out = context.unsqueeze(0).expand(tgt.size(0), -1, -1) + tgt
        return self.out_proj(out)


class AblationGATTransformerODWeek(GATTransformerODWeek):
    def __init__(
        self,
        num_nodes,
        node_feat_dim,
        gat_hid_dim=32,
        heads=4,
        decode_num_layers=2,
        num_future_steps=6,
        weekday_emb_dim=8,
        time_enc_dim=2,
        node_latlon=None,
        use_geo_feature=True,
        use_weekday_feature=True,
        use_time_hist_feature=True,
        use_time_fut_feature=True,
        use_gate_head=True,
        encoder_type="gatv2",
        static_branch_mode="dual",
        use_dynamic_encoder=True,
        factorization_mode="factorized",
        temporal_model="transformer",
    ):
        super().__init__(
            num_nodes=num_nodes,
            node_feat_dim=node_feat_dim,
            gat_hid_dim=gat_hid_dim,
            heads=heads,
            decode_num_layers=decode_num_layers,
            num_future_steps=num_future_steps,
            weekday_emb_dim=weekday_emb_dim,
            time_enc_dim=time_enc_dim,
            node_latlon=node_latlon,
        )
        self.use_geo_feature = use_geo_feature
        self.use_weekday_feature = use_weekday_feature
        self.use_time_hist_feature = use_time_hist_feature
        self.use_time_fut_feature = use_time_fut_feature
        self.use_gate_head = use_gate_head
        self.encoder_type = encoder_type
        self.static_branch_mode = static_branch_mode
        self.use_dynamic_encoder = use_dynamic_encoder
        self.factorization_mode = factorization_mode
        self.temporal_model = temporal_model

        in_dim = node_feat_dim + weekday_emb_dim + self.geo_dim

        if self.encoder_type != "gatv2":
            self.short_spatial_encoder = FlexibleStaticEncoder(
                in_dim,
                gat_hid_dim,
                layers=1,
                heads=heads,
                conv_type=self.encoder_type,
            )
            self.long_spatial_encoder = FlexibleStaticEncoder(
                in_dim,
                gat_hid_dim,
                layers=3,
                heads=heads,
                conv_type=self.encoder_type,
            )
            self.dynamic_encoder = FlexibleDynamicEncoder(
                gat_hid_dim,
                gat_hid_dim,
                heads=heads,
                conv_type=self.encoder_type,
                edge_attr_dim=1,
            )

        if self.temporal_model == "simple":
            self.decoder = SimpleTemporalDecoder(gat_hid_dim, decode_num_layers)
        elif self.temporal_model != "transformer":
            raise ValueError(f"Unsupported temporal_model: {self.temporal_model}")

        if self.factorization_mode == "rowwise":
            self.mag_row_proj = torch.nn.Linear(gat_hid_dim, num_nodes)
            if self.use_gate_head:
                self.gate_row_proj = torch.nn.Linear(gat_hid_dim, num_nodes)
        elif self.factorization_mode != "factorized":
            raise ValueError(f"Unsupported factorization_mode: {self.factorization_mode}")

        if self.static_branch_mode not in {"dual", "short_only", "long_only"}:
            raise ValueError(f"Unsupported static_branch_mode: {self.static_branch_mode}")

    def forward(self, adjency_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday_tensor):
        N = self.num_nodes

        base = self.node_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        weekday = self.weekday_embed(weekday_tensor).unsqueeze(1).repeat(1, N, 1)
        if not self.use_weekday_feature:
            weekday = torch.zeros_like(weekday)

        feats = [base]
        if self.use_geo:
            geo = self.node_latlon.unsqueeze(0).repeat(B, 1, 1)
            if not self.use_geo_feature:
                geo = torch.zeros_like(geo)
            feats.append(geo)
        feats.append(weekday)

        static_feat = torch.cat(feats, dim=-1)
        flat = static_feat.view(B * N, -1)

        edge_index_batched = self._repeat_edge_index(adjency_edge_index, B, N)
        short = self.short_spatial_encoder(flat, edge_index_batched)
        long = self.long_spatial_encoder(flat, edge_index_batched)

        if self.static_branch_mode == "dual":
            spatial = torch.cat([short, long], dim=-1)
            spatial = self.spatial_fuse_proj(spatial)
        elif self.static_branch_mode == "short_only":
            spatial = short
        else:
            spatial = long
        spatial = spatial.view(B, N, -1)

        if self.use_dynamic_encoder:
            dyn_in = spatial.unsqueeze(1).repeat(1, T, 1, 1).view(B * T * N, -1)
            batch_graph.x = dyn_in
            dyn = self.dynamic_encoder(
                batch_graph.x,
                batch_graph.edge_index,
                batch_graph.edge_attr,
            )
            dyn = dyn.view(B, T, N, -1)
        else:
            dyn = spatial.new_zeros(B, T, N, spatial.shape[-1])

        if self.use_time_hist_feature:
            time_enc = self.time_enc_linear(time_enc_hist).unsqueeze(2).repeat(1, 1, N, 1)
        else:
            time_enc = spatial.new_zeros(B, T, N, spatial.shape[-1])
        spatial_exp = spatial.unsqueeze(1).expand(-1, T, -1, -1)

        fused = torch.cat([dyn, spatial_exp, time_enc], dim=-1)
        fused = self.fuse_proj(fused)

        memory = fused.permute(1, 0, 2, 3).reshape(T, B * N, -1)
        memory = self.pos_enc(memory)

        O = self.future_steps
        step_ids = torch.arange(O, device=memory.device)
        tgt = self.future_step_emb(step_ids).unsqueeze(1).expand(O, B * N, -1)
        tgt = self.pos_enc(tgt)

        if time_enc_fut is not None and self.use_time_fut_feature:
            tgt_time = self.time_enc_linear(time_enc_fut)
            tgt_time = tgt_time.unsqueeze(2).expand(-1, -1, N, -1)
            tgt_time = tgt_time.permute(1, 0, 2, 3).reshape(O, B * N, -1)
            tgt = tgt + tgt_time

        out = self.decoder(tgt, memory, tgt_mask=None)
        out = out.view(O, B, N, -1)

        if self.factorization_mode == "factorized":
            H_O = self.origin_proj(out)
            H_D = self.dest_proj(out)
            mag_logits = torch.einsum("obid,objd->obij", H_O, H_D)
            mag_log = F.softplus(mag_logits)
        else:
            mag_logits = self.mag_row_proj(out)
            mag_log = F.softplus(mag_logits)

        gate_logits = None
        if self.use_gate_head:
            if self.factorization_mode == "factorized":
                G_O = self.gate_origin_proj(out)
                G_D = self.gate_dest_proj(out)
                gate_logits = torch.einsum("obid,objd->obij", G_O, G_D) + self.gate_bias
            else:
                gate_logits = self.gate_row_proj(out) + self.gate_bias
            gate_logits = gate_logits.permute(1, 0, 2, 3).contiguous()

        mag_log = mag_log.permute(1, 0, 2, 3).contiguous()
        return mag_log, gate_logits


class AblationMetroGraphWeekLM(L.LightningModule):
    def __init__(
        self,
        model,
        loss=torch.nn.SmoothL1Loss(),
        lr=1e-3,
        mape_eps=1e-3,
        lambda_gate=1.0,
        gate_tau=0.9,
        pos_weight_clip=10.0,
        target_s=None,
        target_e=None,
        use_gate=True,
        dense_zero_lambda=0.05,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.lr = lr
        self.mape_eps = mape_eps
        self.lambda_gate = lambda_gate
        self.gate_tau = gate_tau
        self.pos_weight_clip = pos_weight_clip
        self.target_s = target_s
        self.target_e = target_e
        self.use_gate = use_gate
        self.dense_zero_lambda = dense_zero_lambda

    def forward(self, static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday):
        return self.model(static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday)

    def _compute_metrics(self, y_true, y_pred):
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(torch.clamp(y_pred, min=0.0))
        diff = y_true - y_pred

        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))
        smape_all = smape(y_true, y_pred, eps=self.mape_eps)
        return mse, mae, smape_all, rmse

    def _loss(self, y_true_log, mag_log, gate_logits):
        if not self.use_gate:
            loss = balanced_huber(mag_log, y_true_log, zero_lambda=self.dense_zero_lambda)
            zero = torch.tensor(0.0, device=y_true_log.device)
            return loss, loss, zero, zero

        z = (y_true_log > 0).float()
        num_pos = z.sum()
        num_tot = z.numel()
        num_neg = num_tot - num_pos
        pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=self.pos_weight_clip)

        bce = F.binary_cross_entropy_with_logits(
            gate_logits,
            z,
            pos_weight=pos_weight,
        )

        pos_mask = z.bool()
        if pos_mask.any():
            mag_loss = self.loss_fn(mag_log[pos_mask], y_true_log[pos_mask])
        else:
            mag_loss = torch.tensor(0.0, device=y_true_log.device)

        total = mag_loss + self.lambda_gate * bce
        return total, mag_loss, bce, pos_weight.detach()

    def _apply_gate(self, mag_log, gate_logits):
        if not self.use_gate or gate_logits is None:
            return mag_log, None

        gate_prob = torch.sigmoid(gate_logits)
        mag_log_hard = torch.where(gate_prob > self.gate_tau, mag_log, torch.zeros_like(mag_log))
        return mag_log_hard, gate_prob

    def _log_local_pair_metrics(self, labels, mag_log_eval, gate_prob):
        if self.target_s is None or self.target_e is None:
            return

        local_true = labels[:, :, self.target_s, self.target_e]
        local_pred = mag_log_eval[:, :, self.target_s, self.target_e]
        local_mse, local_mae, local_smape, local_rmse = self._compute_metrics(local_true, local_pred)

        self.log("val/local_mse", local_mse, prog_bar=False)
        self.log("val/local_rmse", local_rmse, prog_bar=False)
        self.log("val/local_mae", local_mae, prog_bar=False)
        self.log("val/local_smape", local_smape, prog_bar=False)

        if gate_prob is not None:
            local_gate = gate_prob[:, :, self.target_s, self.target_e]
            local_true_rate = (local_true > 0).float().mean()
            local_pred_rate = (local_gate > self.gate_tau).float().mean()
            self.log("val/local_true_nonzero_rate", local_true_rate, prog_bar=False)
            self.log("val/local_pred_nonzero_rate", local_pred_rate, prog_bar=False)

    def training_step(self, batch, batch_idx):
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday = batch
        mag_log, gate_logits = self.model(
            static_edge_index,
            batch_graph,
            B,
            T,
            time_enc_hist,
            time_enc_fut,
            weekday,
        )
        loss, mag_loss, bce, pos_w = self._loss(labels, mag_log, gate_logits)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mag_loss", mag_loss, prog_bar=False)
        if self.use_gate:
            self.log("train/gate_bce", bce, prog_bar=False)
            self.log("train/pos_weight", pos_w, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday = batch
        mag_log, gate_logits = self.model(
            static_edge_index,
            batch_graph,
            B,
            T,
            time_enc_hist,
            time_enc_fut,
            weekday,
        )
        loss, mag_loss, bce, pos_w = self._loss(labels, mag_log, gate_logits)
        mag_log_eval, gate_prob = self._apply_gate(mag_log, gate_logits)

        mse, mae, smape_all, rmse = self._compute_metrics(labels, mag_log_eval)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        self.log("val/smape", smape_all, prog_bar=True)
        self.log("val/mse", mse, prog_bar=False)
        self.log("val/mag_loss", mag_loss, prog_bar=False)

        if self.use_gate:
            true_rate = (labels > 0).float().mean()
            pred_rate = (gate_prob > self.gate_tau).float().mean()
            self.log("val/gate_bce", bce, prog_bar=False)
            self.log("val/pos_weight", pos_w, prog_bar=False)
            self.log("val/true_nonzero_rate", true_rate, prog_bar=False)
            self.log("val/pred_nonzero_rate", pred_rate, prog_bar=False)

        self._log_local_pair_metrics(labels, mag_log_eval, gate_prob)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def build_static_edge_index(od_csv):
    od_df = pd.read_csv(od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    static_edge_index, _ = dense_to_sparse(adj)
    return od_df, static_edge_index


def load_node_latlon(args, root, od_df):
    if args.station_latlon_csv is None:
        return None

    station_latlon_csv = resolve_path(args.station_latlon_csv, root)
    if not os.path.exists(station_latlon_csv):
        raise FileNotFoundError(f"station_latlon_csv not found: {station_latlon_csv}")

    station_to_idx_path = resolve_path("station_to_idx.json", root)
    if not os.path.exists(station_to_idx_path):
        raise FileNotFoundError(f"station_to_idx.json not found: {station_to_idx_path}")

    latlon_raw = pd.read_csv(station_latlon_csv)
    with open(station_to_idx_path, "r", encoding="utf-8") as fp:
        station_to_idx = json.load(fp)

    full_nodes = pd.DataFrame(
        {
            "station_ad": list(station_to_idx.keys()),
            "node_id": list(station_to_idx.values()),
        }
    )
    latlon_df = full_nodes.merge(latlon_raw, on="station_ad", how="left")
    adj_np = od_df.values

    missing = latlon_df[latlon_df["lat"].isna() | latlon_df["lon"].isna()]
    if len(missing) > 0:
        print(f"Filling {len(missing)} missing station(s) by neighbor mean")
        for _, row in missing.iterrows():
            node_id = int(row["node_id"])
            neighbors = (adj_np[node_id] > 0).nonzero()[0]
            if len(neighbors) == 0:
                raise ValueError(f"No neighbors available to fill station {node_id}")

            neigh_latlon = latlon_df.iloc[neighbors][["lat", "lon"]].values
            latlon_df.loc[latlon_df["node_id"] == node_id, "lat"] = neigh_latlon[:, 0].mean()
            latlon_df.loc[latlon_df["node_id"] == node_id, "lon"] = neigh_latlon[:, 1].mean()

    latlon_df = latlon_df.sort_values("node_id")
    latlon = torch.tensor(latlon_df[["lat", "lon"]].values, dtype=torch.float32)
    latlon_min = latlon.min(dim=0, keepdim=True)[0]
    latlon_max = latlon.max(dim=0, keepdim=True)[0]
    return (latlon - latlon_min) / (latlon_max - latlon_min + 1e-6)


def build_datasets(args):
    trainset, valset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        time_resolution=args.time_resolution,
        cache_in_mem=True if args.cache_dataset else False,
    )
    return trainset, valset


def build_dataloaders(args, trainset, valset, static_edge_index, batch_size, add_reverse_edges=True):
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=False,
        collate_fn=lambda batch: graph_week_collate_fn_ablation(
            batch,
            static_edge_index,
            add_reverse_edges=add_reverse_edges,
        ),
        pin_memory=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: graph_week_collate_fn_ablation(
            batch,
            static_edge_index,
            add_reverse_edges=add_reverse_edges,
        ),
        pin_memory=True,
    )
    return train_loader, val_loader


def resolve_effective_batch_size(args, experiment_name):
    if not args.progressive_batch_schedule or args.experiment_plan != "progressive_core":
        return args.batch_size

    stage_names = list(PROGRESSIVE_STAGE_REGISTRY.keys())
    normalized_name = normalize_experiment_name(args.experiment_plan, experiment_name)
    if normalized_name not in stage_names:
        return args.batch_size

    stage_idx = stage_names.index(normalized_name)
    scaled = int(args.progressive_batch_start * (args.progressive_batch_decay ** stage_idx))
    return max(args.progressive_batch_min, scaled)


def experiments_require_geo(experiment_plan, experiment_names):
    return any(build_experiment_flags(experiment_plan, name)["use_geo_feature"] for name in experiment_names)


def build_model(args, num_nodes, node_latlon, experiment_plan, experiment_name):
    flags = build_experiment_flags(experiment_plan, experiment_name)
    model = AblationGATTransformerODWeek(
        num_nodes=num_nodes,
        heads=args.gat_heads,
        node_feat_dim=args.node_feat_dim,
        gat_hid_dim=args.gat_hidden,
        num_future_steps=args.pred_size,
        decode_num_layers=args.decode_num_layers,
        node_latlon=node_latlon,
        use_geo_feature=flags["use_geo_feature"],
        use_weekday_feature=flags["use_weekday_feature"],
        use_time_hist_feature=flags["use_time_hist_feature"],
        use_time_fut_feature=flags["use_time_fut_feature"],
        use_gate_head=flags["use_gate_head"],
        encoder_type=flags["encoder_type"],
        static_branch_mode=flags["static_branch_mode"],
        use_dynamic_encoder=flags["use_dynamic_encoder"],
        factorization_mode=flags["factorization_mode"],
        temporal_model=flags["temporal_model"],
    )
    return model, flags


def build_lightning_module(args, model, experiment_plan, experiment_name):
    flags = build_experiment_flags(experiment_plan, experiment_name)
    return AblationMetroGraphWeekLM(
        model=model,
        loss=torch.nn.SmoothL1Loss(),
        lr=args.lr,
        lambda_gate=args.lambda_gate,
        gate_tau=args.gate_tau,
        pos_weight_clip=args.pos_weight_clip,
        target_s=args.target_s,
        target_e=args.target_e,
        use_gate=flags["use_gate_head"],
    )


def evaluate_full_network(lightning_module, val_loader, device):
    lightning_module = lightning_module.to(device)
    lightning_module.eval()

    sq_error_sum = 0.0
    abs_error_sum = 0.0
    smape_term_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday = batch
            static_edge_index = static_edge_index.to(device)
            batch_graph = batch_graph.to(device)
            labels = labels.to(device)
            time_enc_hist = time_enc_hist.to(device)
            time_enc_fut = time_enc_fut.to(device)
            weekday = weekday.to(device)

            mag_log, gate_logits = lightning_module.model(
                static_edge_index,
                batch_graph,
                B,
                T,
                time_enc_hist,
                time_enc_fut,
                weekday,
            )
            mag_log_eval, _ = lightning_module._apply_gate(mag_log, gate_logits)

            pred = torch.expm1(torch.clamp(mag_log_eval, min=0.0))
            true = torch.expm1(labels)
            diff = true - pred
            abs_diff = torch.abs(diff)
            denom = (torch.abs(true) + torch.abs(pred)).clamp(min=lightning_module.mape_eps)

            sq_error_sum += torch.sum(diff ** 2).item()
            abs_error_sum += torch.sum(abs_diff).item()
            smape_term_sum += torch.sum(2.0 * abs_diff / denom).item()
            count += diff.numel()

    if count == 0:
        raise ValueError("Validation loader produced zero elements during full-network evaluation.")

    mse = sq_error_sum / count
    return {
        "smape": 100.0 * (smape_term_sum / count),
        "rmse": mse ** 0.5,
        "mae": abs_error_sum / count,
        "mse": mse,
    }


def create_loggers(args, study_name, variant_name, seed, run_dir):
    csv_logger = CSVLogger(
        save_dir=os.path.join(run_dir, "logs"),
        name="csv",
        version="",
    )
    loggers = [csv_logger]
    if not args.disable_wandb:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                name=f"{study_name}_{variant_name}_seed{seed}",
                group=study_name,
                save_dir=run_dir,
                config={**vars(args), "variant": variant_name, "seed": seed},
            )
        )
    return loggers


def aggregate_results(results_df):
    agg = (
        results_df.groupby("variant", as_index=False)
        .agg(
            runs=("seed", "count"),
            smape_mean=("smape", "mean"),
            smape_std=("smape", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .sort_values(["smape_mean", "rmse_mean", "mae_mean"], ascending=[True, True, True])
    )
    return agg


def write_report(summary_df, per_run_df, study_dir):
    report_path = os.path.join(study_dir, "summary", "report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = [
        "# Ablation Study Report",
        "",
        "## Ranked Summary",
        "",
        "| Variant | Runs | SMAPE mean | SMAPE std | RMSE mean | RMSE std | MAE mean | MAE std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['runs'])} | "
            f"{row['smape_mean']:.6f} | {0.0 if pd.isna(row['smape_std']) else row['smape_std']:.6f} | "
            f"{row['rmse_mean']:.6f} | {0.0 if pd.isna(row['rmse_std']) else row['rmse_std']:.6f} | "
            f"{row['mae_mean']:.6f} | {0.0 if pd.isna(row['mae_std']) else row['mae_std']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Per-run Results",
            "",
            "| Variant | Seed | SMAPE | RMSE | MAE | Checkpoint |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in per_run_df.sort_values(["variant", "seed"]).iterrows():
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['smape']:.6f} | "
            f"{row['rmse']:.6f} | {row['mae']:.6f} | {row['best_checkpoint']} |"
        )

    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--cache_dataset", action="store_true")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")
    parser.add_argument("--station_latlon_csv", type=str, default="./ad_station_latlon.csv")
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=6)
    parser.add_argument("--node_feat_dim", type=int, default=16)
    parser.add_argument("--decode_num_layers", type=int, default=2)
    parser.add_argument("--target_s", type=int, default=None)
    parser.add_argument("--target_e", type=int, default=None)
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--lambda_gate", type=float, default=1.0)
    parser.add_argument("--gate_tau", type=float, default=0.9)
    parser.add_argument("--pos_weight_clip", type=float, default=10.0)
    parser.add_argument("--wandb_project", default="transport-abligation")
    parser.add_argument(
        "--experiment_plan",
        choices=["ablation", "progressive_core"],
        default="ablation",
        help="Choose between leave-one-out ablations and progressive core-module build-up stages.",
    )
    parser.add_argument(
        "--progressive_batch_schedule",
        action="store_true",
        help="When using progressive_core, override batch size by stage using a decaying schedule.",
    )
    parser.add_argument("--progressive_batch_start", type=int, default=64)
    parser.add_argument("--progressive_batch_decay", type=float, default=0.5)
    parser.add_argument("--progressive_batch_min", type=int, default=2)

    parser.add_argument(
        "--variants",
        default=(
            "no_geo,no_weekday,no_time_enc,no_future_time_enc,no_gate,meta_off,"
            "gat_encoder,gcn_encoder,short_only,long_only,no_dynamic,"
            "no_factorization,no_reverse_edge,simple_temporal"
        ),
        help="Comma-separated ablation variants to run.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(PROGRESSIVE_STAGE_REGISTRY.keys()),
        help="Comma-separated progressive stage names to run when --experiment_plan progressive_core is selected.",
    )
    parser.add_argument("--include_base", action="store_true", help="Also run the unmodified base setting.")
    parser.add_argument("--seeds", default="0", help="Comma-separated random seeds.")
    parser.add_argument("--study_name", default="")
    parser.add_argument("--output_root", default="./ablation_runs")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--resume_if_complete", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    args.od_csv = resolve_path(args.od_csv, root)
    args.station_latlon_csv = resolve_path(args.station_latlon_csv, root)
    args.output_root = resolve_path(args.output_root, root)

    if args.experiment_plan == "ablation":
        experiment_names = list(OrderedDict.fromkeys(parse_csv_list(args.variants)))
        if args.include_base and "base" not in experiment_names:
            experiment_names = ["base"] + experiment_names
    else:
        experiment_names = [
            normalize_experiment_name(args.experiment_plan, name)
            for name in parse_csv_list(args.stages)
        ]
        experiment_names = list(OrderedDict.fromkeys(experiment_names))

    registry = get_experiment_registry(args.experiment_plan)
    invalid = [name for name in experiment_names if name != "base" and name not in registry]
    if invalid:
        raise ValueError(f"Unknown experiment names for plan '{args.experiment_plan}': {invalid}")

    seeds = parse_csv_list(args.seeds, int)
    if not seeds:
        raise ValueError("At least one seed must be provided.")

    study_name = args.study_name or datetime.datetime.now().strftime("ablation_%Y%m%d_%H%M%S")
    study_dir = os.path.join(args.output_root, study_name)
    os.makedirs(study_dir, exist_ok=True)

    od_df, static_edge_index = build_static_edge_index(args.od_csv)
    node_latlon = load_node_latlon(args, root, od_df) if experiments_require_geo(args.experiment_plan, experiment_names) else None
    trainset, valset = build_datasets(args)
    num_nodes = static_edge_index.max().item() + 1

    accelerator = resolve_accelerator()
    devices = 1
    strategy = "auto"
    if args.use_ddp:
        strategy = "ddp_find_unused_parameters_true"
        devices = torch.cuda.device_count() if accelerator == "cuda" else os.cpu_count()

    all_results = []
    run_manifest = {
        "study_name": study_name,
        "experiment_plan": args.experiment_plan,
        "experiments": experiment_names,
        "seeds": seeds,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
    }
    with open(os.path.join(study_dir, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(run_manifest, fp, indent=2)

    for experiment_name in experiment_names:
        for seed in seeds:
            run_dir = os.path.join(study_dir, experiment_name, f"seed_{seed}")
            result_path = os.path.join(run_dir, "run_result.json")
            if args.resume_if_complete and os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as fp:
                    all_results.append(json.load(fp))
                print(f"[Skip] {experiment_name} seed={seed} already completed.")
                continue

            os.makedirs(run_dir, exist_ok=True)
            L.seed_everything(seed, workers=True)

            model, flags = build_model(args, num_nodes, node_latlon, args.experiment_plan, experiment_name)
            model.static_edge_index = static_edge_index
            lm = build_lightning_module(args, model, args.experiment_plan, experiment_name)
            effective_batch_size = resolve_effective_batch_size(args, experiment_name)
            train_loader, val_loader = build_dataloaders(
                args,
                trainset,
                valset,
                static_edge_index,
                batch_size=effective_batch_size,
                add_reverse_edges=flags["use_reverse_edges"],
            )

            run_config = {
                "experiment_name": experiment_name,
                "experiment_plan": args.experiment_plan,
                "seed": seed,
                "ablation_flags": flags,
                "effective_batch_size": effective_batch_size,
                "description": describe_experiment(args.experiment_plan, experiment_name),
                "train_args": vars(args),
            }
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fp:
                json.dump(run_config, fp, indent=2)

            close_active_wandb_run()
            loggers = create_loggers(args, study_name, experiment_name, seed, run_dir)
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.config.update({"effective_batch_size": effective_batch_size}, allow_val_change=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(run_dir, "checkpoints"),
                filename="best-{epoch:03d}",
                monitor="val/smape",
                mode="min",
                save_top_k=1,
                save_last=True,
                auto_insert_metric_name=False,
            )

            trainer = L.Trainer(
                default_root_dir=run_dir,
                max_epochs=args.max_epochs,
                accelerator=accelerator,
                devices=devices,
                strategy=strategy,
                logger=loggers,
                callbacks=[checkpoint_callback],
                log_every_n_steps=50,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )

            print(f"[Run] experiment={experiment_name} seed={seed}")
            train_start = time.perf_counter()
            trainer.fit(lm, train_loader, val_loader)
            train_time_sec = time.perf_counter() - train_start

            best_path = checkpoint_callback.best_model_path
            if not best_path:
                raise RuntimeError(f"No best checkpoint found for {experiment_name} seed={seed}")

            best_model, _ = build_model(args, num_nodes, node_latlon, args.experiment_plan, experiment_name)
            best_lm = AblationMetroGraphWeekLM.load_from_checkpoint(
                best_path,
                map_location="cpu",
                model=best_model,
                loss=torch.nn.SmoothL1Loss(),
                lr=args.lr,
                lambda_gate=args.lambda_gate,
                gate_tau=args.gate_tau,
                pos_weight_clip=args.pos_weight_clip,
                target_s=args.target_s,
                target_e=args.target_e,
                use_gate=flags["use_gate_head"],
            )
            full_eval = evaluate_full_network(best_lm, val_loader, accelerator)

            result = {
                "variant": experiment_name,
                "experiment_name": experiment_name,
                "experiment_plan": args.experiment_plan,
                "seed": seed,
                "effective_batch_size": effective_batch_size,
                "smape": full_eval["smape"],
                "rmse": full_eval["rmse"],
                "mae": full_eval["mae"],
                "mse": full_eval["mse"],
                "best_checkpoint": best_path,
                "best_val_smape": float(checkpoint_callback.best_model_score.item()),
                "train_time_sec": train_time_sec,
                "train_time_hours": train_time_sec / 3600.0,
                "best_epoch": int(checkpoint_callback.best_model_path.split("best-")[-1].split(".ckpt")[0])
                if "best-" in checkpoint_callback.best_model_path
                else None,
            }
            with open(result_path, "w", encoding="utf-8") as fp:
                json.dump(result, fp, indent=2)
            all_results.append(result)

            close_active_wandb_run()
            del best_lm, lm, model, best_model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        raise RuntimeError("No ablation results were produced.")

    summary_dir = os.path.join(study_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    per_run_path = os.path.join(summary_dir, "per_run.csv")
    results_df.sort_values(["variant", "seed"]).to_csv(per_run_path, index=False)

    summary_df = aggregate_results(results_df)
    by_variant_path = os.path.join(summary_dir, "by_variant.csv")
    summary_df.to_csv(by_variant_path, index=False)

    if "base" in summary_df["variant"].values:
        base_row = summary_df[summary_df["variant"] == "base"].iloc[0]
        delta_df = summary_df.copy()
        delta_df["smape_delta_vs_base"] = delta_df["smape_mean"] - base_row["smape_mean"]
        delta_df["rmse_delta_vs_base"] = delta_df["rmse_mean"] - base_row["rmse_mean"]
        delta_df["mae_delta_vs_base"] = delta_df["mae_mean"] - base_row["mae_mean"]
        delta_df.to_csv(os.path.join(summary_dir, "delta_vs_base.csv"), index=False)

    write_report(summary_df, results_df, study_dir)

    print("\nAblation study complete.")
    print(f"Per-run results : {per_run_path}")
    print(f"Variant summary : {by_variant_path}")
    print(f"Report          : {os.path.join(summary_dir, 'report.md')}")


if __name__ == "__main__":
    main()
