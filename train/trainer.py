import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.optim import Adam
from SCIE_Benchmark.ODFormer import build_scaled_laplacian

def smape(y_true, y_pred, eps=1e-3):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=eps)
    return 100 * torch.mean(2 * torch.abs(y_pred - y_true) / denom)

class MetroLM(L.LightningModule):
    def __init__(self, model, loss, lr, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.mape_eps = mape_eps

    def forward_batch(self, batch):
        x_tensor = batch['x_tensor']
        weekday_tensor = batch['weekday_tensor']
        time_enc_hist = batch['time_enc_hist']
        time_enc_fut = batch['time_enc_fut']

        y_pred = self.model(
            x_tensor,
            weekday_tensor,
            time_enc_hist,
            time_enc_fut
        )
        y_tensor = batch['y_tensor']
        return y_pred, y_tensor, x_tensor.size(0)

    def _compute_metrics(self, y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)
        mae = torch.mean(torch.abs(y_true - y_pred))
        denom = torch.clamp(torch.abs(y_true), min=self.mape_eps)
        mape = torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0
        smape_ = smape(y_true, y_pred, eps=self.mape_eps)
        return mse, mae, mape, smape_

    def training_step(self, batch, batch_idx):
        y_pred, y_tensor, batch_size = self.forward_batch(batch)
        loss = self.loss(y_tensor, y_pred)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_tensor, batch_size = self.forward_batch(batch)
        mse, mae, mape, smape_ = self._compute_metrics(y_tensor, y_pred)
        self.log(
            "val_mse",
            mse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_mae",
            mae,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_mape",
            mape,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_smape",
            smape_,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return mse
    
    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr)
        return optim
   
class STLSTMLM(L.LightningModule):
    def __init__(self, model, loss, lr=1e-3, pred_size=30, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.pred_size = pred_size
        self.loss_fn = loss
        self.mape_eps = mape_eps
        
    def forward(self, x):
        return self.model(x)
    

    def _compute_metrics(self, y_true, y_pred):
        diff = y_true - y_pred

        # ----- core metrics (no masking) -----
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))

        # ----- MAPE (only where y_true > 0) -----
        mask = y_true > 0
        if mask.any():
            mape = torch.mean(
                torch.abs(diff[mask] / torch.clamp(y_true[mask], min=self.mape_eps))
            ) * 100.0
        else:
            mape = torch.tensor(0.0, device=y_true.device)

        # ----- sMAPE (standard definition) -----
        smape_ = smape(y_true, y_pred, eps=self.mape_eps)

        return mse, mae, mape, smape_, rmse
    
    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y'].squeeze(-1)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y'].squeeze(-1)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        metrics = self._compute_metrics(y, y_pred)
        if metrics is None:
            return  # <- 이게 정답. log도 안 하고 return도 안 함.
        mse, mae, mape, smape_, rmse = metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False)
        self.log("val/mse", mse, prog_bar=True, on_step=False)
        self.log("val/mae", mae, prog_bar=True, on_step=False)
        self.log("val/mape", mape, prog_bar=True, on_step=False)
        self.log("val/smape", smape_, prog_bar=True, on_step=False)
        self.log("val/rmse", rmse, prog_bar=True, on_step=False)
        
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class MetroGraphLM(L.LightningModule):
    def __init__(self, model, loss=torch.nn.MSELoss(), lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.lr = lr

    def forward(self, static_edge_index, batch_graph, B, T):
        return self.model(static_edge_index, batch_graph, B, T)

    def training_step(self, batch, batch_idx):
        # batch is a tuple from graph_collate_fn
        static_edge_index, batch_graph, B, T, labels = batch

        # forward
        preds = self.model(static_edge_index, batch_graph, B, T)
        loss = self.loss_fn(preds, labels)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        static_edge_index, batch_graph, B, T, labels = batch

        preds = self.model(static_edge_index, batch_graph, B, T)
        loss = self.loss_fn(preds, labels)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class MetroGraphWeekLM(L.LightningModule):
    def __init__(
        self,
        model,
        loss=torch.nn.SmoothL1Loss(),
        lr=1e-3,
        mape_eps=1e-3,
        lambda_gate=1.0,
        gate_tau=0.5,
        pos_weight_clip=50.0,
        target_s=None,
        target_e=None,
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

    def forward(self, static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday):
        return self.model(static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday)


    def _compute_metrics(self, y_true, y_pred):
        # back to original scale
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(torch.clamp(y_pred, min=0.0))

        diff = y_true - y_pred

        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))
        wmape = torch.sum(torch.abs(diff)) / torch.sum(torch.abs(y_true)) * 100.0
        smape_all = smape(y_true, y_pred, eps=self.mape_eps)

        return mse, mae, smape_all, rmse, wmape

    def _loss(self, y_true_log, mag_log, gate_logits):
        z = (y_true_log > 0).float()
        num_pos = z.sum()
        num_tot = z.numel()
        num_neg = num_tot - num_pos
        pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=self.pos_weight_clip)

        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits,
            z,
            pos_weight=pos_weight,
        )

        pos_mask = z.bool()
        if pos_mask.any():
            mag_loss = self.loss_fn(mag_log[pos_mask], y_true_log[pos_mask])
        else:
            mag_loss = torch.tensor(0.0, device=y_true_log.device)

        total = mag_loss + self.lambda_gate * gate_loss
        return total, mag_loss, gate_loss, pos_weight.detach()

    def _apply_gate(self, mag_log, gate_logits):
        gate_prob = torch.sigmoid(gate_logits)
        mag_log_hard = torch.where(
            gate_prob > self.gate_tau,
            mag_log,
            torch.zeros_like(mag_log),
        )
        return mag_log_hard, gate_prob

    def _log_local_pair_metrics(self, labels, mag_log_hard, gate_prob):
        if self.target_s is None or self.target_e is None:
            return

        local_true = labels[:, :, self.target_s, self.target_e]
        local_pred = mag_log_hard[:, :, self.target_s, self.target_e]
        local_gate = gate_prob[:, :, self.target_s, self.target_e]

        local_mse, local_mae, local_smape, local_rmse, local_wmape = self._compute_metrics(
            local_true,
            local_pred,
        )
        local_true_rate = (local_true > 0).float().mean()
        local_pred_rate = (local_gate > self.gate_tau).float().mean()

        self.log("val/local_mse", local_mse, prog_bar=False)
        self.log("val/local_rmse", local_rmse, prog_bar=True)
        self.log("val/local_mae", local_mae, prog_bar=False)
        self.log("val/local_smape", local_smape, prog_bar=False)
        self.log("val/local_wmape", local_wmape, prog_bar=False)
        self.log("val/local_true_nonzero_rate", local_true_rate, prog_bar=False)
        self.log("val/local_pred_nonzero_rate", local_pred_rate, prog_bar=False)

    def training_step(self, batch, batch_idx):
        # batch is a tuple from graph_collate_fn
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday = batch

        mag_log, gate_logits = self.model(static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday)
        loss, mag_loss, gate_loss, pos_w = self._loss(labels, mag_log, gate_logits)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mag_loss", mag_loss, prog_bar=False)
        self.log("train/gate_bce", gate_loss, prog_bar=False)
        self.log("train/pos_weight", pos_w, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday = batch

        mag_log, gate_logits = self.model(static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday)
        loss, mag_loss, gate_loss, pos_w = self._loss(labels, mag_log, gate_logits)
        mag_log_hard, gate_prob = self._apply_gate(mag_log, gate_logits)

        mse, mae, smape_all, rmse, wmape = self._compute_metrics(labels, mag_log_hard)
        true_rate = (labels > 0).float().mean()
        pred_rate = (gate_prob > self.gate_tau).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mag_loss", mag_loss, prog_bar=False)
        self.log("val/gate_bce", gate_loss, prog_bar=False)
        self.log("val/pos_weight", pos_w, prog_bar=False)
        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        self.log("val/smape", smape_all, prog_bar=True)
        self.log("val/wmape", wmape, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)
        self.log("val/true_nonzero_rate", true_rate, prog_bar=False)
        self.log("val/pred_nonzero_rate", pred_rate, prog_bar=False)
        self._log_local_pair_metrics(labels, mag_log_hard, gate_prob)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class STDAMHGNLitModule(L.LightningModule):
    def __init__(
        self,
        model,
        loss,
        lr=1e-3,
        mape_eps=1e-3
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss
        self.mape_eps = mape_eps

    def forward(self, tendency, periodicity):
        return self.model(tendency, periodicity)

    # -------------------------
    # Metrics (paper-style)
    # -------------------------

    def _compute_metrics(self, y_true, y_pred):
        # back to original scale
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(y_pred)

        diff = y_true - y_pred

        # ----- core metrics (no masking) -----
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))

        # ----- MAPE (only where y_true > 0) -----
        mask = y_true > 0
        if mask.any():
            mape = torch.mean(
                torch.abs(diff[mask] / torch.clamp(y_true[mask], min=self.mape_eps))
            ) * 100.0
        else:
            mape = torch.tensor(0.0, device=y_true.device)

        # ----- sMAPE (standard definition) -----
        smape_ = smape(y_true, y_pred, eps=self.mape_eps)

        return mse, mae, mape, smape_, rmse

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch, batch_idx):
        y_pred = self(
            batch["tendency"],
            batch["periodicity"]
        )
        y_true = batch["y"]
        mask = (y_true > 0).float()
        loss = ((y_pred - y_true) ** 2 * mask).sum() / mask.sum()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------
    # Validation
    # -------------------------
    def validation_step(self, batch, batch_idx):
        y_pred = self(
            batch["tendency"],
            batch["periodicity"]
        )
        y_true = batch["y"]
        mask = (y_true > 0).float()
        loss = ((y_pred - y_true) ** 2 * mask).sum() / mask.sum()
        mse, mae, mape, smape_, rmse = self._compute_metrics(
            y_true, y_pred
        )

        self.log("val_mse", mse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        self.log("val_smape", smape_, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    # -------------------------
    # Optimizer
    # -------------------------
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class ODformerLM(L.LightningModule):
    def __init__(self, model, adj_matrix, lr=1e-4, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.MSELoss()
        self.mape_eps = mape_eps
        L = build_scaled_laplacian(adj_matrix)
        self.register_buffer("L_origin", L)
        self.register_buffer("L_destination", L.clone())

    def forward(self, X):
        return self.model(X, self.L_origin, self.L_destination)

    # -------------------------
    # Metrics (paper-style)
    # -------------------------

    def _compute_metrics(self, y_true, y_pred):
        # back to original scale
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(y_pred)

        diff = y_true - y_pred

        # ----- core metrics (no masking) -----
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))

        # ----- MAPE (only where y_true > 0) -----
        mask = y_true > 0
        if not mask.any():
            mape = torch.nan
        else:
            mape = torch.mean(
                torch.abs(diff[mask] / torch.clamp(y_true[mask], min=self.mape_eps))
            ) * 100.0

        # ----- sMAPE (standard definition) -----
        smape_ = smape(y_true, y_pred, eps=self.mape_eps)

        return mse, mae, mape, smape_, rmse

    def training_step(self, batch, batch_idx):
        """
        batch:
            X: (B, T_in, N, N, F)
            Y: (B, T_out, N, N, F)
        """
        X = batch["X"]
        Y = batch["Y"]

        preds = self.model(X, self.L_origin, self.L_destination)
        loss = self.loss_fn(preds, Y)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch["X"]
        Y = batch["Y"]

        preds = self.model(X, self.L_origin, self.L_destination)
        loss = self.loss_fn(preds, Y)

        mse, mae, mape, smape_, rmse = self._compute_metrics(
            Y, preds
        )

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mape", mape, prog_bar=True)
        self.log("val/smape", smape_, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }

class MetroGCNLSTMLM(L.LightningModule):
    def __init__(self, model, lr=1e-3, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.mape_eps = mape_eps
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _compute_metrics(self, y_true, y_pred):
        # back to original scale
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(y_pred)

        diff = y_true - y_pred

        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))

        mask = y_true > 0
        if not mask.any():
            mape = torch.nan
        else:
            mape = torch.mean(
                torch.abs(diff[mask] / torch.clamp(y_true[mask], min=self.mape_eps))
            ) * 100.0

        smape_ = smape(y_true, y_pred, eps=self.mape_eps)

        return mse, mae, mape, smape_, rmse

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)

        mse, mae, mape, smape_, rmse = self._compute_metrics(y, preds)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        self.log("val/mape", mape, prog_bar=True)
        self.log("val/smape", smape_, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class MPGCNLM(L.LightningModule):
    def __init__(
        self,
        model,
        static_supports,
        origin_dynamic_supports,
        destination_dynamic_supports,
        loss=None,
        lr=1e-4,
        weight_decay=0.0,
        train_rollout_steps=1,
        mape_eps=1e-3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss if loss is not None else torch.nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_rollout_steps = train_rollout_steps
        self.mape_eps = mape_eps

        self.register_buffer("static_supports", static_supports, persistent=False)
        self.register_buffer(
            "origin_dynamic_supports",
            origin_dynamic_supports,
            persistent=False,
        )
        self.register_buffer(
            "destination_dynamic_supports",
            destination_dynamic_supports,
            persistent=False,
        )

    def _rollout(self, x, future_keys, steps):
        context = x
        preds = []
        for step_idx in range(steps):
            dyn_key = future_keys[:, step_idx].reshape(-1).long()
            dyn_graphs = (
                torch.index_select(self.origin_dynamic_supports, 0, dyn_key),
                torch.index_select(self.destination_dynamic_supports, 0, dyn_key),
            )
            if dyn_graphs[0].dim() != 4 or dyn_graphs[1].dim() != 4:
                raise ValueError(
                    "Indexed dynamic graph must be (B,K,N,N), "
                    f"got origin={tuple(dyn_graphs[0].shape)}, "
                    f"dest={tuple(dyn_graphs[1].shape)}"
                )
            step_pred = self.model(context, [self.static_supports, dyn_graphs])
            preds.append(step_pred)
            context = torch.cat([context[:, 1:], step_pred], dim=1)
        return torch.cat(preds, dim=1)

    def _compute_metrics(self, y_true, y_pred):
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(torch.clamp(y_pred, min=0.0))
        diff = y_true - y_pred

        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))
        wmape = torch.sum(torch.abs(diff)) / torch.sum(torch.abs(y_true)) * 100.0
        smape_all = smape(y_true, y_pred, eps=self.mape_eps)
        return mse, mae, smape_all, rmse, wmape

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        future_keys = batch["future_keys"]

        steps = min(self.train_rollout_steps, y.shape[1])
        preds = self._rollout(x, future_keys, steps)
        loss = self.loss_fn(preds, y[:, :steps])

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        future_keys = batch["future_keys"]

        preds = self._rollout(x, future_keys, y.shape[1])
        loss = self.loss_fn(preds, y)

        mse, mae, smape_all, rmse, wmape = self._compute_metrics(y, preds)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        self.log("val/smape", smape_all, prog_bar=True)
        self.log("val/wmape", wmape, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class MetroAutoformerODLM(L.LightningModule):
    def __init__(self, model, lr=1e-4, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.mape_eps = mape_eps
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, batch):
        return self.model(
            x_log=batch["x"],
            time_enc_hist=batch["time_hist"],
            time_enc_fut=batch["time_fut"],
            weekday=batch["weekday"],
        )

    def _compute_metrics(self, y_true, y_pred):
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(y_pred)

        diff = y_true - y_pred
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))

        mask = y_true > 0
        if mask.any():
            mape = torch.mean(
                torch.abs(diff[mask] / torch.clamp(y_true[mask], min=self.mape_eps))
            ) * 100.0
        else:
            mape = torch.nan

        smape_ = smape(y_true, y_pred, eps=self.mape_eps)
        return mse, mae, mape, smape_, rmse

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.loss_fn(preds, batch["y"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.loss_fn(preds, batch["y"])

        mse, mae, mape, smape_, rmse = self._compute_metrics(batch["y"], preds)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        self.log("val/mape", mape, prog_bar=True)
        self.log("val/smape", smape_, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
