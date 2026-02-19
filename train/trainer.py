import pytorch_lightning as L
import torch
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
    def __init__(self, model, loss=torch.nn.MSELoss(), lr=1e-3, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.lr = lr
        self.mape_eps = mape_eps

    def forward(self, static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday):
        return self.model(static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday)


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
        # batch is a tuple from graph_collate_fn
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday = batch

        # forward
        preds = self.model(static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday)
        loss = self.loss_fn(preds, labels)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, time_enc_fut, weekday = batch

        # forward
        preds = self.model(static_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday)
        loss = self.loss_fn(preds, labels)
        mse, mae, mape, smape_, rmse = self._compute_metrics(labels, preds)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        self.log("val/mape", mape, prog_bar=True)
        self.log("val/smape", smape_, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)
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