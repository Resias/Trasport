import pytorch_lightning as L
import torch
from torch.optim import Adam

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
        return mse, mae, mape

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
        mse, mae, mape = self._compute_metrics(y_tensor, y_pred)
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
        return mse
    
    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr)
        return optim


class TCNMetroLM(L.LightningModule):
    def __init__(self, model, loss, lr, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.mape_eps = mape_eps

    def forward_batch(self, batch):
        x = batch["x"]          # (B, T, F)
        y = batch["y"]          # (B, pred_size, 1)
        pred, att = self.model(x)
        return pred, y, x.size(0)

    def _compute_metrics(self, y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)
        mae = torch.mean(torch.abs(y_true - y_pred))
        denom = torch.clamp(torch.abs(y_true), min=self.mape_eps)
        mape = torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0
        return mse, mae, mape

    def training_step(self, batch, batch_idx):
        y_pred, y_true, batch_size = self.forward_batch(batch)
        y_last = y_true[:, -1, :]  # (B, 1)
        loss = self.loss(y_last, y_pred)

        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_true, batch_size = self.forward_batch(batch)
        y_last = y_true[:, -1, :]

        mse, mae, mape = self._compute_metrics(y_last, y_pred)

        self.log("val_mse", mse, batch_size=batch_size, on_epoch=True)
        self.log("val_mae", mae, batch_size=batch_size, on_epoch=True)
        self.log("val_mape", mape, batch_size=batch_size, on_epoch=True)

        return mse

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)