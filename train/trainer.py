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
        y_true = y_true.squeeze(-1)
        # y_pred is already (B, Pred) due to FC output_dim
        
        mask = (y_true > 0).float()
        mask_sum = mask.sum()

        if mask_sum < 1:
            return None  # skip batch

        diff = (y_pred - y_true)
        mse = ((y_true - y_pred)**2 * mask).sum() / mask_sum
        mae = (torch.abs(y_true - y_pred) * mask).sum() / mask_sum
        
        denom = torch.clamp(torch.abs(y_true), min=self.mape_eps)
        mape = torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0

        denom2 = (y_true.abs() + y_pred.abs()).clamp(min=self.mape_eps)
        smape_ = (2 * diff.abs() / denom2 * mask).sum() / mask_sum * 100.0
        
        rmse = torch.sqrt(mse)
        return mse, mae, mape, smape_, rmse, mask_sum  # mask_sum 같이 넘김(가중 로깅용)
    
    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y'].squeeze(-1)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y'].squeeze(-1)
        y_pred = self(x)
        metrics = self._compute_metrics(y, y_pred)
        if metrics is None:
            return  # <- 이게 정답. log도 안 하고 return도 안 함.
        mse, mae, mape, smape_, rmse, mask_sum = metrics
        bs = int(mask_sum.item())
        self.log("val_mse", mse, prog_bar=True, on_step=False, batch_size=bs)
        self.log("val_mae", mae, prog_bar=True, on_step=False, batch_size=bs)
        self.log("val_mape", mape, prog_bar=True, on_step=False, batch_size=bs)
        self.log("val_smape", smape_, prog_bar=True, on_step=False, batch_size=bs)
        self.log("val_rmse", rmse, prog_bar=True, on_step=False, batch_size=bs)
        
        return mse
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class MPGCNLM(L.LightningModule):
    def __init__(self, model, loss, lr=1e-3, pred_size=1, mape_eps=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.lr = lr
        self.pred_size = pred_size
        self.mape_eps = mape_eps
        
    def construct_dynamic_graph(self, x):
        """
        Construct Sample-wise Dynamic Graph based on Cosine Similarity.
        [cite_start][cite: 1353-1355]
        x: (B, T, N, N)
        Returns: adj_O (B, N, N), adj_D (B, N, N)
        """
        B, T, N, _ = x.shape
        
        # Average over time window for each sample
        flow_mean = x.mean(dim=1) # (B, N, N)
        
        # 1. Origin Dynamic Graph (Similarity between Origins for each sample)
        # norm: (B, N, 1)
        norm_O = torch.norm(flow_mean, p=2, dim=2, keepdim=True) + 1e-6
        normalized_flow_O = flow_mean / norm_O
        # Batch Matrix Multiplication: (B, N, N) @ (B, N, N)^T -> (B, N, N)
        adj_O = torch.bmm(normalized_flow_O, normalized_flow_O.transpose(1, 2))
        
        # 2. Destination Dynamic Graph
        flow_mean_T = flow_mean.transpose(1, 2) # (B, N, N)
        norm_D = torch.norm(flow_mean_T, p=2, dim=2, keepdim=True) + 1e-6
        normalized_flow_D = flow_mean_T / norm_D
        adj_D = torch.bmm(normalized_flow_D, normalized_flow_D.transpose(1, 2))
        
        return adj_O, adj_D

    def forward(self, x):
        # 1. Dynamic Graph Construction (Per Sample)
        with torch.no_grad():
            dyn_adj_O, dyn_adj_D = self.construct_dynamic_graph(x)
            
            # Thresholding (Optional, for sparsity)
            dyn_adj_O = torch.where(dyn_adj_O > 0.5, dyn_adj_O, torch.zeros_like(dyn_adj_O))
            dyn_adj_D = torch.where(dyn_adj_D > 0.5, dyn_adj_D, torch.zeros_like(dyn_adj_D))

        # 2. Model Forward (Passing Batch of Graphs)
        # model expects (B, N, N) dynamic graphs
        y_pred = self.model(x, dynamic_adj_O=dyn_adj_O, dynamic_adj_D=dyn_adj_D)
        return y_pred

    def _compute_metrics(self, y_true, y_pred):
        y_true = y_true.squeeze(-1)
        # y_pred is already (B, Pred) due to FC output_dim
        
        mse = torch.mean((y_true - y_pred) ** 2)
        mae = torch.mean(torch.abs(y_true - y_pred))
        denom = torch.clamp(torch.abs(y_true), min=self.mape_eps)
        mape = torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0
        smape_ = smape(y_true, y_pred, eps=self.mape_eps)
        rmse = torch.sqrt(mse)
        return mse, mae, mape, smape_, rmse
    
    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y'] # (B, Pred, N, N)
        
        y_pred = self(x) # (B, N, N)
        
        if y_pred.dim() == 3: y_pred = y_pred.unsqueeze(1)
        
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        
        y_pred = self(x)
        if y_pred.dim() == 3: y_pred = y_pred.unsqueeze(1)
        
        mse, mae, mape, smape_, rmse = self._compute_metrics(y, y_pred)
        self.log("val_mse", mse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        self.log("val_smape", smape_, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        return mse

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
    def __init__(self, model, loss=torch.nn.MSELoss(), lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.lr = lr

    def forward(self, static_edge_index, batch_graph, B, T):
        return self.model(static_edge_index, batch_graph, B, T)

    def training_step(self, batch, batch_idx):
        # batch is a tuple from graph_collate_fn
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, weekday = batch

        # forward
        preds = self.model(static_edge_index, batch_graph, B, T, time_enc_hist, weekday)
        loss = self.loss_fn(preds, labels)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        static_edge_index, batch_graph, B, T, labels, time_enc_hist, weekday = batch

        # forward
        preds = self.model(static_edge_index, batch_graph, B, T, time_enc_hist, weekday)
        loss = self.loss_fn(preds, labels)

        self.log("val/loss", loss, prog_bar=True)
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
        """
        y_true, y_pred: (B, |V|)
        """
        mask = (y_true > 0).float()

        mse = ((y_true - y_pred) ** 2 * mask).sum() / mask.sum()
        mae = (torch.abs(y_true - y_pred) * mask).sum() / mask.sum()

        denom = torch.clamp(torch.abs(y_true), min=self.mape_eps)
        mape = (torch.abs((y_true - y_pred) / denom) * mask).sum() / mask.sum() * 100

        smape_ = smape(
            y_true * mask,
            y_pred * mask,
            eps=self.mape_eps
        )
        rmse = torch.sqrt(mse)

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
        """
        y_true, y_pred: (B, |V|)
        """
        # Option 1: log1p 사용 시
        y_pred = torch.expm1(y_pred)
        y_true = torch.expm1(y_true)
        mask = (y_true > 0).float()
        den = torch.clamp(mask.sum(), min=1.0)

        mse = ((y_true - y_pred) ** 2 * mask).sum() / den
        mae = (torch.abs(y_true - y_pred) * mask).sum() / den

        denom = torch.clamp(torch.abs(y_true), min=self.mape_eps)
        mape = (torch.abs((y_true - y_pred) / denom) * mask).sum() / den * 100

        smape_ = smape(
            y_true * mask,
            y_pred * mask,
            eps=self.mape_eps
        )
        rmse = torch.sqrt(mse)

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

        self.log("val_mse", mse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        self.log("val_smape", smape_, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
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
