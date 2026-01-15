import torch
import numpy as np

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    smape = 100*np.mean(2*np.abs(preds - trues)/(np.abs(preds)+np.abs(trues)))
    return mae, rmse, smape
