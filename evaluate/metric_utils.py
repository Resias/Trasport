import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def postprocess_for_metrics(y_true, y_pred, output_domain="raw", clip_predictions=True, clip_targets=True):
    y_true = to_numpy(y_true).astype(np.float64, copy=False)
    y_pred = to_numpy(y_pred).astype(np.float64, copy=False)

    if output_domain == "log1p":
        y_true = np.expm1(y_true)
        if clip_predictions:
            y_pred = np.expm1(np.clip(y_pred, a_min=0.0, a_max=None))
        else:
            y_pred = np.expm1(y_pred)
    elif output_domain in {"raw", "original_scale"}:
        pass
    else:
        raise ValueError(f"Unsupported output_domain: {output_domain}")

    if clip_targets:
        y_true = np.clip(y_true, a_min=0.0, a_max=None)

    if output_domain != "log1p" and clip_predictions:
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    return y_true, y_pred


def compute_train_graph_style_metrics(y_true, y_pred, eps=1e-3):
    y_true = to_numpy(y_true).reshape(-1).astype(np.float64, copy=False)
    y_pred = to_numpy(y_pred).reshape(-1).astype(np.float64, copy=False)

    diff = y_true - y_pred
    abs_diff = np.abs(diff)

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff)
    smape = np.mean(
        2.0 * abs_diff / np.clip(np.abs(y_true) + np.abs(y_pred), a_min=eps, a_max=None)
    ) * 100.0
    wmape = (np.sum(abs_diff) / (np.sum(np.abs(y_true)) + 1e-8)) * 100.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "smape": float(smape),
        "wmape": float(wmape),
    }
