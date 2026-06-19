import argparse
import gc
import datetime as dt
import json
import math
import os
import re
import sys
import time

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataset import get_dataset


def _list_day_files(path):
    files = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.endswith(".npy")
        and not name.endswith(".time.npy")
        and not name.endswith(".mask.npy")
    )
    if not files:
        raise FileNotFoundError(f"No OD .npy files found under {path}")
    return files


def _weekday_from_path(path):
    match = re.search(r"(\d{8})", os.path.basename(path))
    if match is None:
        raise ValueError(f"Cannot parse yyyymmdd from {path}")
    ymd = match.group(1)
    return dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])).weekday()


def _new_state():
    return {
        "sq_error_sum": 0.0,
        "abs_error_sum": 0.0,
        "true_abs_sum": 0.0,
        "smape_term_sum": 0.0,
        "count": 0,
    }


def _update_state(state, pred, true, eps):
    diff = pred - true
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(pred) + np.abs(true), eps)
    state["sq_error_sum"] += float(np.sum(diff ** 2))
    state["abs_error_sum"] += float(np.sum(abs_diff))
    state["true_abs_sum"] += float(np.sum(np.abs(true)))
    state["smape_term_sum"] += float(np.sum(2.0 * abs_diff / denom))
    state["count"] += int(diff.size)


def _finalize_state(state):
    if state["count"] == 0:
        raise RuntimeError("No samples were evaluated.")
    mse = state["sq_error_sum"] / state["count"]
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": state["abs_error_sum"] / state["count"],
        "smape": 100.0 * state["smape_term_sum"] / state["count"],
        "wmape": 100.0 * state["abs_error_sum"] / (state["true_abs_sum"] + 1e-8),
    }


def _save_results(results, output_json):
    results["updated_at_utc"] = dt.datetime.utcnow().isoformat() + "Z"
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{output_json}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    os.replace(tmp_path, output_json)


def _load_results(output_json):
    with open(output_json, "r", encoding="utf-8") as fp:
        return json.load(fp)


def train_ha(train_dir, time_resolution):
    sums_wm = {}
    counts_wm = {}
    sums_m = {}
    counts_m = {}
    global_sum = None
    global_count = 0

    for path in tqdm(_list_day_files(train_dir), desc="HA train"):
        weekday = _weekday_from_path(path)
        arr = np.load(path, mmap_mode="r")
        for step in range(arr.shape[0]):
            minute = (step * time_resolution) % 1440
            mat = np.asarray(arr[step], dtype=np.float64)
            key_wm = (weekday, minute)
            if key_wm not in sums_wm:
                sums_wm[key_wm] = mat.copy()
                counts_wm[key_wm] = 1
            else:
                sums_wm[key_wm] += mat
                counts_wm[key_wm] += 1

            if minute not in sums_m:
                sums_m[minute] = mat.copy()
                counts_m[minute] = 1
            else:
                sums_m[minute] += mat
                counts_m[minute] += 1

            if global_sum is None:
                global_sum = mat.copy()
            else:
                global_sum += mat
            global_count += 1

    if global_sum is None:
        raise RuntimeError("No training matrices found for HA.")

    return {
        "by_weekday_minute": {key: sums_wm[key] / counts_wm[key] for key in sums_wm},
        "by_minute": {key: sums_m[key] / counts_m[key] for key in sums_m},
        "global": global_sum / global_count,
    }


def evaluate_ha(ha_model, dataset, time_resolution, eps):
    state = _new_state()
    by_wm = ha_model["by_weekday_minute"]
    by_m = ha_model["by_minute"]
    global_mean = ha_model["global"]

    for idx in tqdm(range(len(dataset)), desc="HA eval"):
        info = dataset.info_list[idx]
        weekday = int(info["weekday"])
        start_step = int(info["start_step"])
        item = dataset[idx]
        true = np.expm1(item["y_tensor"].numpy()).astype(np.float64, copy=False)
        pred = np.empty_like(true)
        for t in range(true.shape[0]):
            minute = ((start_step + dataset.window_size + t) * time_resolution) % 1440
            pred[t] = by_wm.get((weekday, minute), by_m.get(minute, global_mean))
        _update_state(state, pred, true, eps)

    return _finalize_state(state)


def _resolve_pair_slice(flat, pair_start, pair_end):
    total_pairs = int(flat.shape[1])
    end = total_pairs if pair_end is None else min(total_pairs, pair_end)
    if pair_start < 0 or pair_start >= end:
        raise ValueError(f"Invalid pair slice: start={pair_start}, end={end}, total={total_pairs}")
    return total_pairs, end


def collect_series_from_dataset(dataset, hop_size, pair_start=0, pair_end=None):
    chunks = []
    total_pairs = None
    resolved_end = None
    for idx in tqdm(range(len(dataset)), desc="ARIMA collect train series"):
        y = np.expm1(dataset[idx]["y_tensor"].numpy()).astype(np.float32, copy=False)
        flat = y.reshape(y.shape[0], -1)
        if total_pairs is None:
            total_pairs, resolved_end = _resolve_pair_slice(flat, pair_start, pair_end)
        flat = flat[:, pair_start:resolved_end]
        chunks.append(flat if idx == 0 else flat[-hop_size:])
    if not chunks:
        raise RuntimeError("No ARIMA training series collected.")
    return np.concatenate(chunks, axis=0), total_pairs, resolved_end


def collect_eval_targets(dataset, pair_start=0, pair_end=None):
    targets = []
    total_pairs = None
    resolved_end = None
    for idx in tqdm(range(len(dataset)), desc="ARIMA collect eval targets"):
        y = np.expm1(dataset[idx]["y_tensor"].numpy()).astype(np.float32, copy=False)
        flat = y.reshape(y.shape[0], -1)
        if total_pairs is None:
            total_pairs, resolved_end = _resolve_pair_slice(flat, pair_start, pair_end)
        targets.append(flat[:, pair_start:resolved_end])
    if not targets:
        raise RuntimeError("No ARIMA evaluation targets collected.")
    return targets, total_pairs, resolved_end


def _update_arima_state(state, series, eval_targets, hop_size, order, eps, desc="ARIMA train+eval"):
    pair_limit = int(series.shape[1])
    fitted = 0
    constant_fallback = 0

    for pair_idx in tqdm(range(pair_limit), desc=desc):
        seq = np.asarray(series[:, pair_idx], dtype=np.float64)
        model = None
        if np.isfinite(seq).all() and np.std(seq) >= 1e-8:
            try:
                model = ARIMA(seq, order=order).fit()
                fitted += 1
            except Exception:
                model = None
        if model is None:
            constant_fallback += 1

        for target in eval_targets:
            true = np.asarray(target[:, pair_idx], dtype=np.float64)
            if model is None:
                pred = np.full_like(true, seq[-1] if len(seq) else 0.0)
            else:
                try:
                    pred = np.asarray(model.forecast(true.shape[0]), dtype=np.float64)
                except Exception:
                    pred = np.full_like(true, seq[-1] if len(seq) else 0.0)
            if not np.isfinite(pred).all():
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            _update_state(state, pred, true, eps)
            if model is not None:
                try:
                    model = model.append(true[:hop_size], refit=False)
                except Exception:
                    pass

    return fitted, constant_fallback


def train_eval_arima(series, eval_targets, hop_size, order, eps, total_pairs=None, pair_start=0):
    state = _new_state()
    pair_limit = int(series.shape[1])
    total_pairs = pair_limit if total_pairs is None else int(total_pairs)
    fitted, constant_fallback = _update_arima_state(
        state,
        series,
        eval_targets,
        hop_size,
        order,
        eps,
    )

    metrics = _finalize_state(state)
    metrics.update(
        {
            "evaluated_pairs": pair_limit,
            "total_pairs": total_pairs,
            "pair_start": int(pair_start),
            "pair_end": int(pair_start + pair_limit),
            "fitted_pairs": fitted,
            "constant_fallback_pairs": constant_fallback,
        }
    )
    return metrics


def train_eval_arima_chunked(train_dataset, eval_dataset, hop_size, order, eps, max_pairs, chunk_size):
    sample = train_dataset[0]["y_tensor"].numpy()
    total_pairs = int(sample.reshape(sample.shape[0], -1).shape[1])
    pair_limit = total_pairs if max_pairs is None else min(total_pairs, max_pairs)
    if chunk_size <= 0:
        raise ValueError("--arima_pair_chunk_size must be positive for chunked ARIMA.")

    state = _new_state()
    fitted = 0
    constant_fallback = 0

    for pair_start in range(0, pair_limit, chunk_size):
        pair_end = min(pair_limit, pair_start + chunk_size)
        print(f"[ARIMA chunk] pair slice {pair_start}:{pair_end} / {pair_limit}", flush=True)
        series, train_total_pairs, resolved_end = collect_series_from_dataset(
            train_dataset,
            hop_size,
            pair_start=pair_start,
            pair_end=pair_end,
        )
        targets, test_total_pairs, test_resolved_end = collect_eval_targets(
            eval_dataset,
            pair_start=pair_start,
            pair_end=resolved_end,
        )
        if train_total_pairs != total_pairs or test_total_pairs != total_pairs:
            raise ValueError(
                f"Train/Test OD size mismatch: train={train_total_pairs}, test={test_total_pairs}, expected={total_pairs}"
            )
        if test_resolved_end != resolved_end:
            raise ValueError(
                f"Train/Test pair slice mismatch: train_end={resolved_end}, test_end={test_resolved_end}"
            )
        chunk_fitted, chunk_fallback = _update_arima_state(
            state,
            series,
            targets,
            hop_size,
            order,
            eps,
            desc=f"ARIMA train+eval {pair_start}:{resolved_end}",
        )
        fitted += chunk_fitted
        constant_fallback += chunk_fallback
        del series, targets
        gc.collect()

    metrics = _finalize_state(state)
    metrics.update(
        {
            "evaluated_pairs": pair_limit,
            "total_pairs": total_pairs,
            "pair_start": 0,
            "pair_end": int(pair_limit),
            "fitted_pairs": fitted,
            "constant_fallback_pairs": constant_fallback,
            "pair_chunk_size": int(chunk_size),
        }
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="HA and ARIMA baselines for reviewer 3-way split.")
    parser.add_argument("--data_root", default="/root/tmp/Trasport/data_splits/od_minute_review_3way")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--mape_eps", type=float, default=1e-3)
    parser.add_argument("--arima_order", default="1,1,1")
    parser.add_argument("--arima_max_pairs", type=int, default=0, help="0 means all OD pairs.")
    parser.add_argument("--arima_pair_chunk_size", type=int, default=0, help="0 keeps legacy all-at-once collection.")
    parser.add_argument("--reuse_ha_if_available", action="store_true")
    parser.add_argument("--skip_arima", action="store_true")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    order = tuple(int(part) for part in args.arima_order.split(","))
    if len(order) != 3:
        raise ValueError("--arima_order must have 3 comma-separated integers.")

    train_dir = os.path.join(args.data_root, args.train_subdir)
    results = None
    if args.reuse_ha_if_available and os.path.exists(args.output_json):
        try:
            existing = _load_results(args.output_json)
            if existing.get("status", {}).get("HA") == "complete" and "HA" in existing:
                results = existing
                results["config"] = vars(args)
                results.setdefault("status", {})["ARIMA"] = "skipped" if args.skip_arima else "pending"
                print(f"[Reuse HA results] {args.output_json}", flush=True)
        except Exception as exc:
            print(f"[Warn] Could not reuse existing HA results: {exc}", flush=True)

    if results is None:
        _, test_dataset = get_dataset(
            data_root=args.data_root,
            train_subdir=args.test_subdir,
            val_subdir=args.test_subdir,
            window_size=args.window_size,
            hop_size=args.hop_size,
            pred_size=args.pred_size,
            time_resolution=args.time_resolution,
            cache_in_mem=False,
        )
        results = {
            "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
            "config": vars(args),
            "status": {
                "HA": "pending",
                "ARIMA": "skipped" if args.skip_arima else "pending",
            },
        }

        start = time.perf_counter()
        ha = train_ha(train_dir, args.time_resolution)
        results["HA"] = evaluate_ha(ha, test_dataset, args.time_resolution, args.mape_eps)
        results["HA"]["elapsed_sec"] = time.perf_counter() - start
        results["status"]["HA"] = "complete"
        _save_results(results, args.output_json)
        print(f"[Saved partial HA results] {args.output_json}", flush=True)
        del ha, test_dataset
        gc.collect()

    if not args.skip_arima:
        train_dataset, arima_test_dataset = get_dataset(
            data_root=args.data_root,
            train_subdir=args.train_subdir,
            val_subdir=args.test_subdir,
            window_size=args.window_size,
            hop_size=args.hop_size,
            pred_size=args.pred_size,
            time_resolution=args.time_resolution,
            cache_in_mem=False,
        )
        start = time.perf_counter()
        max_pairs = None if args.arima_max_pairs == 0 else args.arima_max_pairs
        if args.arima_pair_chunk_size > 0:
            results["ARIMA"] = train_eval_arima_chunked(
                train_dataset=train_dataset,
                eval_dataset=arima_test_dataset,
                hop_size=args.hop_size,
                order=order,
                eps=args.mape_eps,
                max_pairs=max_pairs,
                chunk_size=args.arima_pair_chunk_size,
            )
        else:
            pair_end = None if max_pairs is None else max_pairs
            series, total_pairs, resolved_end = collect_series_from_dataset(
                train_dataset,
                args.hop_size,
                pair_start=0,
                pair_end=pair_end,
            )
            targets, test_total_pairs, test_resolved_end = collect_eval_targets(
                arima_test_dataset,
                pair_start=0,
                pair_end=resolved_end,
            )
            if test_total_pairs != total_pairs:
                raise ValueError(f"Train/Test OD size mismatch: train={total_pairs}, test={test_total_pairs}")
            if test_resolved_end != resolved_end:
                raise ValueError(f"Train/Test pair slice mismatch: train_end={resolved_end}, test_end={test_resolved_end}")
            results["ARIMA"] = train_eval_arima(
                series=series,
                eval_targets=targets,
                hop_size=args.hop_size,
                order=order,
                eps=args.mape_eps,
                total_pairs=total_pairs,
                pair_start=0,
            )
        results["ARIMA"]["elapsed_sec"] = time.perf_counter() - start
        results["status"]["ARIMA"] = "complete"

    _save_results(results, args.output_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
