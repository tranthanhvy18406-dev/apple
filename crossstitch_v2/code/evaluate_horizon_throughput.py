import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Train"))

from mtl_dataset import MTL_Dataset
from mtl_model import CrossStitch_MTL_Model
from train_direct_rate_gru import calculate_v2_throughput_series


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate horizon-wise throughput prediction for Last Value, RF-Rate, and PRNet."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/4T/xty/new_mcs_dataset/single_user_pf_5Hz"),
    )
    parser.add_argument(
        "--prnet_model_path",
        type=Path,
        default=ROOT / "experiments_all" / "single_user_pf_5Hz" / "best_mtl_model_decoder.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "paper" / "horizon_throughput_5Hz",
    )
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--max_horizon", type=int, default=10)
    parser.add_argument("--skip_frames", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cross_stitch_mode", type=str, default="learn", choices=["learn", "identity", "zeros", "none"])
    parser.add_argument("--rf_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=2)
    parser.add_argument("--rf_jobs", type=int, default=-1)
    parser.add_argument(
        "--rf_feature_mode",
        type=str,
        default="all",
        choices=["all", "rate_only", "rb_sinr", "last_rb_sinr"],
        help="Feature set for the RF-Rate baseline.",
    )
    parser.add_argument(
        "--rf_output_mode",
        type=str,
        default="multi_horizon",
        choices=["multi_horizon", "one_step_hold"],
        help="Whether RF predicts all horizons directly or predicts h=1 and holds that rate for h=1..H.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rf", action="store_true")
    parser.add_argument(
        "--prnet_blend",
        type=str,
        default="none",
        choices=["none", "train_persistence"],
        help="Optionally blend PRNet RB/SINR predictions with the last observed RB/SINR before the protocol block.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="v2",
        choices=["v2", "raw"],
        help="Throughput label used for Last Value/RF/PRNet comparison. v2 matches compare_scheduling_v2.",
    )
    parser.add_argument(
        "--last_value_mode",
        type=str,
        default="last",
        choices=["last", "history_mean"],
        help="Last Value baseline variant: repeat the previous frame or the mean of the history window.",
    )
    return parser.parse_args()


def flatten_series(obj, key: str) -> np.ndarray:
    if isinstance(obj, dict):
        aliases = {
            "SINR": ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR", "eff_sinr"],
            "RB": ["RB", "rb", "Rb", "rB"],
            "Throughput": ["Throughput", "throughput"],
        }
        keys = aliases.get(key, [key])
        data = None
        for candidate in keys:
            if candidate in obj:
                data = obj[candidate]
                break
        if data is None:
            raise KeyError(f"None of {keys} found in dataset keys {list(obj.keys())}.")
    else:
        data = obj
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def load_trace(path: Path, skip_frames: int = 0, target_mode: str = "v2") -> dict[str, np.ndarray]:
    payload = pd.read_pickle(path)
    rb = flatten_series(payload, "RB")
    sinr = flatten_series(payload, "SINR")
    throughput_raw = flatten_series(payload, "Throughput")
    if target_mode == "v2":
        throughput = calculate_v2_throughput_series(rb, sinr)
    elif target_mode == "raw":
        throughput = throughput_raw
    else:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")

    trace = {
        "rb": rb,
        "sinr": sinr,
        "throughput": throughput,
    }
    if skip_frames > 0:
        trace = {key: value[skip_frames:] for key, value in trace.items()}
    return trace


def add_history_features(values: np.ndarray) -> list[np.ndarray]:
    idx = np.arange(values.shape[1], dtype=np.float32)
    idx = idx - idx.mean()
    denom = float(np.sum(idx ** 2) + 1e-8)
    centered = values - values.mean(axis=1, keepdims=True)
    slope = centered @ idx / denom
    return [
        values,
        values[:, -1:],
        values.mean(axis=1, keepdims=True),
        values.std(axis=1, keepdims=True),
        values.min(axis=1, keepdims=True),
        values.max(axis=1, keepdims=True),
        (values[:, -1:] - values[:, :1]),
        slope[:, None],
    ]


def make_rate_windows(
    trace: dict[str, np.ndarray],
    hist_window: int,
    horizon: int,
    last_value_mode: str,
    rf_feature_mode: str = "all",
):
    n = min(len(trace["rb"]), len(trace["sinr"]), len(trace["throughput"]))
    samples = n - hist_window - horizon + 1
    if samples <= 0:
        raise ValueError("Not enough frames for the requested history and horizon.")

    starts = np.arange(samples)
    hist_idx = starts[:, None] + np.arange(hist_window)[None, :]
    target_idx = starts[:, None] + hist_window + np.arange(horizon)[None, :]

    rb_hist = trace["rb"][hist_idx]
    sinr_hist = trace["sinr"][hist_idx]
    thr_hist = trace["throughput"][hist_idx]
    target = trace["throughput"][target_idx]

    if rf_feature_mode == "all":
        feature_series = (thr_hist, rb_hist, sinr_hist)
    elif rf_feature_mode == "rate_only":
        feature_series = (thr_hist,)
    elif rf_feature_mode == "rb_sinr":
        feature_series = (rb_hist, sinr_hist)
    elif rf_feature_mode == "last_rb_sinr":
        feature_series = ()
    else:
        raise ValueError(f"Unsupported rf_feature_mode={rf_feature_mode!r}")

    feature_blocks = []
    for series in feature_series:
        feature_blocks.extend(add_history_features(series.astype(np.float32, copy=False)))
    if rf_feature_mode == "last_rb_sinr":
        feature_blocks.extend([rb_hist[:, -1:], sinr_hist[:, -1:]])
    features = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    if last_value_mode == "last":
        baseline_value = thr_hist[:, -1:]
    elif last_value_mode == "history_mean":
        baseline_value = thr_hist.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported last_value_mode={last_value_mode!r}")
    last_value = np.repeat(baseline_value, horizon, axis=1).astype(np.float32, copy=False)
    return features, target.astype(np.float32, copy=False), last_value


def train_or_load_rf(args, train_x: np.ndarray, train_y: np.ndarray):
    rf_path = (
        args.output_dir
        / (
            f"rf_rate_model_{args.rf_feature_mode}_{args.rf_output_mode}_{args.target_mode}"
            f"_h{args.hist_window}_p{args.max_horizon}.joblib"
        )
    )
    if rf_path.exists() and not args.force_rf:
        return joblib.load(rf_path), rf_path

    target = train_y[:, 0] if args.rf_output_mode == "one_step_hold" else train_y
    model = RandomForestRegressor(
        n_estimators=args.rf_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.rf_jobs,
    )
    model.fit(train_x, target)
    joblib.dump(model, rf_path)
    return model, rf_path


def build_prnet(args, device):
    return CrossStitch_MTL_Model(
        rb_input_size=1,
        sinr_input_size=1,
        rb_hidden_size=256,
        sinr_hidden_size=64,
        rb_num_layers=3,
        sinr_num_layers=1,
        rb_output_size=106,
        sinr_output_size=1,
        pre_window=args.max_horizon,
        device=device,
        cross_stitch_mode=getattr(args, "cross_stitch_mode", "learn"),
    ).to(device)


@torch.no_grad()
def predict_prnet_components(args, data_path: Path, sinr_mean: float, sinr_std: float, skip_frames: int = 0):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = MTL_Dataset(
        str(data_path),
        str(data_path),
        obs_window=args.hist_window,
        pre_window=args.max_horizon,
        skip_initial_frames=skip_frames,
        sinr_mean=sinr_mean,
        sinr_std=sinr_std,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_prnet(args, device)
    model.load_state_dict(torch.load(args.prnet_model_path, map_location=device, weights_only=True))
    model.eval()

    chunks = []
    rb_chunks = []
    sinr_chunks = []
    last_rb_chunks = []
    last_sinr_chunks = []
    for rb_seq, sinr_seq, _, _ in loader:
        rb_seq = rb_seq.to(device)
        sinr_seq = sinr_seq.to(device)
        rb_logits, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)
        rb_pred = (torch.argmax(rb_logits, dim=2) + 1).cpu().numpy().astype(np.float32)
        sinr_pred = (sinr_pred.squeeze(-1) * sinr_std + sinr_mean).cpu().numpy().astype(np.float32)
        throughput = calculate_v2_throughput_series(rb_pred.reshape(-1), sinr_pred.reshape(-1))
        chunks.append(throughput.reshape(rb_pred.shape))
        rb_chunks.append(rb_pred)
        sinr_chunks.append(sinr_pred)
        last_rb_chunks.append(rb_seq[:, -1, 0].cpu().numpy().astype(np.float32))
        last_sinr_chunks.append((sinr_seq[:, -1, 0] * sinr_std + sinr_mean).cpu().numpy().astype(np.float32))
    return {
        "throughput": np.concatenate(chunks, axis=0),
        "rb": np.concatenate(rb_chunks, axis=0),
        "sinr": np.concatenate(sinr_chunks, axis=0),
        "last_rb": np.concatenate(last_rb_chunks, axis=0),
        "last_sinr": np.concatenate(last_sinr_chunks, axis=0),
    }


def apply_prnet_blend(components: dict[str, np.ndarray], alphas: list[tuple[float, float]]) -> np.ndarray:
    chunks = []
    for h, (alpha_rb, alpha_sinr) in enumerate(alphas):
        rb = np.rint(
            alpha_rb * components["rb"][:, h] + (1.0 - alpha_rb) * components["last_rb"]
        ).clip(1, 106)
        sinr = alpha_sinr * components["sinr"][:, h] + (1.0 - alpha_sinr) * components["last_sinr"]
        chunks.append(calculate_v2_throughput_series(rb, sinr))
    return np.stack(chunks, axis=1).astype(np.float32, copy=False)


def learn_prnet_blend_alphas(components: dict[str, np.ndarray], target: np.ndarray) -> list[tuple[float, float]]:
    grid = np.linspace(0.0, 1.0, 21)
    alphas = []
    for h in range(target.shape[1]):
        best_mae = float("inf")
        best_pair = (1.0, 1.0)
        for alpha_rb in grid:
            for alpha_sinr in grid:
                rb = np.rint(
                    alpha_rb * components["rb"][:, h] + (1.0 - alpha_rb) * components["last_rb"]
                ).clip(1, 106)
                sinr = alpha_sinr * components["sinr"][:, h] + (1.0 - alpha_sinr) * components["last_sinr"]
                pred = calculate_v2_throughput_series(rb, sinr)
                mae = float(np.mean(np.abs(pred - target[:, h])))
                if mae < best_mae:
                    best_mae = mae
                    best_pair = (float(alpha_rb), float(alpha_sinr))
        alphas.append(best_pair)
    return alphas


def horizon_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    err = pred - target
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err ** 2, axis=0)
    return mae, mse


def plot_metric(horizons, curves: dict[str, np.ndarray], ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(6.4, 4.2))
    styles = {
        "Last Value": ("#4C78A8", "o"),
        "RF-Rate": ("#F58518", "s"),
        "PRNet": ("#54A24B", "^"),
    }
    for name, values in curves.items():
        color, marker = styles[name]
        plt.plot(horizons, values, marker=marker, linewidth=2.2, markersize=5, label=name, color=color)
    plt.xlabel("Prediction horizon h (frame)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(horizons)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_trace = load_trace(
        args.dataset_dir / "train_9000_HDF5.pkl",
        skip_frames=args.skip_frames,
        target_mode=args.target_mode,
    )
    test_trace = load_trace(args.dataset_dir / "test_1000_HDF5.pkl", target_mode=args.target_mode)

    train_x, train_y, _ = make_rate_windows(
        train_trace,
        args.hist_window,
        args.max_horizon,
        args.last_value_mode,
        args.rf_feature_mode,
    )
    test_x, test_y, last_pred = make_rate_windows(
        test_trace,
        args.hist_window,
        args.max_horizon,
        args.last_value_mode,
        args.rf_feature_mode,
    )

    rf_model, rf_path = train_or_load_rf(args, train_x, train_y)
    rf_pred = rf_model.predict(test_x).astype(np.float32, copy=False)
    if args.rf_output_mode == "one_step_hold":
        rf_pred = np.repeat(rf_pred[:, None], args.max_horizon, axis=1)

    train_dataset_for_stats = MTL_Dataset(
        str(args.dataset_dir / "train_9000_HDF5.pkl"),
        str(args.dataset_dir / "train_9000_HDF5.pkl"),
        obs_window=args.hist_window,
        pre_window=args.max_horizon,
        skip_initial_frames=args.skip_frames,
    )
    sinr_mean = float(train_dataset_for_stats.sinr_mean)
    sinr_std = float(train_dataset_for_stats.sinr_std)
    test_components = predict_prnet_components(
        args,
        args.dataset_dir / "test_1000_HDF5.pkl",
        sinr_mean=sinr_mean,
        sinr_std=sinr_std,
    )
    blend_alphas = None
    if args.prnet_blend == "train_persistence":
        train_components = predict_prnet_components(
            args,
            args.dataset_dir / "train_9000_HDF5.pkl",
            sinr_mean=sinr_mean,
            sinr_std=sinr_std,
            skip_frames=args.skip_frames,
        )
        blend_alphas = learn_prnet_blend_alphas(train_components, train_y)
        prnet_pred = apply_prnet_blend(test_components, blend_alphas)
    else:
        prnet_pred = test_components["throughput"]

    methods = {
        "Last Value": last_pred,
        "RF-Rate": rf_pred,
        "PRNet": prnet_pred,
    }
    horizons = np.arange(1, args.max_horizon + 1)
    mae_curves = {}
    mse_curves = {}
    for name, pred in methods.items():
        mae, mse = horizon_metrics(pred, test_y)
        mae_curves[name] = mae
        mse_curves[name] = mse

    rows = []
    for i, h in enumerate(horizons):
        row = {"h": int(h)}
        for name in methods:
            key = name.lower().replace(" ", "_").replace("-", "_")
            row[f"{key}_mae"] = float(mae_curves[name][i])
            row[f"{key}_mse"] = float(mse_curves[name][i])
        rows.append(row)

    csv_path = args.output_dir / "horizon_throughput_metrics.csv"
    json_path = args.output_dir / "horizon_throughput_metrics.json"
    mae_path = args.output_dir / "throughput_mae_by_horizon.png"
    mse_path = args.output_dir / "throughput_mse_by_horizon.png"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset_dir.name,
                "hist_window": args.hist_window,
                "max_horizon": args.max_horizon,
                "target_mode": args.target_mode,
                "last_value_mode": args.last_value_mode,
                "rf_feature_mode": args.rf_feature_mode,
                "rf_output_mode": args.rf_output_mode,
                "prnet_blend": args.prnet_blend,
                "prnet_blend_alphas": blend_alphas,
                "rf_model_path": str(rf_path),
                "prnet_model_path": str(args.prnet_model_path),
                "metrics": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plot_metric(
        horizons,
        mae_curves,
        ylabel="Throughput MAE(h)",
        title="Throughput MAE by Prediction Horizon",
        out_path=mae_path,
    )
    plot_metric(
        horizons,
        mse_curves,
        ylabel="Throughput MAE (bps)",
        title="Throughput MSE by Prediction Horizon",
        out_path=mse_path,
    )

    print(json.dumps({"csv": str(csv_path), "mae_plot": str(mae_path), "mse_plot": str(mse_path)}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
