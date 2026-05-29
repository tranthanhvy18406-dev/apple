import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import savemat

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Train"))

from compare_scheduling_v2 import build_model

_MCS_TABLE = torch.tensor(
    [
        [1, 4, 0.10, -6.4131],
        [2, 4, 0.13, -4.7292],
        [3, 4, 0.17, -3.6312],
        [4, 4, 0.22, -2.9651],
        [5, 4, 0.25, -2.0542],
        [6, 4, 0.34, -1.2451],
        [7, 4, 0.40, -0.3223],
        [8, 4, 0.45, 0.8364],
        [9, 4, 0.52, 1.5261],
        [10, 4, 0.59, 2.5709],
        [11, 16, 0.31, 3.1961],
        [12, 16, 0.32, 3.7165],
        [13, 16, 0.37, 4.9642],
        [14, 16, 0.45, 5.4653],
        [15, 16, 0.47, 6.9965],
        [16, 16, 0.54, 7.1562],
        [17, 16, 0.57, 7.2301],
        [18, 16, 0.59, 7.9862],
        [19, 64, 0.35, 8.0651],
        [20, 64, 0.38, 8.6985],
        [21, 64, 0.41, 9.0224],
        [22, 64, 0.43, 9.3017],
        [23, 64, 0.45, 9.9628],
        [24, 64, 0.47, 10.3957],
        [25, 64, 0.49, 10.7214],
        [26, 64, 0.55, 12.0541],
        [27, 64, 0.61, 12.8769],
        [28, 64, 0.63, 13.5547],
        [29, 64, 0.65, 14.6139],
    ],
    dtype=torch.float32,
)

_TBS_TABLE = torch.tensor(
    [
        24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
        104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
        184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
        336, 352, 368, 384, 408, 432, 456, 480, 504, 528,
        552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
        888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
        1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
        1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
        2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
        3624, 3752, 3824,
    ],
    dtype=torch.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the base Cross-Stitch 5Hz model on the send trace RB/SINR and "
            "export a MATLAB-readable predicted datarate sequence."
        )
    )
    parser.add_argument(
        "--traffic_dataset_dir",
        type=Path,
        default=Path("/4T/xty/traffic_hdf5_datasets/http"),
        help="Directory containing the RB/SINR/Throughput PKL files used by send/final.m.",
    )
    parser.add_argument(
        "--stats_dataset_dir",
        type=Path,
        default=Path("/4T/xty/new_mcs_dataset/single_user_pf_5Hz"),
        help="Directory used to compute the base model's SINR normalization stats.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=ROOT / "experiments_all" / "single_user_pf_5Hz" / "best_mtl_model_decoder.pth",
    )
    parser.add_argument(
        "--output_mat",
        type=Path,
        default=Path("/4T/xty/send/cross_stitch_base_predicted_datarate.mat"),
    )
    parser.add_argument("--skip_frames", type=int, default=10)
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--pre_window", type=int, default=100)
    parser.add_argument("--rb_hidden_dim", type=int, default=256)
    parser.add_argument("--sinr_hidden_dim", type=int, default=64)
    parser.add_argument("--rb_layers", type=int, default=3)
    parser.add_argument("--sinr_layers", type=int, default=1)
    parser.add_argument("--cross_stitch_mode", type=str, default="learn")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--preserve_zero_throughput_mask",
        action="store_true",
        default=False,
        help=(
            "When exporting effective datarate, force predicted throughput to zero on frames "
            "where the dataset throughput label is zero. This avoids the compare_v2 behavior "
            "that reconstructs all throughput only from RB/SINR."
        ),
    )
    return parser.parse_args()


def _pick_series(payload, names):
    if not isinstance(payload, dict):
        raise TypeError(f"Expected pickle payload to be dict, got {type(payload)!r}")
    for name in names:
        if name in payload:
            return np.asarray(payload[name], dtype=np.float32).reshape(-1)
    raise KeyError(f"None of {names} found in payload keys {list(payload.keys())}")


def load_trace_pair(dataset_dir: Path):
    train_payload = pd.read_pickle(dataset_dir / "train_9000_HDF5.pkl")
    test_payload = pd.read_pickle(dataset_dir / "test_1000_HDF5.pkl")

    train_rb = _pick_series(train_payload, ["RB", "rb", "Rb", "rB"])
    train_sinr = _pick_series(train_payload, ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR"])
    train_thr = _pick_series(train_payload, ["Throughput", "throughput"])

    test_rb = _pick_series(test_payload, ["RB", "rb", "Rb", "rB"])
    test_sinr = _pick_series(test_payload, ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR"])
    test_thr = _pick_series(test_payload, ["Throughput", "throughput"])

    return (
        np.concatenate([train_rb, test_rb], axis=0),
        np.concatenate([train_sinr, test_sinr], axis=0),
        np.concatenate([train_thr, test_thr], axis=0),
        train_rb.shape[0],
    )


def compute_stats(dataset_dir: Path, skip_frames: int):
    train_payload = pd.read_pickle(dataset_dir / "train_9000_HDF5.pkl")
    sinr = _pick_series(train_payload, ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR"])
    if skip_frames > 0:
        sinr = sinr[skip_frames:]
    return float(sinr.mean()), float(sinr.std())


def calculate_throughput_batch_fast(rb_pred: torch.Tensor, sinr_pred: torch.Tensor) -> torch.Tensor:
    device = rb_pred.device
    rb = rb_pred.to(dtype=torch.float32)
    sinr = sinr_pred.to(dtype=torch.float32)

    orig_shape = rb.shape
    rb = rb.reshape(-1)
    sinr = sinr.reshape(-1)

    table = _MCS_TABLE.to(device)
    sinr_edges = table[:, 3]
    idx = torch.bucketize(sinr, sinr_edges, right=True) - 1
    idx = idx.clamp(min=0, max=table.size(0) - 1)

    code_rate = table[:, 2][idx]
    qam_vals = table[:, 1][idx]
    mod_order = torch.log2(qam_vals).to(torch.float32)

    c2 = torch.tensor(2.0, device=device)
    c8 = torch.tensor(8.0, device=device)
    c24 = torch.tensor(24.0, device=device)
    c3840 = torch.tensor(3840.0, device=device)
    c3816 = torch.tensor(3816.0, device=device)
    c8424 = torch.tensor(8424.0, device=device)

    n0 = mod_order * rb * 136.0 * code_rate
    mask_hi = n0 > 3824.0
    tbs = torch.empty_like(n0, dtype=torch.float32)

    if mask_hi.any():
        hi_idx = torch.nonzero(mask_hi, as_tuple=False).squeeze(1)
        n0_hi = n0[hi_idx]
        code_rate_hi = code_rate[hi_idx]

        n_hi = torch.floor(torch.log2(n0_hi - c24)) - 5.0
        two_n_hi = torch.pow(c2, n_hi)
        n_hi_val = torch.maximum(c3840, torch.round((n0_hi - c24) / two_n_hi) * two_n_hi)

        mask_lowrate = code_rate_hi <= 0.25
        if mask_lowrate.any():
            sub = hi_idx[mask_lowrate]
            n_hi_lr = n_hi_val[mask_lowrate]
            c = torch.ceil((n_hi_lr + c24) / c3816)
            tbs[sub] = c8 * c * torch.ceil((n_hi_lr + c24) / c8 / c) - c24

        mask_large = (~mask_lowrate) & (n_hi_val > c8424)
        if mask_large.any():
            sub = hi_idx[mask_large]
            n_hi_lg = n_hi_val[mask_large]
            c = torch.ceil((n_hi_lg + c24) / c8424)
            tbs[sub] = c8 * c * torch.ceil((n_hi_lg + c24) / c8 / c) - c24

        mask_middle = (~mask_lowrate) & (~mask_large)
        if mask_middle.any():
            sub = hi_idx[mask_middle]
            n_hi_md = n_hi_val[mask_middle]
            tbs[sub] = c8 * torch.ceil((n_hi_md + c24) / c8) - c24

    mask_low = ~mask_hi
    if mask_low.any():
        low_idx = torch.nonzero(mask_low, as_tuple=False).squeeze(1)
        n0_low = n0[low_idx]
        n_low = torch.floor(torch.log2(n0_low.clamp_min(1e-6))) - 6.0
        n_low = torch.maximum(torch.tensor(3.0, device=device), n_low)
        two_n_low = torch.pow(c2, n_low)
        n_low_val = torch.maximum(c24, torch.floor(n0_low / two_n_low) * two_n_low)

        tbs_tab = _TBS_TABLE.to(device)
        jdx = torch.bucketize(n_low_val, tbs_tab, right=False).clamp(max=tbs_tab.numel() - 1)
        tbs[low_idx] = tbs_tab[jdx].to(torch.float32)

    throughput = tbs * 1e-6 / (5e-4)
    return throughput.reshape(orig_shape)


def predict_avg_rate(args, rb_all, sinr_all, throughput_all, sinr_mean, sinr_std):
    device = torch.device(args.device)
    model = build_model(args, device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total_len = rb_all.shape[0]
    num_samples = total_len - args.hist_window - args.pre_window + 1
    if num_samples <= 0:
        raise ValueError("Not enough samples to build sliding windows.")

    rb_all = rb_all.astype(np.float32, copy=False)
    sinr_norm = ((sinr_all - sinr_mean) / (sinr_std + 1e-8)).astype(np.float32, copy=False)
    pred_avg_rate = np.full(total_len, np.nan, dtype=np.float32)
    pred_avg_rate_raw = np.full(total_len, np.nan, dtype=np.float32)

    with torch.no_grad():
        for begin in range(0, num_samples, args.batch_size):
            end = min(begin + args.batch_size, num_samples)
            starts = np.arange(begin, end, dtype=np.int64)

            rb_batch = np.stack([rb_all[s : s + args.hist_window] for s in starts], axis=0)
            sinr_batch = np.stack([sinr_norm[s : s + args.hist_window] for s in starts], axis=0)

            rb_seq = torch.from_numpy(rb_batch[:, :, None]).to(device)
            sinr_seq = torch.from_numpy(sinr_batch[:, :, None]).to(device)

            rb_pred, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)
            rb_pred = torch.argmax(rb_pred, dim=2) + 1
            sinr_pred_denorm = (sinr_pred * sinr_std + sinr_mean).squeeze(-1)
            thr_pred = calculate_throughput_batch_fast(rb_pred, sinr_pred_denorm)
            avg_rate_raw = thr_pred.mean(dim=1).cpu().numpy().astype(np.float32, copy=False)

            if args.preserve_zero_throughput_mask:
                zero_mask_batch = np.stack(
                    [(throughput_all[s + args.hist_window : s + args.hist_window + args.pre_window] <= 0).astype(np.float32)
                     for s in starts],
                    axis=0,
                )
                zero_mask_batch_t = torch.from_numpy(zero_mask_batch).to(device)
                thr_pred = thr_pred * (1.0 - zero_mask_batch_t)

            avg_rate = thr_pred.mean(dim=1).cpu().numpy().astype(np.float32, copy=False)

            label_starts = starts + args.hist_window
            pred_avg_rate[label_starts] = avg_rate
            pred_avg_rate_raw[label_starts] = avg_rate_raw

    return pred_avg_rate, pred_avg_rate_raw


def main():
    args = parse_args()
    sinr_mean, sinr_std = compute_stats(args.stats_dataset_dir, args.skip_frames)
    rb_all, sinr_all, throughput_all, train_len = load_trace_pair(args.traffic_dataset_dir)
    pred_avg_rate, pred_avg_rate_raw = predict_avg_rate(
        args,
        rb_all,
        sinr_all,
        throughput_all,
        sinr_mean,
        sinr_std,
    )

    valid_mask = np.isfinite(pred_avg_rate)
    zero_frame_mask = throughput_all <= 0
    actual_avg_rate = np.full_like(pred_avg_rate, np.nan, dtype=np.float32)
    for idx in range(args.hist_window, rb_all.shape[0] - args.pre_window + 1):
        actual_avg_rate[idx] = float(np.mean(throughput_all[idx : idx + args.pre_window]))

    args.output_mat.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        args.output_mat,
        {
            "predAvgRateMbps": pred_avg_rate,
            "predAvgRateMbpsRaw": pred_avg_rate_raw,
            "actualAvgRateMbps": actual_avg_rate,
            "throughputTraceMbps": throughput_all.astype(np.float32, copy=False),
            "validMask": valid_mask.astype(np.uint8),
            "zeroThroughputMask": zero_frame_mask.astype(np.uint8),
            "histWindow": np.array([[args.hist_window]], dtype=np.int32),
            "preWindow": np.array([[args.pre_window]], dtype=np.int32),
            "trainLength": np.array([[train_len]], dtype=np.int32),
            "testLength": np.array([[rb_all.shape[0] - train_len]], dtype=np.int32),
            "sinrMean": np.array([[sinr_mean]], dtype=np.float32),
            "sinrStd": np.array([[sinr_std]], dtype=np.float32),
        },
    )

    mae = float(np.nanmean(np.abs(pred_avg_rate - actual_avg_rate)))
    mae_raw = float(np.nanmean(np.abs(pred_avg_rate_raw - actual_avg_rate)))
    print(f"Saved predicted datarate to {args.output_mat}")
    print(f"Total length: {rb_all.shape[0]}")
    print(f"Valid estimate count: {int(valid_mask.sum())}")
    print(f"Zero-throughput frames in dataset: {int(zero_frame_mask.sum())}")
    print(f"Avg-rate MAE on valid positions (zero-aware): {mae:.6f} Mbps")
    print(f"Avg-rate MAE on valid positions (raw v2-style): {mae_raw:.6f} Mbps")


if __name__ == "__main__":
    main()
