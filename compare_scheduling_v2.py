import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "Train"))
from mtl_dataset import MTL_Dataset
from mtl_model import CrossStitch_MTL_Model


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Stitch on PF 5Hz with compare_scheduling_v2 metrics")
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/4T/xty/new_mcs_dataset/single_user_pf_5Hz"),
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=root / "experiments_all" / "single_user_pf_5Hz" / "best_mtl_model_decoder.pth",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=root / "compare_scheduling_v2_5Hz_pf.json",
    )
    parser.add_argument(
        "--stats_dir",
        type=Path,
        default=None,
        help="Directory used to compute SINR normalization stats. Defaults to dataset_dir.",
    )
    parser.add_argument("--skip_frames", type=int, default=10)
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--pre_window", type=int, default=100)
    parser.add_argument("--rb_hidden_dim", type=int, default=256)
    parser.add_argument("--sinr_hidden_dim", type=int, default=64)
    parser.add_argument("--rb_layers", type=int, default=3)
    parser.add_argument("--sinr_layers", type=int, default=1)
    parser.add_argument("--cross_stitch_mode", type=str, default="learn")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def calculate_metrics(pred, label):
    pred = pred.float()
    label = label.float()
    return {
        "mae": torch.abs(pred - label).mean().item(),
        "mse": ((pred - label) ** 2).mean().item(),
    }


def mcs_table(sinr: torch.Tensor):
    table = torch.tensor(
        [
            [1, 4, 0.10, -6.4131],
            [2, 4, 0.14, -4.7292],
            [3, 4, 0.18, -3.6312],
            [4, 4, 0.23, -2.9651],
            [5, 4, 0.26, -2.0542],
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
            [18, 16, 0.57, 7.2301],
            [17, 16, 0.59, 7.9862],
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
        device=sinr.device,
    )
    sinr_thresh = table[:, 3]
    sinr = sinr.view(1)
    if sinr < sinr_thresh.min():
        i_mcs = int(table[0, 0].item())
    else:
        idx = (sinr >= sinr_thresh).nonzero()[-1].item()
        i_mcs = int(table[idx, 0].item())
    row = table[table[:, 0] == i_mcs][0]
    code_rate = row[2].item()
    mod_order = int(torch.log2(torch.tensor(int(row[1].item()), device=sinr.device)).item())
    return code_rate, mod_order


def lookup_tbs(n_info_prime: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
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
        device=n_info_prime.device,
    )
    idx = (table >= n_info_prime).nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        return table[idx[0]].to(dtype=torch.int32)
    return table[-1].to(dtype=torch.int32)


def calculate_single_throughput(rb_value, sinr_value, device):
    code_rate, mod_order = mcs_table(sinr_value)
    n_info_0 = torch.tensor(mod_order * rb_value * 136 * code_rate, device=device)
    if n_info_0 > 3824:
        n = torch.floor(torch.log2(n_info_0 - 24)) - 5
        n_info = torch.maximum(
            torch.tensor(3840.0, device=device),
            torch.round((n_info_0 - 24) / (2 ** n)) * (2 ** n),
        )
        if code_rate <= 0.25:
            c = torch.ceil((n_info + 24) / 3816)
            tbs = 8 * c * torch.ceil((n_info + 24) / 8 / c) - 24
        elif n_info > 8424:
            c = torch.ceil((n_info + 24) / 8424)
            tbs = 8 * c * torch.ceil((n_info + 24) / 8 / c) - 24
        else:
            tbs = 8 * torch.ceil((n_info + 24) / 8) - 24
    else:
        n = torch.floor(torch.log2(n_info_0)) - 6
        n = torch.maximum(torch.tensor(3.0, device=device), n)
        n_info = torch.maximum(
            torch.tensor(24.0, device=device),
            torch.floor(n_info_0 / (2 ** n)) * (2 ** n),
        )
        tbs = lookup_tbs(n_info)
    return tbs * 1e-6 / (5e-4)


def calculate_throughput_batch(rb_preds, sinr_preds, device):
    if rb_preds.dim() == 1:
        rb_preds = rb_preds.unsqueeze(0)
        sinr_preds = sinr_preds.unsqueeze(0)
    batch_size, pre_window = rb_preds.shape
    throughput = torch.zeros(batch_size, pre_window, device=device)
    for b in range(batch_size):
        for t in range(pre_window):
            throughput[b, t] = calculate_single_throughput(rb_preds[b, t], sinr_preds[b, t], device)
    return throughput


def build_model(args, device):
    return CrossStitch_MTL_Model(
        rb_input_size=1,
        sinr_input_size=1,
        rb_hidden_size=args.rb_hidden_dim,
        sinr_hidden_size=args.sinr_hidden_dim,
        rb_num_layers=args.rb_layers,
        sinr_num_layers=args.sinr_layers,
        rb_output_size=106,
        sinr_output_size=1,
        pre_window=args.pre_window,
        device=device,
        cross_stitch_mode=args.cross_stitch_mode,
    ).to(device)


def main():
    args = get_args()
    device = torch.device(args.device)

    stats_dir = args.stats_dir if args.stats_dir is not None else args.dataset_dir

    train_path = stats_dir / "train_9000_HDF5.pkl"
    test_path = args.dataset_dir / "test_1000_HDF5.pkl"

    train_dataset = MTL_Dataset(
        str(train_path),
        str(train_path),
        obs_window=args.hist_window,
        pre_window=args.pre_window,
        skip_initial_frames=args.skip_frames,
    )
    sinr_mean = float(train_dataset.sinr_mean)
    sinr_std = float(train_dataset.sinr_std)
    test_dataset = MTL_Dataset(
        str(test_path),
        str(test_path),
        obs_window=args.hist_window,
        pre_window=args.pre_window,
        sinr_mean=sinr_mean,
        sinr_std=sinr_std,
    )
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    all_rb_pred = []
    all_rb_label = []
    all_sinr_pred = []
    all_sinr_label = []
    all_thr_pred = []
    all_thr_label = []

    with torch.no_grad():
        for rb_seq, sinr_seq, rb_label, sinr_label in loader:
            rb_seq = rb_seq.to(device)
            sinr_seq = sinr_seq.to(device)
            rb_label = rb_label.to(device).squeeze(-1)
            sinr_label = sinr_label.to(device)

            rb_pred, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)
            rb_pred = torch.argmax(rb_pred, dim=2) + 1

            sinr_pred_denorm = (sinr_pred * sinr_std + sinr_mean).squeeze(-1)
            sinr_label_denorm = (sinr_label * sinr_std + sinr_mean).squeeze(-1)

            thr_pred = calculate_throughput_batch(rb_pred, sinr_pred_denorm, device)
            thr_label = calculate_throughput_batch(rb_label, sinr_label_denorm, device)

            all_rb_pred.append(rb_pred.cpu())
            all_rb_label.append(rb_label.cpu())
            all_sinr_pred.append(sinr_pred_denorm.cpu())
            all_sinr_label.append(sinr_label_denorm.cpu())
            all_thr_pred.append(thr_pred.cpu())
            all_thr_label.append(thr_label.cpu())

    all_rb_pred = torch.cat(all_rb_pred).reshape(-1)
    all_rb_label = torch.cat(all_rb_label).reshape(-1)
    all_sinr_pred = torch.cat(all_sinr_pred).reshape(-1)
    all_sinr_label = torch.cat(all_sinr_label).reshape(-1)
    all_thr_pred = torch.cat(all_thr_pred).reshape(-1)
    all_thr_label = torch.cat(all_thr_label).reshape(-1)

    result = {
        "model": "Cross-Stitch",
        "dataset": args.dataset_dir.name,
        "metrics": {
            "rb": calculate_metrics(all_rb_pred, all_rb_label),
            "sinr": calculate_metrics(all_sinr_pred, all_sinr_label),
            "throughput": calculate_metrics(all_thr_pred, all_thr_label),
        },
        "count": int(all_rb_pred.numel()),
    }

    args.output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
