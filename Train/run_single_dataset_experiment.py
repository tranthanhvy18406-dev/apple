import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtl_dataset import MTL_Dataset
from mtl_model import CrossStitch_MTL_Model


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
    parser = argparse.ArgumentParser(description="Train and evaluate Cross-Stitch MTL on one Doppler dataset.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--init_model_path",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize the model before training.",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--pre_window", type=int, default=100)
    parser.add_argument("--skip_frames", type=int, default=10)
    parser.add_argument("--rb_hidden_dim", type=int, default=256)
    parser.add_argument("--sinr_hidden_dim", type=int, default=64)
    parser.add_argument("--rb_layers", type=int, default=3)
    parser.add_argument("--sinr_layers", type=int, default=1)
    parser.add_argument("--cross_stitch_mode", type=str, default="learn", choices=["learn", "identity", "zeros"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sinr_loss", type=str, default="mse", choices=["mse", "mae", "huber"])
    parser.add_argument("--huber_beta", type=float, default=0.05)
    parser.add_argument("--gamma_rb_dist", type=float, default=0.1)
    parser.add_argument("--pareto_eps_acc", type=float, default=0.0)
    parser.add_argument("--pareto_eps_mae", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="Early-stop if no Pareto improvement for this many epochs. 0 disables.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def val_to_index(val: torch.Tensor) -> torch.Tensor:
    return (val - 1).long()


def build_sinr_criterion(kind: str, beta: float):
    if kind == "mse":
        return nn.MSELoss()
    if kind == "mae":
        return nn.L1Loss()
    if kind == "huber":
        return nn.SmoothL1Loss(beta=beta)
    raise ValueError(kind)


def pareto_better(curr_acc, curr_mae, best_acc, best_mae, eps_acc=0.0, eps_mae=0.0):
    better_on_acc = (curr_acc > best_acc + eps_acc) and (curr_mae <= best_mae + eps_mae)
    better_on_mae = (curr_mae < best_mae - eps_mae) and (curr_acc >= best_acc - eps_acc)
    return better_on_acc or better_on_mae


def flatten_series(obj, keys):
    data = obj
    if isinstance(obj, dict):
        data = None
        for key in keys:
            if key in obj:
                data = obj[key]
                break
        if data is None:
            raise KeyError(f"Missing keys {keys} in pickle object.")
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


@torch.no_grad()
def calculate_throughput_from_predictions_parallel(
    rb_pred: torch.Tensor,
    sinr_pred: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    rb = rb_pred.to(device=device, dtype=torch.float32)
    sinr = sinr_pred.to(device=device, dtype=torch.float32)
    table = _MCS_TABLE.to(device)

    sinr_edges = table[:, 3].contiguous()
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
        n_hi_rounded = torch.maximum(c3840, torch.round((n0_hi - c24) / two_n_hi) * two_n_hi)

        mask_lowrate = code_rate_hi <= 0.25
        if mask_lowrate.any():
            sub = hi_idx[mask_lowrate]
            n_hi_lr = n_hi_rounded[mask_lowrate]
            c = torch.ceil((n_hi_lr + c24) / c3816)
            tbs[sub] = c8 * c * torch.ceil((n_hi_lr + c24) / c8 / c) - c24

        mask_large = (~mask_lowrate) & (n_hi_rounded > c8424)
        if mask_large.any():
            sub = hi_idx[mask_large]
            n_hi_lg = n_hi_rounded[mask_large]
            c = torch.ceil((n_hi_lg + c24) / c8424)
            tbs[sub] = c8 * c * torch.ceil((n_hi_lg + c24) / c8 / c) - c24

        mask_middle = (~mask_lowrate) & (~mask_large)
        if mask_middle.any():
            sub = hi_idx[mask_middle]
            n_hi_md = n_hi_rounded[mask_middle]
            tbs[sub] = c8 * torch.ceil((n_hi_md + c24) / c8) - c24

    mask_low = ~mask_hi
    if mask_low.any():
        low_idx = torch.nonzero(mask_low, as_tuple=False).squeeze(1)
        n0_low = n0[mask_low]

        n_low = torch.floor(torch.log2(n0_low.clamp_min(1e-6))) - 6.0
        n_low = torch.maximum(torch.tensor(3.0, device=device), n_low)
        two_n_low = torch.pow(c2, n_low)
        n_low_rounded = torch.maximum(c24, torch.floor(n0_low / two_n_low) * two_n_low)

        tbs_tab = _TBS_TABLE.to(device)
        jdx = torch.bucketize(n_low_rounded, tbs_tab, right=False).clamp(max=tbs_tab.numel() - 1)
        tbs[low_idx] = tbs_tab[jdx].to(torch.float32)

    return tbs * 1e-6 / (5e-4)


def build_model(args, device: torch.device):
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


def evaluate_model(model, loader, sinr_mean, sinr_std, device, throughput_labels=None, hist_window=16):
    model.eval()
    total_rb_correct = 0
    total_rb_abs_err = 0.0
    total_sinr_mae_denorm = 0.0
    total_sinr_mse_denorm = 0.0
    total_points = 0
    total_throughput_abs_err = 0.0
    total_throughput_points = 0

    with torch.no_grad():
        for batch_idx, (rb_seq, sinr_seq, rb_label, sinr_label) in enumerate(loader):
            rb_seq = rb_seq.to(device)
            sinr_seq = sinr_seq.to(device)
            rb_label = rb_label.to(device)
            sinr_label = sinr_label.to(device)

            rb_pred, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)
            rb_class = torch.argmax(rb_pred, dim=2) + 1

            batch_points = rb_label.numel()
            total_points += batch_points
            total_rb_correct += (rb_class == rb_label.squeeze(-1)).sum().item()
            total_rb_abs_err += torch.abs(rb_class - rb_label.squeeze(-1)).sum().item()

            sinr_pred_denorm = sinr_pred * sinr_std + sinr_mean
            sinr_label_denorm = sinr_label * sinr_std + sinr_mean
            total_sinr_mae_denorm += torch.abs(sinr_pred_denorm - sinr_label_denorm).sum().item()
            total_sinr_mse_denorm += ((sinr_pred_denorm - sinr_label_denorm) ** 2).sum().item()

            if throughput_labels is not None:
                batch_start = batch_idx * loader.batch_size
                batch_size = rb_class.shape[0]
                for sample_idx in range(batch_size):
                    dataset_idx = batch_start + sample_idx
                    label_start = dataset_idx + hist_window
                    label_end = label_start + rb_class.shape[1]
                    throughput_true = throughput_labels[label_start:label_end]
                    if len(throughput_true) != rb_class.shape[1]:
                        continue
                    throughput_true = torch.as_tensor(throughput_true, device=device, dtype=torch.float32)
                    throughput_pred = calculate_throughput_from_predictions_parallel(
                        rb_class[sample_idx],
                        sinr_pred_denorm[sample_idx].squeeze(-1),
                        device,
                    )
                    total_throughput_abs_err += torch.abs(throughput_pred - throughput_true).sum().item()
                    total_throughput_points += throughput_pred.numel()

    metrics = {
        "rb_acc": total_rb_correct / total_points,
        "rb_mae": total_rb_abs_err / total_points,
        "sinr_mae": total_sinr_mae_denorm / total_points,
        "sinr_mse": total_sinr_mse_denorm / total_points,
    }
    if throughput_labels is not None:
        metrics["throughput_mae"] = total_throughput_abs_err / total_throughput_points
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_root / dataset_dir.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "best_mtl_model_decoder.pth"
    history_path = output_dir / "history.json"

    if metrics_path.exists() and not args.force:
        with open(metrics_path, "r", encoding="utf-8") as fp:
            metrics = json.load(fp)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_path = dataset_dir / "train_9000_HDF5.pkl"
    test_path = dataset_dir / "test_1000_HDF5.pkl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing pickle dataset in {dataset_dir}")

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    throughput_labels = flatten_series(pd.read_pickle(test_path), ["Throughput", "throughput"])

    model = build_model(args, device)
    if args.init_model_path is not None:
        init_state = torch.load(args.init_model_path, map_location=device, weights_only=True)
        model.load_state_dict(init_state)
        print(f"Loaded initialization checkpoint from {args.init_model_path}")
    criterion_rb_ce = nn.CrossEntropyLoss()
    criterion_sinr = build_sinr_criterion(args.sinr_loss, args.huber_beta)
    log_var_rb = torch.zeros((1,), requires_grad=True, device=device)
    log_var_sinr = torch.zeros((1,), requires_grad=True, device=device)
    params_to_optimize = list(model.parameters()) + [log_var_rb, log_var_sinr]
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_rb_acc = 0.0
    best_sinr_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    classes = torch.arange(1, 107, device=device).float()
    tf_ratio_start, tf_ratio_end, tf_decay_epochs = 0.7, 0.2, 50
    history = []

    print(f"Starting training for {dataset_dir.name} on {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tf_ratio = tf_ratio_start - (tf_ratio_start - tf_ratio_end) * min(1.0, epoch / tf_decay_epochs)
        pbar = tqdm(train_loader, desc=f"{dataset_dir.name} epoch {epoch}/{args.epochs} tf={tf_ratio:.2f}", ncols=120)

        for rb_seq, sinr_seq, rb_label, sinr_label in pbar:
            rb_seq = rb_seq.to(device)
            sinr_seq = sinr_seq.to(device)
            rb_label = rb_label.to(device)
            sinr_label = sinr_label.to(device)
            optimizer.zero_grad()

            teacher_rb_onehot = F.one_hot(val_to_index(rb_label.squeeze(-1)), num_classes=model.rb_output_size).float()
            rb_pred, sinr_pred = model(
                rb_seq,
                sinr_seq,
                teacher_rb=teacher_rb_onehot,
                teacher_sinr=sinr_label,
                tf_ratio=tf_ratio,
            )

            loss_rb = criterion_rb_ce(rb_pred.reshape(-1, 106), val_to_index(rb_label.reshape(-1)))
            loss_sinr = criterion_sinr(sinr_pred, sinr_label)

            log_var_rb.data.clamp_(-5.0, 5.0)
            log_var_sinr.data.clamp_(-5.0, 5.0)
            precision_rb = torch.exp(-log_var_rb)
            precision_sinr = torch.exp(-log_var_sinr)
            loss_rb_weighted = precision_rb * loss_rb + 0.5 * log_var_rb
            loss_sinr_weighted = 0.5 * (precision_sinr * loss_sinr + log_var_sinr)
            loss = loss_rb_weighted + loss_sinr_weighted

            if args.gamma_rb_dist > 0.0:
                probs = F.softmax(rb_pred, dim=2)
                rb_expect = (probs * classes).sum(dim=2)
                rb_true = rb_label.squeeze(-1).float()
                loss_rb_dist = F.l1_loss(rb_expect, rb_true)
                loss = loss + args.gamma_rb_dist * loss_rb_dist

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        eval_metrics = evaluate_model(model, eval_loader, sinr_mean, sinr_std, device)
        history_item = {"epoch": epoch, **{k: float(v) for k, v in eval_metrics.items()}}
        history.append(history_item)
        print(
            f"{dataset_dir.name} epoch {epoch}: "
            f"RB MAE={eval_metrics['rb_mae']:.4f} | "
            f"SINR MAE={eval_metrics['sinr_mae']:.4f} | "
            f"RB Acc={eval_metrics['rb_acc'] * 100:.2f}%"
        )

        if pareto_better(
            eval_metrics["rb_acc"],
            eval_metrics["sinr_mae"],
            best_rb_acc,
            best_sinr_mae,
            eps_acc=args.pareto_eps_acc,
            eps_mae=args.pareto_eps_mae,
        ):
            best_rb_acc = max(best_rb_acc, eval_metrics["rb_acc"])
            best_sinr_mae = min(best_sinr_mae, eval_metrics["sinr_mae"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"{dataset_dir.name} saved new best checkpoint at epoch {epoch}")
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"{dataset_dir.name} early-stopped after {stale_epochs} stale epochs")
            break

    if not model_path.exists():
        raise RuntimeError(f"No checkpoint was saved for {dataset_dir.name}")

    best_model = build_model(args, device)
    best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    final_metrics = evaluate_model(
        best_model,
        eval_loader,
        sinr_mean,
        sinr_std,
        device,
        throughput_labels=throughput_labels,
        hist_window=args.hist_window,
    )

    doppler_hz = None
    if "_pf_" in dataset_dir.name and dataset_dir.name.endswith("Hz"):
        doppler_hz = int(dataset_dir.name.split("_pf_")[1].replace("Hz", ""))
    result = {
        "dataset": dataset_dir.name,
        "doppler_hz": doppler_hz,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "rb_mae": float(final_metrics["rb_mae"]),
        "sinr_mae": float(final_metrics["sinr_mae"]),
        "throughput_mae": float(final_metrics["throughput_mae"]),
        "rb_acc": float(final_metrics["rb_acc"]),
        "sinr_mse": float(final_metrics["sinr_mse"]),
        "model_path": str(model_path),
    }

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
