# test_mtl_throughput.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt

from mtl_dataset import MTL_Dataset
from mtl_model import CrossStitch_MTL_Model

def get_args():
    """Parses command-line arguments for testing the MTL model with throughput calculation."""
    parser = argparse.ArgumentParser(description="Test Cross-Stitch MTL Model with Throughput Calculation")
    
    parser.add_argument('--train_rb_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/train_dataset_10000.pkl')
    parser.add_argument('--train_sinr_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/train_dataset_10000.pkl')
    parser.add_argument('--rb_test_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/test_dataset_10000.pkl')
    parser.add_argument('--sinr_test_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/test_dataset_10000.pkl')
    parser.add_argument('--throughput_test_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/test_dataset_10000.pkl',
                       help='Path to throughput test data')

    parser.add_argument('--hist_window', type=int, default=16, help='Observation history window size')
    parser.add_argument('--pre_window', type=int, default=100, help='Prediction window size (changed to 100 for throughput calculation)')
    parser.add_argument('--rb_hidden_dim', type=int, default=256)
    parser.add_argument('--sinr_hidden_dim', type=int, default=64)
    parser.add_argument('--rb_layers', type=int, default=3)
    parser.add_argument('--sinr_layers', type=int, default=1)
    parser.add_argument('--cross_stitch_mode', type=str, default='learn')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing (set to 1 for frame-by-frame processing)')
    parser.add_argument('--model_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Task3_VoIP/Cross_Stitch/Train/Model/best_mtl_model_decoder.pth', help='Path to the trained model weights')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to numpy files')
    
    return parser.parse_args()

_MCS_TABLE = torch.tensor([
    [1,  4,  0.10, -6.4131],
    [2,  4,  0.13, -4.7292],
    [3,  4,  0.17, -3.6312],
    [4,  4,  0.22, -2.9651],
    [5,  4,  0.25, -2.0542],
    [6,  4,  0.34, -1.2451],
    [7,  4,  0.40, -0.3223],
    [8,  4,  0.45,  0.8364],
    [9,  4,  0.52,  1.5261],
    [10, 4,  0.59,  2.5709],
    [11, 16, 0.31,  3.1961],
    [12, 16, 0.32,  3.7165],
    [13, 16, 0.37,  4.9642],
    [14, 16, 0.45,  5.4653],
    [15, 16, 0.47,  6.9965],
    [16, 16, 0.54,  7.1562],
    [17, 16, 0.57,  7.2301],
    [18, 16, 0.59,  7.9862],
    [19, 64, 0.35,  8.0651],
    [20, 64, 0.38,  8.6985],
    [21, 64, 0.41,  9.0224],
    [22, 64, 0.43,  9.3017],
    [23, 64, 0.45,  9.9628],
    [24, 64, 0.47, 10.3957],
    [25, 64, 0.49, 10.7214],
    [26, 64, 0.55, 12.0541],
    [27, 64, 0.61, 12.8769],
    [28, 64, 0.63, 13.5547],
    [29, 64, 0.65, 14.6139],
], dtype=torch.float32)

_TBS_TABLE = torch.tensor([
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
    104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
    184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 408, 432, 456, 480, 504, 528,
    552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
    888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
    1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
    1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
    2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
    3624, 3752, 3824
], dtype=torch.float32)
# ================================================================


@torch.no_grad()
def calculate_throughput_from_predictions_parallel(
    rb_pred: torch.Tensor,       # [T], 已转换到 1..106
    sinr_pred: torch.Tensor,     # [T], 已反归一化（dB）
    device: torch.device
) -> torch.Tensor:

    # --- 准备输入张量 ---
    rb   = rb_pred.to(device=device, dtype=torch.float32)      # [T]
    sinr = sinr_pred.to(device=device, dtype=torch.float32)    # [T]
    table = _MCS_TABLE.to(device)

    # 1) 查表获取码率，调制阶数等
    sinr_edges = table[:, 3]                                   # 递增阈值
    idx = torch.bucketize(sinr, sinr_edges, right=True) - 1    # [-1..K-1]
    idx = idx.clamp(min=0, max=table.size(0)-1)                # [0..K-1]

    code_rate = table[:, 2][idx]                               # [T]
    qam_vals  = table[:, 1][idx]                               # [T] (4/16/64)
    mod_order = torch.log2(qam_vals).to(torch.float32)         # [T] (2/4/6)

    #  2) 常数张量化
    c2    = torch.tensor(2.0,    device=device)
    c8    = torch.tensor(8.0,    device=device)
    c24   = torch.tensor(24.0,   device=device)
    c3840 = torch.tensor(3840.0, device=device)
    c3816 = torch.tensor(3816.0, device=device)
    c8424 = torch.tensor(8424.0, device=device)

    N0 = mod_order * rb * 136.0 * code_rate
    mask_hi = N0 > 3824.0

    # 结果占位
    TBS = torch.empty_like(N0, dtype=torch.float32)
    if mask_hi.any():
        hi_idx = torch.nonzero(mask_hi, as_tuple=False).squeeze(1)
        N0_hi = N0[hi_idx]
        code_rate_hi = code_rate[hi_idx]

        n_hi = torch.floor(torch.log2(N0_hi - c24)) - 5.0
        two_n_hi = torch.pow(c2, n_hi)
        N_hi = torch.maximum(c3840, torch.round((N0_hi - c24)/two_n_hi) * two_n_hi)

        # ===== 组1：hi & 低码率 =====
        mask_lowrate = code_rate_hi <= 0.25
        if mask_lowrate.any():
            sub = hi_idx[mask_lowrate]
            N_hi_lr = N_hi[mask_lowrate]
            C = torch.ceil((N_hi_lr + c24) / c3816)
            TBS_lowrate = c8 * C * torch.ceil((N_hi_lr + c24) / c8 / C) - c24
            TBS[sub] = TBS_lowrate

        # ===== 组2：hi & 非低码率 & N_hi > 8424（大块）=====
        mask_large = (~mask_lowrate) & (N_hi > c8424)
        if mask_large.any():
            sub = hi_idx[mask_large]
            N_hi_lg = N_hi[mask_large]
            C = torch.ceil((N_hi_lg + c24) / c8424)
            TBS_large = c8 * C * torch.ceil((N_hi_lg + c24) / c8 / C) - c24
            TBS[sub] = TBS_large

        # ===== 组3：hi & 非低码率 & 非大块=====
        mask_middle = (~mask_lowrate) & (~mask_large)
        if mask_middle.any():
            sub = hi_idx[mask_middle]
            N_hi_md = N_hi[mask_middle]
            TBS_middle = c8 * torch.ceil((N_hi_md + c24) / c8) - c24
            TBS[sub] = TBS_middle

    # ===== 组4：lo（N0 <= 3824）=====
    mask_low = ~mask_hi
    if mask_low.any():
        low_idx = torch.nonzero(mask_low, as_tuple=False).squeeze(1)
        N0_low = N0[mask_low]

        n_low = torch.floor(torch.log2(N0_low.clamp_min(1e-6))) - 6.0
        n_low = torch.maximum(torch.tensor(3.0, device=device), n_low)
        two_n_low = torch.pow(c2, n_low)
        N_low = torch.maximum(c24, torch.floor(N0_low / two_n_low) * two_n_low)

        tbs_tab = _TBS_TABLE.to(device)
        jdx = torch.bucketize(N_low, tbs_tab, right=False).clamp(max=tbs_tab.numel()-1)
        TBS_low = tbs_tab[jdx].to(torch.float32)
        TBS[low_idx] = TBS_low

    # 计算 Throughput，TBS / 0.5ms， 以 Mbps 为单位
    throughput = TBS * 1e-6 / (5e-4)
    return throughput

def main(args):
    DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}\nArguments: {args}\n")

    # 构建SINR和RB数据集
    print("Loading training dataset to acquire statistics...")
    train_dataset = MTL_Dataset(
        rb_file_path=args.train_rb_path,
        sinr_file_path=args.train_sinr_path,
        obs_window=args.hist_window,
        pre_window=args.pre_window
    )
    sinr_mean = train_dataset.sinr_mean
    sinr_std = train_dataset.sinr_std

    print("\nLoading test data...")
    test_dataset = MTL_Dataset(
        rb_file_path=args.rb_test_path,
        sinr_file_path=args.sinr_test_path,
        obs_window=args.hist_window,
        pre_window=args.pre_window,
        sinr_mean=sinr_mean,
        sinr_std=sinr_std
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载 throughput 数据
    print("Loading throughput labels...")
    import pandas as pd
    throughput_data = pd.read_pickle(args.throughput_test_path)
    if isinstance(throughput_data, dict):
        throughput_labels = throughput_data.get('Throughput', throughput_data.get('throughput', None))
    else:
        throughput_labels = throughput_data
    
    throughput_labels = np.array(throughput_labels)
    print(f"Original throughput labels shape: {throughput_labels.shape}")
    
    # 展平 throughput_labels
    if throughput_labels.ndim == 2:
        if throughput_labels.shape[0] == 1:
            throughput_labels = throughput_labels[0]  # Shape was [1, N]
        elif throughput_labels.shape[1] == 1:
            throughput_labels = throughput_labels[:, 0]  # Shape was [N, 1]
    
    throughput_labels = torch.tensor(throughput_labels, dtype=torch.float32)
    print(f"Processed throughput labels shape: {throughput_labels.shape}")
    
    print(f"\nInitializing model and loading weights from '{args.model_path}'...")
    model = CrossStitch_MTL_Model(
        rb_input_size=1, sinr_input_size=1, 
        rb_hidden_size=args.rb_hidden_dim,
        sinr_hidden_size=args.sinr_hidden_dim, 
        rb_num_layers=args.rb_layers,
        sinr_num_layers=args.sinr_layers, 
        rb_output_size=106, 
        sinr_output_size=1,
        pre_window=args.pre_window, 
        device=DEVICE, 
        cross_stitch_mode=args.cross_stitch_mode
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    print("Model weights loaded successfully.\n")

    total_mse = 0
    total_mae = 0
    total_are = 0
    total_samples = 0
    
    all_mse = []
    all_mae = []
    all_are = []
    frame_avg_mse = []
    frame_avg_mae = []
    frame_avg_are = []
        
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", unit="batch",ncols=120)
        step_idx = args.hist_window   #第一个 throughput_label 从 16 帧开始
        for batch_idx, (rb_seq, sinr_seq, rb_label, sinr_label) in enumerate(pbar):

            rb_seq = rb_seq.to(DEVICE)
            sinr_seq = sinr_seq.to(DEVICE)
            
            # 从MTL模型中获取 RB 预测和 SINR 预测
            rb_pred, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)
            
            # 处理 RB 预测
            rb_class_preds = torch.argmax(rb_pred, dim=2) + 1  # Convert to 1-106 range
            rb_class_preds = rb_class_preds.squeeze(0)
            
            # 处理 SINR 预测
            sinr_pred_denorm = sinr_pred * sinr_std + sinr_mean
            sinr_pred_denorm = sinr_pred_denorm.squeeze(0).squeeze(-1)
            
            # 从预测中计算 throughput
            throughput_pred = calculate_throughput_from_predictions_parallel(
                rb_class_preds, sinr_pred_denorm, DEVICE
            )

            # 获取对应的 throughput_label
            start_idx = step_idx
            end_idx = start_idx + args.pre_window
            step_idx += 1
            
            if end_idx > len(throughput_labels):
                continue
                
            throughput_label_batch = throughput_labels[start_idx:end_idx].to(DEVICE)
            if len(throughput_pred) != len(throughput_label_batch):
                continue

            # 计算 MSE 和 MAE
            mse = ((throughput_pred - throughput_label_batch) ** 2).cpu().numpy()
            mae = torch.abs(throughput_pred - throughput_label_batch).cpu().numpy()
            
            avg_throughput_pred = torch.mean(throughput_pred)
            avg_throughput_label = torch.mean(throughput_label_batch)
            batch_are = (torch.abs(avg_throughput_pred - avg_throughput_label) / (avg_throughput_label + 1e-8) * 100).cpu().numpy()
            
            all_mse.extend(mse)
            all_mae.extend(mae)
            all_are.append(batch_are)
            frame_mse = np.mean(mse)
            frame_mae = np.mean(mae)
            frame_are = batch_are
            
            frame_avg_mse.append(frame_mse)
            frame_avg_mae.append(frame_mae)
            frame_avg_are.append(frame_are)
            
            total_mse += np.sum(mse)
            total_mae += np.sum(mae)
            total_are += batch_are
            total_samples += len(mse)  

            pbar.set_postfix({
                'MSE': f"{frame_mse:.5f}", 
                'MAE': f"{frame_mae:.5f}", 
                'ARE': f"{frame_are:.5f}%"
            })
    
    avg_mse = total_mse / total_samples 
    avg_mae = total_mae / total_samples  
    avg_are = total_are / len(frame_avg_are) 
    
    # 输出结果
    print("\n" + "="*50)
    print("     Throughput Prediction Results (MTL Model)    ")
    print("="*50)
    print(f"  Average MSE  : {avg_mse:.5f}")
    print(f"  Average MAE  : {avg_mae:.5f}")
    print(f"  Average ARE  : {avg_are:.5f}% (calculated using batch averages)")
    print("="*50)
    
    print("\nSaving detailed results...")
    np.save('MTL_all_mse.npy', np.array(all_mse))
    np.save('MTL_all_mae.npy', np.array(all_mae))
    np.save('MTL_all_are.npy', np.array(all_are))
    np.save('MTL_frame_avg_mse.npy', np.array(frame_avg_mse))
    np.save('MTL_frame_avg_mae.npy', np.array(frame_avg_mae))
    np.save('MTL_frame_avg_are.npy', np.array(frame_avg_are))
    print("Results saved!")
    plot_results(frame_avg_mse, frame_avg_mae, frame_avg_are)

def plot_results(mse_list, mae_list, are_list):
    """Plot MSE, MAE, and ARE over test samples"""
    plt.figure(figsize=(30, 10))
    
    plt.subplot(1, 3, 1)
    plt.plot(mse_list, 'b-', alpha=0.7)
    plt.title('MSE per 100-frame Window')
    plt.xlabel('Test Sample')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(mae_list, 'g-', alpha=0.7)
    plt.title('MAE per 100-frame Window')
    plt.xlabel('Test Sample')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(are_list, 'r-', alpha=0.7)
    plt.title('ARE per 100-frame Window (Batch Average)')
    plt.xlabel('Test Sample')
    plt.ylabel('ARE (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('MTL_throughput_metrics.png', dpi=150)
    plt.close()
    print("Plots saved to 'MTL_throughput_metrics.png'")

if __name__ == '__main__':
    args = get_args()
    main(args)