# train_mtl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
import random

from mtl_dataset import MTL_Dataset
from mtl_model import CrossStitch_MTL_Model

def seed_initial(seed=0):
    print(f"Setting random seed to {seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def get_args():
    parser = argparse.ArgumentParser(description="Train an Asymmetrical Cross-Stitch MTL Model with Autoregressive Decoder")
    parser.add_argument('--rb_train_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/train_dataset_10000.pkl')
    parser.add_argument('--sinr_train_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/train_dataset_10000.pkl')
    parser.add_argument('--rb_test_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/test_dataset_10000.pkl')
    parser.add_argument('--sinr_test_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Dataset/Dataset_Traffic_Model_VoIP/test_dataset_10000.pkl')
    parser.add_argument('--skip_frames', type=int, default=10)
    parser.add_argument('--hist_window', type=int, default=16)
    parser.add_argument('--pre_window', type=int, default=100)
    parser.add_argument('--rb_hidden_dim', type=int, default=256)
    parser.add_argument('--sinr_hidden_dim', type=int, default=64)
    parser.add_argument('--rb_layers', type=int, default=3)
    parser.add_argument('--sinr_layers', type=int, default=1)
    parser.add_argument('--cross_stitch_mode', type=str, default='learn', choices=['learn','identity','zeros'])
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sinr_loss', type=str, default='mse', choices=['mse','mae','huber'])
    parser.add_argument('--huber_beta', type=float, default=0.05)
    parser.add_argument('--gamma_rb_dist', type=float, default=0.1)
    parser.add_argument('--uncertainty-loss', default=True, action='store_true')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--log_file', type=str, default='training_log_decoder.json')
    parser.add_argument('--model_save_path', type=str, default='/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Task3_VoIP/Cross_Stitch/Train/Model_2/best_mtl_model_decoder.pth')
    parser.add_argument('--pareto_eps_acc', type=float, default=0.0)
    parser.add_argument('--pareto_eps_mae', type=float, default=0.0)
    return parser.parse_args()

def val_to_index(val):
    return (val - 1).long()

def build_sinr_criterion(kind, beta):
    if kind == 'mse': return nn.MSELoss()
    if kind == 'mae': return nn.L1Loss()
    if kind == 'huber': return nn.SmoothL1Loss(beta=beta)
    raise ValueError(kind)

def pareto_better(curr_acc, curr_mae, best_acc, best_mae, eps_acc=0.0, eps_mae=0.0):
    better_on_acc = (curr_acc > best_acc + eps_acc) and (curr_mae <= best_mae + eps_mae)
    better_on_mae = (curr_mae < best_mae - eps_mae) and (curr_acc >= best_acc - eps_acc)
    return better_on_acc or better_on_mae

def main(args):
    seed_initial(args.seed)   # 随机数，保证实验可复现
    DEVICE = torch.device(args.device)  # 设备：GPU/CPU
    print(f"Using device: {DEVICE}\nArguments: {args}\n")

    # 构建训练集和测试集
    train_dataset = MTL_Dataset(args.rb_train_path, args.sinr_train_path, obs_window=args.hist_window, pre_window=args.pre_window, skip_initial_frames=args.skip_frames)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    sinr_mean, sinr_std = train_dataset.sinr_mean, train_dataset.sinr_std
    test_dataset = MTL_Dataset(args.rb_test_path, args.sinr_test_path, obs_window=args.hist_window, pre_window=args.pre_window, sinr_mean=sinr_mean, sinr_std=sinr_std)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 构建模型
    model = CrossStitch_MTL_Model(
        rb_input_size=1, sinr_input_size=1, rb_hidden_size=args.rb_hidden_dim,
        sinr_hidden_size=args.sinr_hidden_dim, rb_num_layers=args.rb_layers,
        sinr_num_layers=args.sinr_layers, rb_output_size=106, sinr_output_size=1,
        pre_window=args.pre_window, device=DEVICE, cross_stitch_mode=args.cross_stitch_mode
    ).to(DEVICE)

    model.load_state_dict(torch.load(
        "/8T2/lhl/lhl_network/Channel_Predict_file/Code_Traffic_Model/Task3_VoIP/Cross_Stitch/Train/Model/best_mtl_model_decoder.pth"))


    # 构建损失函数
    criterion_rb_ce = nn.CrossEntropyLoss()
    criterion_sinr  = build_sinr_criterion(args.sinr_loss, args.huber_beta)
    params_to_optimize = list(model.parameters())
    # 构建 uncertainty_loss
    if args.uncertainty_loss:
        log_var_rb = torch.zeros((1,), requires_grad=True, device=DEVICE)    # 可训练参数
        log_var_sinr = torch.zeros((1,), requires_grad=True, device=DEVICE)  # 可训练参数
        params_to_optimize.extend([log_var_rb, log_var_sinr])
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_rb_acc, best_sinr_mae = 0.0, float('inf')
    classes = torch.arange(1, 107, device=DEVICE).float()
    tf_ratio_start, tf_ratio_end, tf_decay_epochs = 0.7, 0.2, 50

    for epoch in range(1, args.epochs + 1):
        # 训练过程
        model.train()
        tf_ratio = tf_ratio_start - (tf_ratio_start - tf_ratio_end) * min(1.0, epoch / tf_decay_epochs)
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train, TF={tf_ratio:.2f}]")
        for rb_seq, sinr_seq, rb_label, sinr_label in pbar_train:
            rb_seq, sinr_seq, rb_label, sinr_label = rb_seq.to(DEVICE), sinr_seq.to(DEVICE), rb_label.to(DEVICE), sinr_label.to(DEVICE)
            optimizer.zero_grad()

            teacher_rb_onehot = F.one_hot(val_to_index(rb_label.squeeze(-1)), num_classes=model.rb_output_size).float()

            rb_pred, sinr_pred = model(
                rb_seq, sinr_seq, teacher_rb=teacher_rb_onehot,
                teacher_sinr=sinr_label, tf_ratio=tf_ratio
            )

            # 计算损失函数
            loss_rb = criterion_rb_ce(rb_pred.reshape(-1, 106), val_to_index(rb_label.reshape(-1)))
            loss_sinr = criterion_sinr(sinr_pred, sinr_label)

            if args.uncertainty_loss:
                log_var_rb.data.clamp_(-5.0, 5.0)
                log_var_sinr.data.clamp_(-5.0, 5.0)
                precision_rb = torch.exp(-log_var_rb)           # 1/σ^2
                precision_sinr = torch.exp(-log_var_sinr)       # 1/σ^2
                loss_rb_weighted   = precision_rb * loss_rb +  0.5 * log_var_rb   # 注意分类模型不需要 0.5 的权重系数
                loss_sinr_weighted = 0.5 * (precision_sinr * loss_sinr + log_var_sinr)     # 回归模型需要 0.5 的权重系数
                loss = loss_rb_weighted + loss_sinr_weighted
            else:
                loss = 1.0 * loss_rb + 1.0 * loss_sinr

            if args.gamma_rb_dist > 0.0:
                probs = F.softmax(rb_pred, dim=2)
                rb_expect = (probs * classes).sum(dim=2)
                rb_true = rb_label.squeeze(-1).float()
                loss_rb_dist = F.l1_loss(rb_expect, rb_true)
                loss = loss + args.gamma_rb_dist * loss_rb_dist

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 梯度裁剪
            optimizer.step()
        scheduler.step()

        model.eval()
        total_rb_correct, total_rb_abs_err = 0, 0.0
        total_sinr_mae_denorm, total_sinr_mse_denorm = 0.0, 0.0

        # 测试过程
        with torch.no_grad():
            for rb_seq, sinr_seq, rb_label, sinr_label in test_loader:
                rb_seq, sinr_seq, rb_label, sinr_label = rb_seq.to(DEVICE), sinr_seq.to(DEVICE), rb_label.to(DEVICE), sinr_label.to(DEVICE)
                rb_pred, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)

                rb_class = torch.argmax(rb_pred, dim=2) + 1
                total_rb_correct += (rb_class == rb_label.squeeze(-1)).sum().item()
                total_rb_abs_err += torch.abs(rb_class - rb_label.squeeze(-1)).sum().item()

                sinr_pred_denorm = sinr_pred * sinr_std + sinr_mean
                sinr_label_denorm = sinr_label * sinr_std + sinr_mean
                total_sinr_mae_denorm += torch.abs(sinr_pred_denorm - sinr_label_denorm).sum().item()
                total_sinr_mse_denorm += ((sinr_pred_denorm - sinr_label_denorm) ** 2).sum().item()

        N_test_points = len(test_dataset) * args.pre_window
        test_rb_acc = total_rb_correct / N_test_points
        test_rb_mae = total_rb_abs_err / N_test_points
        test_sinr_mae = total_sinr_mae_denorm / N_test_points
        test_sinr_mse = total_sinr_mse_denorm / N_test_points

        print(f"\nEpoch {epoch} Summary:")
        print(f"  [Metrics] Test RB Acc: {test_rb_acc*100:.2f}% | RB MAE: {test_rb_mae:.4f}")
        print(f"  [Metrics] Test SINR MAE (original scale): {test_sinr_mae:.4f} | SINR MSE: {test_sinr_mse:.4f}")
        if args.uncertainty_loss:
            print(f"  [Uncertainty] Learned Weights -> RB: {torch.exp(-log_var_rb).item():.4f}, SINR: {torch.exp(-log_var_sinr).item():.4f}")

        if pareto_better(test_rb_acc, test_sinr_mae, best_rb_acc, best_sinr_mae, eps_acc=args.pareto_eps_acc, eps_mae=args.pareto_eps_mae):
            old_acc, old_mae = best_rb_acc, best_sinr_mae
            best_rb_acc = max(best_rb_acc, test_rb_acc)
            best_sinr_mae = min(best_sinr_mae, test_sinr_mae)
            torch.save(model.state_dict(), args.model_save_path)
            print(f"  -> [Model Saved] New Pareto best. ({old_acc*100:.2f}%, {old_mae:.4f}) -> ({test_rb_acc*100:.2f}%, {test_sinr_mae:.4f})")

    print("\nTraining finished.")

if __name__ == "__main__":
    args = get_args()
    main(args)