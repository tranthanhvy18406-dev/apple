# mtl_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossStitchUnit(nn.Module):
    def __init__(self, num_features):
        super(CrossStitchUnit, self).__init__()
        self.alpha = nn.Parameter(torch.eye(2, 2))

    def forward(self, x_a, x_b):
        x_a_reshaped = x_a.unsqueeze(-1)
        x_b_reshaped = x_b.unsqueeze(-1)
        input_stack = torch.cat([x_a_reshaped, x_b_reshaped], dim=-1)
        output_stack = torch.matmul(input_stack, self.alpha)
        return output_stack[..., 0], output_stack[..., 1]

class TinyDecoderGRU(nn.Module):
    def __init__(self, hidden_dim, rb_classes, sinr_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim + rb_classes + sinr_dim, hidden_size=hidden_dim, batch_first=True)
        self.rb_head = nn.Linear(hidden_dim, rb_classes)
        self.sinr_head = nn.Linear(hidden_dim, sinr_dim)
        self.go = nn.Parameter(torch.randn(1, 1, rb_classes + sinr_dim))

    def forward(self, hT, context, steps, teacher_rb=None, teacher_sinr=None, tf_ratio=0.0):
        B, H = hT.shape[1], hT.shape[2]
        outputs_rb, outputs_sinr = [], []
        prev_y = self.go.expand(B, 1, -1)
        h = hT

        # 迭代预测 100 帧
        for t in range(steps):
            dec_in = torch.cat([context, prev_y], dim=-1)
            out, h = self.gru(dec_in, h)
            rb_logit = self.rb_head(out)
            sinr_hat = self.sinr_head(out)
            outputs_rb.append(rb_logit)
            outputs_sinr.append(sinr_hat)
            
            use_teacher = (teacher_rb is not None) and (torch.rand(1).item() < tf_ratio)
            if use_teacher:
                prev_y = torch.cat([teacher_rb[:, t:t+1, :], teacher_sinr[:, t:t+1, :]], dim=-1)
            else:
                rb_onehot = F.one_hot(rb_logit.argmax(dim=-1), num_classes=rb_logit.shape[-1]).float()
                prev_y = torch.cat([rb_onehot, sinr_hat], dim=-1)
        
        return torch.cat(outputs_rb, dim=1), torch.cat(outputs_sinr, dim=1)

class CrossStitch_MTL_Model(nn.Module):
    def __init__(self, rb_input_size, sinr_input_size, 
                 rb_hidden_size, sinr_hidden_size, 
                 rb_num_layers, sinr_num_layers,
                 rb_output_size, sinr_output_size,
                 pre_window, device, cross_stitch_mode='learn'):
        super(CrossStitch_MTL_Model, self).__init__()
        self.device = device
        self.pre_window = pre_window
        self.rb_output_size = rb_output_size
        self.cross_stitch_mode = cross_stitch_mode

        self.rb_lstms = nn.ModuleList([nn.LSTM(rb_input_size, rb_hidden_size, batch_first=True)] + 
                                      [nn.LSTM(rb_hidden_size, rb_hidden_size, batch_first=True) for _ in range(rb_num_layers - 1)])
        self.sinr_lstms = nn.ModuleList([nn.LSTM(sinr_input_size, sinr_hidden_size, batch_first=True)] + 
                                        [nn.LSTM(sinr_hidden_size, sinr_hidden_size, batch_first=True) for _ in range(sinr_num_layers - 1)])
        self.num_cross_stitch = 0 if cross_stitch_mode == 'none' else min(rb_num_layers, sinr_num_layers)
        if self.num_cross_stitch > 0:
            self.cross_stitch_dim = max(rb_hidden_size, sinr_hidden_size)
            self.rb_projections_up = nn.ModuleList()
            self.sinr_projections_up = nn.ModuleList()
            self.cross_stitch_units = nn.ModuleList()
            self.rb_projections_down = nn.ModuleList()
            self.sinr_projections_down = nn.ModuleList()
            for i in range(self.num_cross_stitch):
                self.rb_projections_up.append(nn.Linear(rb_hidden_size, self.cross_stitch_dim))
                self.sinr_projections_up.append(nn.Linear(sinr_hidden_size, self.cross_stitch_dim))
                unit = CrossStitchUnit(self.cross_stitch_dim)
                if cross_stitch_mode == 'identity': unit.alpha.data = torch.eye(2,2); unit.alpha.requires_grad = False
                elif cross_stitch_mode == 'zeros': unit.alpha.data = torch.zeros(2,2); unit.alpha.requires_grad = False
                self.cross_stitch_units.append(unit)
                self.rb_projections_down.append(nn.Linear(self.cross_stitch_dim, rb_hidden_size))
                self.sinr_projections_down.append(nn.Linear(self.cross_stitch_dim, sinr_hidden_size))

        self.decoder_hidden_dim = max(rb_hidden_size, sinr_hidden_size)
        self.proj_rb_for_dec = nn.Linear(rb_hidden_size, self.decoder_hidden_dim) if rb_hidden_size != self.decoder_hidden_dim else nn.Identity()
        self.proj_sinr_for_dec = nn.Linear(sinr_hidden_size, self.decoder_hidden_dim) if sinr_hidden_size != self.decoder_hidden_dim else nn.Identity()
        self.decoder = TinyDecoderGRU(self.decoder_hidden_dim, rb_output_size, sinr_output_size)

    def forward(self, rb_seq, sinr_seq, teacher_rb=None, teacher_sinr=None, tf_ratio=0.0):
        rb_out, sinr_out = rb_seq, sinr_seq
        rb_h_tuple, sinr_h_tuple = None, None
        max_layers = max(len(self.rb_lstms), len(self.sinr_lstms))
        for i in range(max_layers):
            if i < len(self.rb_lstms):
                rb_out, rb_h_tuple = self.rb_lstms[i](rb_out)
            if i < len(self.sinr_lstms):
                sinr_out, sinr_h_tuple = self.sinr_lstms[i](sinr_out)
            if i < self.num_cross_stitch:
                rb_proj, sinr_proj = self.rb_projections_up[i](rb_out), self.sinr_projections_up[i](sinr_out)
                rb_cross, sinr_cross = self.cross_stitch_units[i](rb_proj, sinr_proj)
                rb_out, sinr_out = rb_out + self.rb_projections_down[i](rb_cross), sinr_out + self.sinr_projections_down[i](sinr_cross)
        
        h_rb_last = self.proj_rb_for_dec(rb_h_tuple[0])
        h_sinr_last = self.proj_sinr_for_dec(sinr_h_tuple[0])
        h_T = (h_rb_last + h_sinr_last) / 2.0
        h_T = h_T[-1, :, :].unsqueeze(0)
        context_vec = h_T.transpose(0, 1)

        rb_pred, sinr_pred = self.decoder(
            h_T, context_vec, self.pre_window,
            teacher_rb, teacher_sinr, tf_ratio
        )
        return rb_pred, sinr_pred
