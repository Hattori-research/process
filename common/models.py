"""
common/models.py

step3〜5 で共通に使うネットワーク定義
（構造パラメータは config.toml [model] [narx] [controller]。変更したら再学習が必要）
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config


# ========================================================
# NARX順モデル（step3 で学習、step4 では重み固定で使用）
# ========================================================
class TrajectoryNet(nn.Module):
    """
    入力:
      x_past   (B, PAST_SEQ, 7)   過去の [エンコーダ相対値(4) + 相対位置(3)]（正規化済み）
      w_future (B, FUTURE_SEQ, 4) 未来の目標引張量（正規化済み）
    出力:
      (B, FUTURE_SEQ, 3) 未来の相対位置（正規化済み）
    """
    def __init__(self, past_seq=None, future_seq=None):
        super().__init__()
        cfg = config.load()
        ncfg = cfg["narx"]
        if future_seq is None:
            future_seq = cfg["model"]["future_seq"]
        self.future_seq = future_seq
        hidden = ncfg["lstm_hidden"]
        fc     = ncfg["fc_hidden"]

        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden,
                            num_layers=ncfg["lstm_layers"], batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden + future_seq * 4, fc), nn.ReLU(),
            nn.Linear(fc, fc),                    nn.ReLU(),
            nn.Linear(fc, future_seq * 3)
        )

    def forward(self, x_past, w_future):
        out, _ = self.lstm(x_past)
        summary = out[:, -1, :]
        w_flat  = w_future.reshape(w_future.size(0), -1)
        out     = self.fc(torch.cat([summary, w_flat], dim=1))
        return out.reshape(out.size(0), self.future_seq, 3)


# ========================================================
# MLPコントローラ（手法A・B共通アーキテクチャ）
# ========================================================
class ControllerMLP(nn.Module):
    """
    入力:
      current_pos (3,)         現在位置（相対座標正規化済み）
      target_pos  (3,)         目標位置（相対座標正規化済み）
      past_w      (PAST_SEQ,4) 過去引張量（正規化済み）
      past_angle  (PAST_SEQ,4) 過去エンコーダ（正規化済み）
    出力:
      w_out (FUTURE_SEQ, 4)  [0,1] → 後でW_MIN/W_MAXにスケール
    """
    def __init__(self, past_seq=None, future_seq=None,
                 hidden_size=None, num_layers=None, dropout=None):
        super().__init__()
        cfg  = config.load()
        ccfg = cfg["controller"]
        if past_seq is None:
            past_seq = cfg["model"]["past_seq"]
        if future_seq is None:
            future_seq = cfg["model"]["future_seq"]
        if hidden_size is None:
            hidden_size = ccfg["hidden_size"]
        if num_layers is None:
            num_layers = ccfg["num_layers"]
        if dropout is None:
            dropout = ccfg["dropout"]

        self.future_seq = future_seq
        in_dim  = 3 + 3 + past_seq * 4 + past_seq * 4
        out_dim = future_seq * 4

        layers, d = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_size),
                       nn.LayerNorm(hidden_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            d = hidden_size
        layers += [nn.Linear(d, out_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, cur, tgt, pw, pa):
        x = torch.cat([cur, tgt,
                        pw.reshape(pw.size(0), -1),
                        pa.reshape(pa.size(0), -1)], dim=1)
        return self.net(x).reshape(x.size(0), self.future_seq, 4)
