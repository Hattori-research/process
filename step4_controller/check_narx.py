# check_narx.py として保存
import torch
import joblib
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'step3_narx'))

WEIGHTS_DIR = "Data/Weights"

class TrajectoryNet(torch.nn.Module):
    def __init__(self, past_seq=40, future_seq=20):
        super().__init__()
        self.future_seq = future_seq
        self.lstm = torch.nn.LSTM(input_size=7, hidden_size=64,
                                   num_layers=2, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 + future_seq * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, future_seq * 3)
        )
    def forward(self, x_past, w_future):
        out, _ = self.lstm(x_past)
        summary = out[:, -1, :]
        w_flat = w_future.reshape(w_future.size(0), -1)
        out = self.fc(torch.cat([summary, w_flat], dim=1))
        return out.reshape(out.size(0), self.future_seq, 3)

narx = TrajectoryNet()
narx.load_state_dict(torch.load(
    os.path.join(WEIGHTS_DIR, "best_trajectory_model.pth"),
    weights_only=True))
narx.eval()

target_scaler = joblib.load(os.path.join(WEIGHTS_DIR, "target_scaler.pkl"))

# ゼロ入力でNARXの出力を確認
x_past = torch.zeros(1, 40, 7)
# 引張量8mm（スケール変換）
w_8mm = np.full((20, 4), 8.0)
w_scaled = target_scaler.transform(w_8mm.reshape(-1, 4)).reshape(1, 20, 4)
w_tensor = torch.from_numpy(w_scaled.astype(np.float32))

with torch.no_grad():
    pred = narx(x_past, w_tensor)
    print(f"引張量8mm時の予測位置変化 (相対座標mm):")
    print(f"  step+1:  {pred[0, 0].numpy()}")
    print(f"  step+10: {pred[0, 9].numpy()}")
    print(f"  step+20: {pred[0, 19].numpy()}")