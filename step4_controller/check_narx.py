# check_narx.py
# 学習済みNARXに「ゼロ入力 + 一定引張量」を与えて出力を確認する
import torch
import joblib
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from common.models import TrajectoryNet

mcfg = config.load()["model"]
PAST_SEQ, FUTURE_SEQ = mcfg["past_seq"], mcfg["future_seq"]
TEST_PULL_MM = 8.0   # 確認用の一定引張量 [mm]

narx = TrajectoryNet()
narx.load_state_dict(torch.load(config.path("narx_model"), weights_only=True))
narx.eval()

target_scaler = joblib.load(config.path("target_scaler"))

# ゼロ入力でNARXの出力を確認
x_past = torch.zeros(1, PAST_SEQ, 7)
# 一定引張量（スケール変換）
w_const = np.full((FUTURE_SEQ, 4), TEST_PULL_MM)
w_scaled = target_scaler.transform(w_const.reshape(-1, 4)).reshape(1, FUTURE_SEQ, 4)
w_tensor = torch.from_numpy(w_scaled.astype(np.float32))

with torch.no_grad():
    pred = narx(x_past, w_tensor)
    print(f"引張量{TEST_PULL_MM}mm時の予測位置変化 (相対座標mm):")
    for k in (1, FUTURE_SEQ // 2, FUTURE_SEQ):
        print(f"  step+{k:<3d}: {pred[0, k - 1].numpy()}")
