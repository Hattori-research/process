import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 1. ハイパーパラメータ
# ==========================================
TRAIN_CSV_DIR = "Data/rnn_csv/train"
TEST_CSV_DIR  = "Data/rnn_csv/test"
WEIGHTS_DIR   = "Data/Weights"         

PAST_SEQ = 40        # 2秒で十分です！
FUTURE_SEQ = 20      # 1秒予測
BATCH_SIZE = 64       
EPOCHS   = 200
PATIENCE = 20          
LEARNING_RATE = 1e-3  

os.makedirs(WEIGHTS_DIR, exist_ok=True)

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file_paths, past_seq=40, future_seq=20, is_train=True):
        self.past_seq = past_seq
        self.future_seq = future_seq
        
        self.x_past = []    
        self.w_target_future = []  
        self.y_future = []  
        
        all_target_data = []
        all_angle_data = []
        all_coord_data = []
        
        MOVEMENT_LIMIT = 120.0 
        CUT_INITIAL_STEPS = 100 
        
        print(f"[{len(csv_file_paths)} 個のファイルを読み込みます...]")
        for file in csv_file_paths:
            df = pd.read_csv(file)
            if len(df) <= CUT_INITIAL_STEPS + self.past_seq + self.future_seq:
                continue
            
            # ====================================================
            # ★ 超重要修正：切り取る"前"の1行目を「真のゼロ点」とする
            # ====================================================
            true_base_angle = df[['Angle_W0', 'Angle_W1', 'Angle_W2', 'Angle_W3']].values[0]
            true_base_coord = df[['X', 'Y', 'Z']].values[0]
                
            targets = df[['Target_W0', 'Target_W1', 'Target_W2', 'Target_W3']].values[CUT_INITIAL_STEPS:]
            angles = df[['Angle_W0', 'Angle_W1', 'Angle_W2', 'Angle_W3']].values[CUT_INITIAL_STEPS:]
            coords = df[['X', 'Y', 'Z']].values[CUT_INITIAL_STEPS:]
            
            # エンコーダ値は常に「真のゼロ点」からの引張量
            angles_rel = angles - true_base_angle
            
            # ノイズ除去用のクリッピングも「真のゼロ点」を基準に行う
            coords_cleaned = np.clip(coords, true_base_coord - MOVEMENT_LIMIT, true_base_coord + MOVEMENT_LIMIT)
            
            all_target_data.append(targets)
            all_angle_data.append(angles_rel)
            all_coord_data.append(coords_cleaned)
            
        combined_target = np.vstack(all_target_data)
        combined_angle = np.vstack(all_angle_data)
        
        # 相対座標（現在地からの変位）を集めて定規を作る
        all_rel_coords = []
        for c_data in all_coord_data:
            for t in range(self.past_seq, len(c_data) - self.future_seq):
                current_pos = c_data[t - 1]
                all_rel_coords.append(c_data[t - self.past_seq : t] - current_pos)
                all_rel_coords.append(c_data[t : t + self.future_seq] - current_pos)
                
        combined_rel_coords = np.vstack(all_rel_coords)
        
        target_scaler_path = os.path.join(WEIGHTS_DIR, 'target_scaler.pkl')
        angle_scaler_path = os.path.join(WEIGHTS_DIR, 'angle_scaler.pkl')
        rel_coord_scaler_path = os.path.join(WEIGHTS_DIR, 'rel_coord_scaler.pkl')
        
        if is_train:
            self.target_scaler = StandardScaler().fit(combined_target)
            self.angle_scaler = StandardScaler().fit(combined_angle)
            self.rel_coord_scaler = StandardScaler().fit(combined_rel_coords) 
            joblib.dump(self.target_scaler, target_scaler_path)
            joblib.dump(self.angle_scaler, angle_scaler_path)
            joblib.dump(self.rel_coord_scaler, rel_coord_scaler_path)
        else:
            self.target_scaler = joblib.load(target_scaler_path)
            self.angle_scaler = joblib.load(angle_scaler_path)
            self.rel_coord_scaler = joblib.load(rel_coord_scaler_path)
            
        for t_data, a_data, c_data in zip(all_target_data, all_angle_data, all_coord_data):
            t_scaled = self.target_scaler.transform(t_data)
            a_scaled = self.angle_scaler.transform(a_data)

            for t in range(self.past_seq, len(t_scaled) - self.future_seq):
                current_pos = c_data[t - 1]


                past_rel_mm   = c_data[t - self.past_seq : t] - current_pos
                future_rel_mm = c_data[t : t + self.future_seq] - current_pos

                # ジャンプ対策：ウィンドウ内の座標変化量が閾値を超えたらスキップ
                JUMP_THRESH = 150.0  # mm：1ステップの最大変化量
                past_diff   = np.abs(np.diff(past_rel_mm, axis=0)).max()
                future_diff = np.abs(np.diff(future_rel_mm, axis=0)).max()
                if past_diff > JUMP_THRESH or future_diff > JUMP_THRESH:
                    continue

                past_rel_scaled   = self.rel_coord_scaler.transform(past_rel_mm)
                future_rel_scaled = self.rel_coord_scaler.transform(future_rel_mm)

                xp_a = a_scaled[t - self.past_seq : t]
                self.x_past.append(np.hstack((xp_a, past_rel_scaled)))
                self.w_target_future.append(t_scaled[t : t + self.future_seq])
                self.y_future.append(future_rel_scaled)

    def __len__(self): return len(self.x_past)
    def __getitem__(self, idx):
        return (torch.tensor(self.x_past[idx], dtype=torch.float32),
                torch.tensor(self.w_target_future[idx], dtype=torch.float32),
                torch.tensor(self.y_future[idx], dtype=torch.float32))

class TrajectoryNet(nn.Module):
    def __init__(self, past_seq=40, future_seq=20):
        super(TrajectoryNet, self).__init__()
        self.future_seq = future_seq
        self.lstm = nn.LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64 + (future_seq * 4), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, future_seq * 3)
        )

    def forward(self, x_past, w_target_future):
        lstm_out, _ = self.lstm(x_past)
        summary = lstm_out[:, -1, :]
        w_flat = w_target_future.reshape(w_target_future.size(0), -1)
        combined = torch.cat([summary, w_flat], dim=1)
        out_flat = self.fc(combined)
        return out_flat.reshape(out_flat.size(0), self.future_seq, 3)

def main():
    # 学習データ（①②を結合）
    patience_cnt = 0
    train_csv = glob.glob(os.path.join(TRAIN_CSV_DIR, "*.csv"))
    test_csv  = glob.glob(os.path.join(TEST_CSV_DIR,  "*.csv"))

    if not train_csv:
        raise FileNotFoundError(f"学習データが見つかりません: {TRAIN_CSV_DIR}")
    if not test_csv:
        raise FileNotFoundError(f"テストデータが見つかりません: {TEST_CSV_DIR}")

    print(f"学習データ ({len(train_csv)}ファイル): {train_csv}")
    print(f"テストデータ ({len(test_csv)}ファイル): {test_csv}")


    # 学習データでスケーラをフィット
    full_dataset = TrajectoryDataset(train_csv, past_seq=PAST_SEQ, future_seq=FUTURE_SEQ, is_train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # テストデータは学習済みスケーラを使用
    test_dataset = TrajectoryDataset(test_csv, past_seq=PAST_SEQ, future_seq=FUTURE_SEQ, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryNet(past_seq=PAST_SEQ, future_seq=FUTURE_SEQ).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    model_save_path = os.path.join(WEIGHTS_DIR, "best_trajectory_model.pth")

    print(f"\n=== エゴセントリック相対座標系モデル 学習開始 ===")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x_past, w_future, y_future in train_loader:
            x_past, w_future, y_future = x_past.to(device), w_future.to(device), y_future.to(device)
            optimizer.zero_grad()
            pred = model(x_past, w_future)
            loss = criterion(pred, y_future)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_past.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_past, w_future, y_future in val_loader:
                x_past, w_future, y_future = x_past.to(device), w_future.to(device), y_future.to(device)
                pred = model(x_past, w_future)
                val_loss += criterion(pred, y_future).item() * x_past.size(0)
        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[Early Stopping] {PATIENCE}エポック改善なし → 終了")
                break


    # === テスト評価 ===
    print("\n=== テストデータ評価 ===")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.load_state_dict(torch.load(model_save_path, weights_only=False))
    model.eval()

    preds_all, trues_all = [], []
    with torch.no_grad():
        for x_past, w_future, y_future in test_loader:
            x_past, w_future = x_past.to(device), w_future.to(device)
            pred = model(x_past, w_future).cpu().numpy()
            preds_all.append(pred)
            trues_all.append(y_future.numpy())

    preds = np.concatenate(preds_all, axis=0)  # (N, future_seq, 3)
    trues = np.concatenate(trues_all, axis=0)

    # 逆正規化してmm単位で評価
    scaler = full_dataset.rel_coord_scaler
    N, F, C = preds.shape
    preds_mm = scaler.inverse_transform(preds.reshape(-1, C)).reshape(N, F, C)
    trues_mm = scaler.inverse_transform(trues.reshape(-1, C)).reshape(N, F, C)

    err = preds_mm - trues_mm
    rmse_axis = np.sqrt((err**2).mean(axis=(0, 1)))   # (3,)
    rmse_total = np.sqrt((err**2).sum(axis=-1).mean())
    mae_step   = np.abs(err).mean(axis=(0, 2))        # (future_seq,)

    print(f"  RMSE (X): {rmse_axis[0]:.3f} mm")
    print(f"  RMSE (Y): {rmse_axis[1]:.3f} mm")
    print(f"  RMSE (Z): {rmse_axis[2]:.3f} mm")
    print(f"  RMSE (3D Euclidean): {rmse_total:.3f} mm")
    print(f"\n  MAE per prediction step (mm):")
    for i, mae in enumerate(mae_step):
        print(f"    step +{i+1:2d}: {mae:.3f} mm")

if __name__ == "__main__":
    main()