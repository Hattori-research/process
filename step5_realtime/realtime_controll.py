"""
step5_realtime/realtime_control.py

リアルタイム制御プログラム

実験モード:
  [1] ランダム点追従テスト
      テストデータからランダムに目標位置を選び、制御を繰り返して精度を評価する
  [2] キーボード手動入力
      ユーザーがXYZ座標を入力して目標位置を指定する

コントローラ切り替え:
  起動時に手法A（NARX間接）または手法B（直接逆モデル）を選択できる
"""

import os
import sys
import time
import glob
import threading
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'step1_cameracalibration'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'step2_datasampling'))

from stereo_triangulate import StereoTracker, KalmanFilter3D
from motor_control import MotorController

# ========================================================
# パス設定
# ========================================================
WEIGHTS_DIR  = "Data/Weights"
TEST_CSV_DIR = "Data/rnn_csv/test"
CTRL_A_PATH  = os.path.join(WEIGHTS_DIR, "best_controller_narx.pth")
CTRL_B_PATH  = os.path.join(WEIGHTS_DIR, "best_controller_direct.pth")
RESULT_DIR   = "Data/control_results"
VIDEO_DIR = "Data/control_results/video"

# ========================================================
# ハイパーパラメータ（train_rnn.py・train_controller.pyと統一）
# ========================================================
PAST_SEQ   = 40
FUTURE_SEQ = 20
W_MIN, W_MAX = 0.0, 16.0

# 制御設定
SAMPLING_RATE     = 20.0
INTERVAL          = 1.0 / SAMPLING_RATE
HOLD_STEPS        = 40        # 目標位置に向けて制御するステップ数
N_TRIALS          = 20        # ランダムテストの試行回数
CUT_INITIAL_STEPS = 100
MOVEMENT_LIMIT    = 120.0
POSITION_THRESH   = 10.0   # 到達判定の閾値 [mm]
STABLE_STEPS      = 5      # 閾値以下をこの回数連続で確認したら終了

# ========================================================
# モデル定義（train_rnn.py・train_controller.pyと同一）
# ========================================================
class TrajectoryNet(nn.Module):
    def __init__(self, past_seq=40, future_seq=20):
        super().__init__()
        self.future_seq = future_seq
        self.lstm = nn.LSTM(input_size=7, hidden_size=64,
                            num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64 + future_seq * 4, 128), nn.ReLU(),
            nn.Linear(128, 128),                  nn.ReLU(),
            nn.Linear(128, future_seq * 3)
        )

    def forward(self, x_past, w_future):
        out, _ = self.lstm(x_past)
        summary = out[:, -1, :]
        w_flat  = w_future.reshape(w_future.size(0), -1)
        out     = self.fc(torch.cat([summary, w_flat], dim=1))
        return out.reshape(out.size(0), self.future_seq, 3)


class ControllerMLP(nn.Module):
    def __init__(self, past_seq=40, future_seq=20,
                 hidden_size=256, num_layers=3, dropout=0.1):
        super().__init__()
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


# ========================================================
# 共有データ（カメラスレッド用）
# ========================================================
shared = {"pos": None, "frame_top": None, "frame_under": None,
          "running": True, "display_info": None}
data_lock = threading.Lock()


def camera_worker(tracker, video_writer=None):
    consecutive_errors = 0
    while shared["running"]:
        try:
            pos, frame_top, frame_under = tracker.get_3d_coordinates_and_frames()
            with data_lock:
                shared["pos"]         = pos
                shared["frame_top"]   = frame_top
                shared["frame_under"] = frame_under
                info = shared.get("display_info")   # メインスレッドから描画情報を受け取る
            if video_writer is not None and frame_top is not None:
                frame_out = frame_top.copy()
                if info is not None:
                    trial   = info.get("trial", "")
                    target  = info.get("target")
                    current = info.get("current")
                    err     = info.get("error")
                    w_cmd   = info.get("w_cmd")
                    step    = info.get("step", "")
                    y = 30
                    def put(text, color=(255,255,255)):
                        nonlocal y
                        cv2.putText(frame_out, text, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y += 30
                    put(f"Trial {trial}  Step {step}")
                    if target is not None:
                        put(f"Target : X={target[0]:.1f}  Y={target[1]:.1f}  Z={target[2]:.1f} mm",
                            (0, 255, 255))
                    if current is not None:
                        put(f"Current: X={current[0]:.1f}  Y={current[1]:.1f}  Z={current[2]:.1f} mm",
                            (200, 255, 200))
                    if err is not None:
                        color = (0, 255, 0) if err < POSITION_THRESH else (0, 80, 255)
                        put(f"Error  : {err:.1f} mm", color)
                    if w_cmd is not None:
                        put(f"W: [{w_cmd[0]:.1f}, {w_cmd[1]:.1f}, {w_cmd[2]:.1f}, {w_cmd[3]:.1f}]",
                            (255, 200, 100))
                video_writer.write(frame_out)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors % 10 == 1:
                print(f"\n[CameraThread] エラー({consecutive_errors}回目): {e}")
            if consecutive_errors > 100:
                print("\n[CameraThread] エラーが続くためスレッドを終了します")
                shared["running"] = False
                break
            time.sleep(0.05)


# ========================================================
# 状態バッファ管理
# ========================================================
class StateBuffer:
    """過去PAST_SEQ分の状態を管理するFIFOバッファ"""

    def __init__(self, past_seq, angle_scaler, target_scaler,
                 rel_coord_scaler, base_angle):
        self.past_seq         = past_seq
        self.angle_scaler     = angle_scaler
        self.target_scaler    = target_scaler
        self.rel_coord_scaler = rel_coord_scaler
        self.base_angle       = base_angle

        self.pos_buf   = []
        self.angle_buf = []
        self.w_buf     = []

    def push(self, pos, angle, w_cmd):
        self.pos_buf.append(pos.copy())
        self.angle_buf.append(angle.copy())
        self.w_buf.append(w_cmd.copy())
        if len(self.pos_buf) > self.past_seq:
            self.pos_buf.pop(0)
            self.angle_buf.pop(0)
            self.w_buf.pop(0)

    def is_ready(self):
        return len(self.pos_buf) >= self.past_seq

    def get_ctrl_input(self, target_pos):
        """コントローラへの入力テンソルを生成"""
        cur_p = np.array(self.pos_buf[-1])
        tgt_rel = (target_pos - cur_p).reshape(1, -1)

        cur_sc = np.zeros(3, dtype=np.float32)
        tgt_sc = self.rel_coord_scaler.transform(tgt_rel).flatten().astype(np.float32)

        w_arr = np.array(self.w_buf[-self.past_seq:])
        a_arr = np.array(self.angle_buf[-self.past_seq:])
        angle_rel = a_arr - self.base_angle

        pw_sc = self.target_scaler.transform(w_arr).astype(np.float32)
        pa_sc = self.angle_scaler.transform(angle_rel).astype(np.float32)

        return cur_sc, tgt_sc, pw_sc, pa_sc

    def current_pos(self):
        return np.array(self.pos_buf[-1]) if self.pos_buf else None


# ========================================================
# コントローラ推論
# ========================================================
@torch.no_grad()
def infer_w(controller, state_buf, target_pos, device):
    """コントローラから引張量[mm]を推論（次の1ステップ分）"""
    cur_sc, tgt_sc, pw_sc, pa_sc = state_buf.get_ctrl_input(target_pos)

    cur_t = torch.from_numpy(cur_sc).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(tgt_sc).unsqueeze(0).to(device)
    pw_t  = torch.from_numpy(pw_sc).unsqueeze(0).to(device)
    pa_t  = torch.from_numpy(pa_sc).unsqueeze(0).to(device)

    w_norm = controller(cur_t, tgt_t, pw_t, pa_t)          # (1, FUTURE_SEQ, 4)
    w_phys = (w_norm * (W_MAX - W_MIN) + W_MIN).squeeze(0) # (FUTURE_SEQ, 4)
    return w_phys[0].cpu().numpy()                           # 次の1ステップ (4,)


# ========================================================
# 結果ロガー
# ========================================================
class ResultLogger:
    def __init__(self, method_name):
        os.makedirs(RESULT_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(RESULT_DIR, f"{method_name}_{ts}.csv")
        self.rows = []

    def log(self, trial, step, target, current, error, w_cmd):
        self.rows.append({
            "trial": trial, "step": step,
            "tgt_x": target[0],  "tgt_y": target[1],  "tgt_z": target[2],
            "cur_x": current[0], "cur_y": current[1],  "cur_z": current[2],
            "error_3d": error,
            "w0": w_cmd[0], "w1": w_cmd[1], "w2": w_cmd[2], "w3": w_cmd[3],
        })

    def save(self):
        df = pd.DataFrame(self.rows)
        df.to_csv(self.path, index=False)
        print(f"  結果を保存: {self.path}")
        return df


# ========================================================
# テストCSVから目標位置をサンプリング
# ========================================================
def sample_target_positions(n, seed=42,
                             y_abs_max=15.0,
                             z_min=295.0, z_max=340.0):
    csv_files = glob.glob(os.path.join(TEST_CSV_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"テストデータが見つかりません: {TEST_CSV_DIR}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df = df.iloc[CUT_INITIAL_STEPS:].dropna(subset=["X", "Y", "Z"])
        frames.append(df[["X", "Y", "Z"]])
    all_pos = pd.concat(frames, ignore_index=True).values

    # 信頼できる範囲でフィルタリング
    mask = (
        (np.abs(all_pos[:, 1]) <= y_abs_max) &   # |Y| <= 30mm
        (all_pos[:, 2] >= z_min) &                # Z下限
        (all_pos[:, 2] <= z_max)                  # Z上限
    )
    filtered = all_pos[mask]
    print(f"  フィルタ後候補数: {len(filtered)} / {len(all_pos)}")

    if len(filtered) < n:
        print(f"  [Warning] 候補が{n}点未満のため全候補を使用")
        return [filtered[i] for i in range(len(filtered))]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(filtered), size=n, replace=False)
    return [filtered[i] for i in idx]


# ========================================================
# ランダム点追従テスト
# ========================================================
def run_random_test(controller, motor, kf, state_buf,
                    target_positions, method_name, device, cam_thread):
    logger  = ResultLogger(method_name)
    results = []

    for trial_i, target in enumerate(target_positions):
        print(f"\n--- Trial {trial_i+1}/{len(target_positions)} ---")
        print(f"  目標: X={target[0]:.1f}  Y={target[1]:.1f}  Z={target[2]:.1f} mm")

        next_time = time.perf_counter() + INTERVAL


        # カメラスレッド生存確認
        if not cam_thread.is_alive():
            print("\n[Error] カメラスレッドが停止しています。終了します。")
            return logger.save()

        
        errors_all = []
        stable_count = 0
        next_time = time.perf_counter() + INTERVAL

        for step in range(HOLD_STEPS):
            with data_lock:
                pos = shared["pos"]

            if pos is None:
                print(f"\r  [Warning] マーカ未検出 (step={step})", end="", flush=True)
                time.sleep(INTERVAL)
                continue

            pos_f = np.array(kf.update(pos))

            if not motor.communicate():
                print("\n[Error] モータ通信失敗")
                break

            state_buf.push(pos_f, motor.read_angles.copy(),
                           motor.target_pull_mm.copy())

            if state_buf.is_ready():
                w_next = infer_w(controller, state_buf, target, device)
                motor.set_targets(np.clip(w_next, W_MIN, W_MAX).tolist())

            err = np.linalg.norm(pos_f - target)
            errors_all.append(err)

            with data_lock:
                shared["display_info"] = {
                    "trial":   f"{trial_i+1}/{len(target_positions)}",
                    "step":    step,
                    "target":  target,
                    "current": pos_f,
                    "error":   err,
                    "w_cmd":   motor.target_pull_mm.copy(),
                }
            logger.log(trial_i, step, target, pos_f, err,
                       motor.target_pull_mm.copy())

            print(f"\r  step={step:3d}  "
                  f"({pos_f[0]:6.1f},{pos_f[1]:6.1f},{pos_f[2]:6.1f})  "
                  f"err={err:.1f}mm  stable={stable_count}/{STABLE_STEPS}  "
                  f"w=[{motor.target_pull_mm[0]:.1f},{motor.target_pull_mm[1]:.1f},"
                  f"{motor.target_pull_mm[2]:.1f},{motor.target_pull_mm[3]:.1f}]",
                  end="", flush=True)


            now = time.perf_counter()
            if next_time - now > 0:
                time.sleep(next_time - now)
            next_time += INTERVAL

        errors_back_half = [e for s, e in enumerate(errors_all) if s >= HOLD_STEPS // 2]
        final_err   = errors_all[-1] if errors_all else float("nan")
        mean_err    = np.mean(errors_back_half) if errors_back_half else float("nan")
        min_err_val = min(errors_all) if errors_all else float("nan")
        print(f"\n  Trial {trial_i+1} 完了  "
              f"最終誤差: {final_err:.2f}mm  "
              f"後半平均: {mean_err:.2f}mm  "
              f"最小誤差: {min_err_val:.2f}mm")
        results.append((final_err, mean_err, min_err_val))

        motor.set_targets([0.0] * 4)
        print(f"  自然長へ復帰中...")
        for _ in range(10):
            motor.communicate()
            time.sleep(0.5)
        print(f"  復帰完了")

    finals = [r[0] for r in results if not np.isnan(r[0])]
    means  = [r[1] for r in results if not np.isnan(r[1])]
    mins   = [r[2] for r in results if not np.isnan(r[2])]
    print(f"\n{'='*50}")
    print(f"[{method_name}] {len(results)}試行サマリ")
    print(f"  最終誤差   mean={np.mean(finals):.2f}  std={np.std(finals):.2f}  max={np.max(finals):.2f} mm")
    print(f"  後半平均   mean={np.mean(means):.2f}  std={np.std(means):.2f}  max={np.max(means):.2f} mm")
    print(f"  最小誤差   mean={np.mean(mins):.2f}  std={np.std(mins):.2f}  max={np.max(mins):.2f} mm")
    print(f"{'='*50}")

    return logger.save()


# ========================================================
# キーボード手動入力モード
# ========================================================
def run_manual_mode(controller, motor, kf, state_buf, device):
    print("\n=== 手動入力モード ===")
    print("目標座標を入力してください（例: 10 5 280）。'q'で終了。")

    while True:
        try:
            raw = input("Target X Y Z [mm] > ").strip()
        except EOFError:
            break
        if raw.lower() == 'q':
            break

        try:
            vals = list(map(float, raw.split()))
            if len(vals) != 3:
                print("  3つの数値を入力してください")
                continue
            target = np.array(vals)
        except ValueError:
            print("  数値を入力してください")
            continue

        print(f"  目標: {target}  {HOLD_STEPS}ステップ制御します...")
        next_time = time.perf_counter() + INTERVAL

        for step in range(HOLD_STEPS):
            with data_lock:
                pos   = shared["pos"]
                frame = shared["frame_top"]

            if pos is None:
                time.sleep(INTERVAL)
                continue

            pos_f = np.array(kf.update(pos))

            if not motor.communicate():
                print("[Error] モータ通信失敗")
                break

            state_buf.push(pos_f, motor.read_angles.copy(),
                           motor.target_pull_mm.copy())

            if state_buf.is_ready():
                w_next = infer_w(controller, state_buf, target, device)
                motor.set_targets(np.clip(w_next, W_MIN, W_MAX).tolist())

            err = np.linalg.norm(pos_f - target)
            print(f"\r  step={step:3d}  "
                  f"({pos_f[0]:6.1f},{pos_f[1]:6.1f},{pos_f[2]:6.1f})  "
                  f"err={err:.1f}mm", end="")

            print(f"\r  step={step:3d}  "
                  f"({pos_f[0]:6.1f},{pos_f[1]:6.1f},{pos_f[2]:6.1f})  "
                  f"err={err:.1f}mm", end="", flush=True)

            now = time.perf_counter()
            if next_time - now > 0:
                time.sleep(next_time - now)
            next_time += INTERVAL

        print()
        motor.set_targets([0.0] * 4)


# ========================================================
# メイン
# ========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # --- コントローラ選択 ---
    print("\n=== コントローラ選択 ===")
    print("  [A] NARX間接学習コントローラ")
    print("  [B] 直接逆モデルコントローラ")
    choice = input("選択 (A/B) > ").strip().upper()
    if choice == "A":
        ctrl_path   = CTRL_A_PATH
        method_name = "NARX間接"
    else:
        ctrl_path   = CTRL_B_PATH
        method_name = "直接逆モデル"
    print(f"  → {method_name} を使用")

    # --- 実験モード選択 ---
    print("\n=== 実験モード選択 ===")
    print("  [1] ランダム点追従テスト（自動）")
    print("  [2] キーボード手動入力")
    mode = input("選択 (1/2) > ").strip()

    # --- スケーラ・モデル読み込み ---
    target_scaler    = joblib.load(os.path.join(WEIGHTS_DIR, "target_scaler.pkl"))
    angle_scaler     = joblib.load(os.path.join(WEIGHTS_DIR, "angle_scaler.pkl"))
    rel_coord_scaler = joblib.load(os.path.join(WEIGHTS_DIR, "rel_coord_scaler.pkl"))

    controller = ControllerMLP(PAST_SEQ, FUTURE_SEQ).to(device)
    controller.load_state_dict(torch.load(ctrl_path, map_location=device,
                                          weights_only=True))
    controller.eval()
    print(f"コントローラ読み込み完了: {ctrl_path}")

    # --- ハードウェア初期化 ---
    print("\nモータ接続中...")
    motor = MotorController(port="COM3", baudrate=115200)
    if not motor.connect():
        raise RuntimeError("モータ接続失敗")

    print("カメラ初期化中...")
    tracker = StereoTracker(cam_top_idx=1, cam_under_idx=0)
    kf      = KalmanFilter3D(process_noise=1e-4, measurement_noise=0.05)

    print("ホワイトバランス安定待ち（10秒）...")
    time.sleep(10.0)
    
    # 動画保存の準備
    os.makedirs(VIDEO_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = os.path.join(VIDEO_DIR, f"{method_name}_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, SAMPLING_RATE, (1280, 720))
    print(f"動画保存先: {video_path}")

    # カメラスレッド起動
    shared["running"] = True
    cam_thread = threading.Thread(
        target=camera_worker, args=(tracker, video_writer), daemon=True)
    cam_thread.start()

    # 初期エンコーダ値を基準として保存
    motor.communicate()
    base_angle = motor.read_angles.copy()
    print(f"基準エンコーダ値: {base_angle}")

    state_buf = StateBuffer(PAST_SEQ, angle_scaler, target_scaler,
                            rel_coord_scaler, base_angle)

    # バッファウォームアップ（PAST_SEQ分の初期状態を蓄積）
    print(f"バッファウォームアップ中（{PAST_SEQ}ステップ）...")
    wm_next = time.perf_counter() + INTERVAL
    for _ in range(PAST_SEQ):
        with data_lock:
            pos = shared["pos"]
        if pos is None:
            pos = np.zeros(3)
        pos_f = np.array(kf.update(pos))
        motor.communicate()
        state_buf.push(pos_f, motor.read_angles.copy(),
                       motor.target_pull_mm.copy())
        now = time.perf_counter()
        if wm_next - now > 0:
            time.sleep(wm_next - now)
        wm_next += INTERVAL
    print("ウォームアップ完了")

    try:
        if mode == "1":
            print(f"\nテストデータから{N_TRIALS}点をサンプリング...")
            targets = sample_target_positions(N_TRIALS)
            run_random_test(controller, motor, kf, state_buf,
                            targets, method_name, device, cam_thread)
        else:
            run_manual_mode(controller, motor, kf, state_buf, device)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 中断")

    finally:
        print("\n終了処理中...")
        motor.set_targets([0.0] * 4)
        for _ in range(5):
            motor.communicate()
            time.sleep(0.1)
        time.sleep(2.0)

        shared["running"] = False
        cam_thread.join(timeout=2.0)
        motor.disconnect()
        tracker.close()
        cv2.destroyAllWindows()
        print("終了しました。")
        video_writer.release()
        print(f"動画保存完了: {video_path}")


if __name__ == "__main__":
    main()