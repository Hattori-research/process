import time
import threading
import csv
import datetime
import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import config
from common.motor_control import MotorController
from common.stereo_triangulate import StereoTracker, KalmanFilter3D

shared_data = {
    "pos": None,
    "frame_r": None,
    "running": True
}
data_lock = threading.Lock()

def camera_thread_worker(tracker):
    """カメラトラッキングをバックグラウンドで常時回すスレッド"""
    while shared_data["running"]:
        pos, frame_l, frame_r = tracker.get_3d_coordinates_and_frames()
        with data_lock:
            shared_data["pos"] = pos
            shared_data["frame_r"] = frame_r

def create_video_writer(fps=20.0, width=1280, height=720):
    """動画保存用のVideoWriterを生成するヘルパー関数"""
    timestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(config.path("mp4_dir"), f"{timestr}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(f"[Video] 録画を開始します: {filename}")
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))

def main():
    global shared_data

    # 保存先ディレクトリの確保（サブディレクトリもすべて作成）
    os.makedirs(config.path("mp4_dir"), exist_ok=True)
    os.makedirs(config.path("png_dir"), exist_ok=True)
    os.makedirs(config.path("rnn_csv_dir"), exist_ok=True)

    cfg  = config.load()
    scfg = cfg["sampling"]
    kcfg = cfg["kalman"]

    # --- サンプリング設定（config.toml [system] [sampling]）---
    SAMPLING_RATE = cfg["system"]["sampling_rate"]
    INTERVAL = 1.0 / SAMPLING_RATE
    COMMAND_INTERVAL = scfg["command_interval"]

    TARGET_SAMPLES = scfg["target_samples"]
    collected_samples = 0

    #--引張量の設定
    under_limit = scfg["pull_min"]
    upper_limit = scfg["pull_max"]
    total_max = scfg["pull_total_max"]

    # --- 画像/動画保存設定 ---
    SAVE_MODE = scfg["save_mode"]                 # "mp4" または "png" で切り替え
    SAVE_INTERVAL_SEC = scfg["save_interval_sec"] # 保存間隔 [s]
    video_out = None
    last_save_time = time.time()

    # csv保存先
    timestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(config.path("rnn_csv_dir"),
                                f"{under_limit}_{upper_limit}_{total_max}_{timestr}.csv")


    kf = KalmanFilter3D(process_noise=kcfg["process_noise_sampling"],
                        measurement_noise=kcfg["measurement_noise"])

    # 初回のみCSVのヘッダーを書き込む
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ["Sample_Count", "Time"] + \
                  [f"Target_W{i}" for i in range(4)] + \
                  [f"Angle_W{i}" for i in range(4)] + \
                  [f"Cur_W{i}" for i in range(4)] + \
                  ["X", "Y", "Z"]
        writer.writerow(headers)

    print(f"Data sampling started. Target: {TARGET_SAMPLES} samples. Saving to {csv_filename}...")
    print(f"Save Mode: {SAVE_MODE.upper()} (Interval: {SAVE_INTERVAL_SEC} sec)")

    # =========================================================
    # 自動リカバリ用のアウターループ（目標データ数に達するまで回る）
    # =========================================================
    while collected_samples < TARGET_SAMPLES:
        print(f"\n=== システム初期化 (現在 {collected_samples}/{TARGET_SAMPLES} サンプル) ===")
        
        total_tensile = np.zeros(4)

        motor = MotorController()
        if not motor.connect():
            print("[Warning] シリアル通信の接続に失敗しました。5秒後に再試行します...")
            time.sleep(5.0)
            continue
        
        print("カメラを初期化しています...")
        try:
            tracker = StereoTracker()
        except Exception as e:
            print(f"[Warning] カメラの初期化に失敗しました: {e}。5秒後に再試行します...")
            motor.disconnect()
            time.sleep(5.0)
            continue

        #ここの時間を長くした
        time.sleep(scfg["camera_warmup_sec"])
        shared_data["running"] = True
        cam_thread = threading.Thread(target=camera_thread_worker, args=(tracker,), daemon=True)
        cam_thread.start()
        
        last_command_time = 0
        start_time = time.perf_counter()
        next_time = start_time + INTERVAL
        
        fail_count = 0
        last_save_time = time.time()
        
        try:
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                while collected_samples < TARGET_SAMPLES:
                    current_time = time.perf_counter() - start_time
                    absolute_time = time.time()
                    
                    # --- 【指令更新】2秒ごとに相対引張量を計算 ---
                    if current_time - last_command_time >= COMMAND_INTERVAL:
                        w_idx = np.random.randint(0, 4)
                        pull_amount = np.random.uniform(under_limit, upper_limit)
                        
                        if total_tensile[w_idx] + pull_amount >= total_max:
                            print(f"\n[{current_time:.1f}s] W{w_idx}が限界到達！ゼロにして自然長に戻します。")
                            total_tensile = np.zeros(4)
                        else:
                            total_tensile[w_idx] += pull_amount
                            print(f"\n[{current_time:.1f}s] W{w_idx} を追加引張(+{pull_amount:.1f}mm) -> トータル: {total_tensile.round(1)}mm")

                        motor.set_targets(total_tensile.tolist())
                        last_command_time = current_time

                    # --- 【サンプリング】sampling_rate [Hz] で通信と記録 ---
                    if motor.communicate():
                        fail_count = 0 
                        with data_lock:
                            pos = shared_data["pos"]
                            frame_r = shared_data["frame_r"]

                        if pos is not None:
                            # カルマンフィルタで座標を滑らかにする
                            filtered_coords = kf.update(pos)
                            pos_filtered = np.array([filtered_coords[0], filtered_coords[1], filtered_coords[2]])
                                
                            row = [collected_samples, current_time] + \
                                  list(motor.target_pull_mm) + \
                                  list(motor.read_angles) + \
                                  list(motor.read_currents) + \
                                  list(pos_filtered)
                            writer.writerow(row)
                            collected_samples += 1
                            
                            print(f"\rSample:{collected_samples}/{TARGET_SAMPLES} | Tgt:{total_tensile.round(1)} | X:{pos_filtered[0]:6.1f} Y:{pos_filtered[1]:6.1f} Z:{pos_filtered[2]:6.1f}", end="")
                            
                            # ★ 画面表示・保存用のフレームにカルマンフィルタ後の正確な座標を上書き
                            if frame_r is not None:
                                text = f"X:{pos_filtered[0]:.1f} Y:{pos_filtered[1]:.1f} Z:{pos_filtered[2]:.1f}"
                                cv2.putText(frame_r, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            print(f"\r[Warning] マーカ見失い中... (Sample:{collected_samples}){' '*20}", end="")
                    else:
                        fail_count += 1
                        if fail_count > 10:
                            print("\n[Error] Arduinoとの通信途絶を検知しました。システムを安全に再起動します。")
                            break 
                    
                    # --- 映像表示と録画/画像保存処理 ---
                    if shared_data["frame_r"] is not None:
                        current_frame = shared_data["frame_r"]
                        cv2.imshow('Right Tracking (Preview)', current_frame)
                        
                        if SAVE_MODE == "mp4":
                            if video_out is None:
                                h, w, _ = current_frame.shape
                                video_out = create_video_writer(fps=SAMPLING_RATE, width=w, height=h)
                                last_save_time = absolute_time

                            video_out.write(current_frame)
                                
                            if absolute_time - last_save_time >= SAVE_INTERVAL_SEC:
                                print("\n[Video] 20分経過。動画ファイルをローテーションします。")
                                if video_out is not None:
                                    video_out.release()
                                h, w, _ = current_frame.shape
                                video_out = create_video_writer(fps=SAMPLING_RATE, width=w, height=h)
                                last_save_time = absolute_time
                        
                        elif SAVE_MODE == "png":
                            if absolute_time - last_save_time >= SAVE_INTERVAL_SEC:
                                timestr_img = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                img_filename = os.path.join(config.path("png_dir"), f"{timestr_img}.png")
                                cv2.imwrite(img_filename, current_frame)
                                print(f"\n[Image] 20分経過。スナップショットを保存しました: {img_filename}")
                                last_save_time = absolute_time

                    if cv2.waitKey(1) & 0xFF == 27:
                        print("\n[ESC] ユーザーによって中断されました。")
                        TARGET_SAMPLES = collected_samples 
                        break
                    
                    now = time.perf_counter()
                    sleep_time = next_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    next_time += INTERVAL
                    
        except KeyboardInterrupt:
            print("\nSampling interrupted by user.")
            TARGET_SAMPLES = collected_samples 
        
        finally:
            print("\nシステムを一時停止し、蓄積した引張量をマイナスして戻しています...")
            motor.set_targets([0.0, 0.0, 0.0, 0.0])
            for _ in range(5):
                motor.communicate()
                time.sleep(0.1)
                
            print("モーターの物理的な巻き戻り・解放待ち...")
            time.sleep(2.0) 

            shared_data["running"] = False
            cam_thread.join(timeout=1.0)
            motor.disconnect()
            
            try:
                tracker.close()
            except:
                pass
            cv2.destroyAllWindows()
            
            if SAVE_MODE == "mp4" and video_out is not None:
                video_out.release()
                video_out = None
            
            if collected_samples < TARGET_SAMPLES:
                print("\n3秒後にシステムの再構築とデータ収集を再開します...")
                time.sleep(3.0)

    print(f"\n=== 全データ収集完了 (Total: {collected_samples} samples) ===")

if __name__ == "__main__":
    main()