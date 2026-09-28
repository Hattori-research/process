import os
import sys
import serial
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

class MotorController:
    def __init__(self, port=None, baudrate=None, timeout=None):
        mcfg = config.load()["motor"]
        self.port = port if port is not None else mcfg["port"]
        self.baudrate = baudrate if baudrate is not None else mcfg["baudrate"]
        self.timeout = timeout if timeout is not None else mcfg["timeout"]
        self.ser = None

        self.num_motors = 4
        self.diameter = mcfg["pulley_diameter"]
        self.digit = 360.0
        self.max_current = mcfg["max_current"]
        self.return_current = mcfg["return_current"]
        
        # モータの初期位置（絶対角度）を保持
        self.initial_angles = None
        
        self.target_pull_mm = np.zeros(self.num_motors)
        self.currents = np.zeros(self.num_motors)
        self.target_angles = np.zeros(self.num_motors)
        
        self.read_time = 0.0
        self.read_angles = np.zeros(self.num_motors)
        self.read_currents = np.zeros(self.num_motors)
        self.read_targets = np.zeros(self.num_motors)

        self.MAX_PULL = mcfg["max_pull"]
        self.MIN_PULL = mcfg["min_pull"]

    def connect(self):
        """シリアルポートを開き、現在の初期角度を取得する"""
        print(f"Connecting to {self.port}...")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=1)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            read_line = self.ser.readline().decode("utf8").strip()
            if read_line:
                data = read_line.split()
                if len(data) >= 1 + self.num_motors * 3:
                    for i in range(self.num_motors):
                        self.read_angles[i] = float(data[i*3 + 1])
                    
                    # 取得した生の絶対角度を「自然長における初期位置」として記憶
                    self.initial_angles = self.read_angles.copy()
                    self.target_angles = self.initial_angles.copy()
                    print(f"Connected. 初期位置(deg)を取得しました: {self.initial_angles}")
                    return True
                    
            print("[Error] 初期位置の取得に失敗しました。")
            return False
            
        except serial.SerialException as e:
            print(f"\n[Error] Serial connection failed: {e}")
            return False

    def set_targets(self, pull_mm_list, max_current=None):
        """引張量[mm]を受け取り、初期位置からの相対的な絶対角度[deg]に変換してセットする"""
        if self.initial_angles is None:
            return
        if max_current is None:
            max_current = self.max_current

        for i in range(self.num_motors):
            # 安全リミッタ
            safe_pull = max(self.MIN_PULL, min(self.MAX_PULL, pull_mm_list[i]))
            self.target_pull_mm[i] = safe_pull
            self.currents[i] = max_current

            # 引張量(mm)をモータの回転角度(deg)に変換
            angle_change = (safe_pull * self.digit) / (self.diameter * np.pi)

            # 初期位置を基準とし、引張方向（マイナス）へ計算した絶対角度をセット
            self.target_angles[i] = self.initial_angles[i] - angle_change

    def disconnect(self):
        """終了時は自然長（初期位置）に戻してから切断する"""
        if self.ser and self.ser.is_open:
            if self.initial_angles is not None:
                print("モータを初期位置（自然長）へ戻します...")
                # 目標を初期位置（プラス方向）にして送信
                pos_str = ",".join(map(str, self.initial_angles))
                cur_str = ",".join([str(self.return_current)] * self.num_motors)
                stop_msg = f"{cur_str},{pos_str}e\n"
                self.ser.write(stop_msg.encode())
                self.ser.flush()
                time.sleep(2.0) # 巻き戻り待ち
            
            # 最後に電流を0にして切断
            stop_msg = "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0e\n"
            self.ser.write(stop_msg.encode())
            self.ser.flush()
            self.ser.close()
            print("Disconnected.")


    def communicate(self):
        """Arduinoへ絶対角度を送信し、データを受信する"""
        if not self.ser or not self.ser.is_open:
            return False

        # Arduinoへはmmではなく「角度(deg)」を送る
        cur_str = ",".join(map(str, self.currents))
        pos_str = ",".join(map(str, self.target_angles))
        msg = f"{cur_str},{pos_str}e\n"
        
        try:
            self.ser.write(msg.encode())
            self.ser.flush()
            
            read_line = self.ser.readline().decode("utf8").strip()
            if read_line:
                data = read_line.split()
                if len(data) >= 1 + self.num_motors * 3:
                    self.read_time = float(data[0])
                    for i in range(self.num_motors):
                        self.read_angles[i] = float(data[i*3 + 1])
                        self.read_currents[i] = float(data[i*3 + 2])
                        self.read_targets[i] = float(data[i*3 + 3])
            return True
            
        except serial.SerialTimeoutException:
            return False
        except Exception as e:
            print(f"Serial communication error: {e}")
            return False