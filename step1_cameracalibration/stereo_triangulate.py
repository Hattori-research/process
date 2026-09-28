import numpy as np
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

class StereoTracker:
    def __init__(self, cam_top_idx=1, cam_under_idx=0):
        # 左カメラの絶対座標 (X=水平左が正, Y=奥行方向が正, Z=鉛直下が正) [mm]
        CAM_OFFSET = np.array([-78.7,-277.0, 362.7])
        self.FILTER_ALPHA = 0.3
        self.smoothed_pos = None

        # 最新のハンドアイキャリブレーション値を適用
        HAND_EYE_CAL = np.array([0,0,0])
        self.CAM_OFFSET = CAM_OFFSET + HAND_EYE_CAL

        # test_track.py と完全に同じHSV閾値に戻す
        self.lower_green = np.array([83, 12, 121], dtype=np.uint8)
        self.upper_green = np.array([104, 255, 255], dtype=np.uint8)

        self._load_params()
        self._init_cameras(cam_top_idx, cam_under_idx)

    def _load_params(self):
        try:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            params = np.load(os.path.join(base_dir, 'stereo_params.npz'))
            self.M1, self.D1 = params['cameraMatrix1'], params['distCoeffs1']
            self.M2, self.D2 = params['cameraMatrix2'], params['distCoeffs2']
            self.R = params['R']
            self.T = params['T']
            self.R1 = params['R1'] 
            self.R2 = params['R2']
            self.Q        = params['Q']

            # 基線長からスケール誤差を自動補正 (test_track.py と同一)
            # 変更後（28〜42行目）
            ACTUAL_BASELINE = 31.73
            calc_baseline = np.linalg.norm(self.T)
            self.scale_factor = ACTUAL_BASELINE / calc_baseline
            print(f"Calibration parameters loaded successfully. スケール係数 = {self.scale_factor:.3f}")

            # ステレオキャリブレーションで得られたP1, P2を直接使用
            self.P1_calib = params['P1']
            self.P2_calib = params['P2']
        except Exception as e:
            print("Error loading stereo_params.npz. Please recalibrate.")
            raise e

    def _init_cameras(self, idx_l, idx_r):
        self.cap_top = cv2.VideoCapture(idx_l, cv2.CAP_DSHOW)
        self.cap_under = cv2.VideoCapture(idx_r, cv2.CAP_DSHOW)
        
        for cap in (self.cap_top, self.cap_under):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 変更後（55〜67行目）
    def _transform_to_robot_coords(self, cam_x, cam_y, cam_z):
        """カメラローカル座標をロボットの絶対座標（右手系）に変換"""

        # R1はrectified画像用のためここでは使用しない
        # 直接三角測量結果をそのまま使用
        rect_x = cam_x * self.scale_factor
        rect_y = cam_y * self.scale_factor
        rect_z = cam_z * self.scale_factor

        # 軸の入れ替え
        rob_x_base = rect_y   
        rob_y_base = rect_z   
        rob_z_base = rect_x
        
        # カメラの位置オフセットを加算
        rob_x = rob_x_base + self.CAM_OFFSET[0]
        rob_y = rob_y_base + self.CAM_OFFSET[1]
        rob_z = rob_z_base + self.CAM_OFFSET[2]
        
        return np.array([rob_x, rob_y, rob_z])

    def _get_marker_center(self, frame):
        """test_track.py と完全に同一の重心取得アルゴリズム"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 30:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return (cX, cY)
        return None

    def get_3d_coordinates_and_frames(self):
        ret_l, frame_l = self.cap_top.read()
        ret_r, frame_r = self.cap_under.read()

        # カメラそのものの読み取りに失敗した場合だけ全員 None を返す
        if not ret_l or not ret_r:
            return None, None, None

        # 変更後
        # 下カメラ180度回転（物理反転の補正）
        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)

        # 計算用：90度回転（キャリブレーションと同じ前処理）
        # 変更後
        frame_l_calc = cv2.rotate(frame_l, cv2.ROTATE_90_CLOCKWISE)
        frame_r_calc = cv2.rotate(frame_r, cv2.ROTATE_90_CLOCKWISE)
        #print(f"calc shape: {frame_l_calc.shape}")  # 確認用

        #print(f"\r  l_calc:{frame_l_calc.shape}  r_calc:{frame_r_calc.shape}", end="")


        # 変更後
        # --- マーカー検出処理（計算用回転画像を使用）---
        center_l = self._get_marker_center(frame_l_calc)
        center_r = self._get_marker_center(frame_r_calc)

        current_pos = None

        if center_l and center_r:
            # 変更後
            pt_l_raw = np.array([[[center_l[0], center_l[1]]]], dtype=np.float64)
            pt_r_raw = np.array([[[center_r[0], center_r[1]]]], dtype=np.float64)

            # 変更後
            # 正規化座標（歪み補正のみ）
            pt_l_norm = cv2.undistortPoints(pt_l_raw, self.M1, self.D1)[0][0]
            pt_r_norm = cv2.undistortPoints(pt_r_raw, self.M2, self.D2)[0][0]

            disparity_norm = pt_l_norm[0] - pt_r_norm[0]

            if abs(disparity_norm) < 1e-4:
                return None, frame_l, frame_r

            # 正規化座標系での三角測量
            # 基線長が約1.54倍になっています。
            # これはRマトリクスの影響で、2台のカメラが完全に平行でないため
            # 正規化座標の視差にスケール誤差が生じています。
            # 実測値から直接補正係数を求めます。
            # 自然状態でのdisp_norm:+0.1808
            B_eff = 0.1808 * 277.0 
            cam_z = B_eff / abs(disparity_norm)
            cam_x = pt_l_norm[0] * cam_z
            cam_y = pt_l_norm[1] * cam_z

            if abs(disparity_norm) < 0.1:   # 視差が小さすぎる場合はスキップ
                return None, frame_l, frame_r

            raw_pos = self._transform_to_robot_coords(cam_x, cam_y, cam_z)
            
            # 指数移動平均フィルタ
            if self.smoothed_pos is None:
                self.smoothed_pos = raw_pos
            else:
                self.smoothed_pos = self.FILTER_ALPHA * raw_pos + (1.0 - self.FILTER_ALPHA) * self.smoothed_pos
                
            current_pos = self.smoothed_pos

            # マーカーへの赤円描画は表示用フレームに（回転前座標に変換）
            h, w = frame_l.shape[:2]
            # 90度時計回り回転の逆変換: (cx, cy) → (cy, w_calc - cx)
            
            # 変更後
            # top: ROTATE_90_CLOCKWISE の逆変換 (cx,cy) → (cy, w_calc-cx)
            w_calc = frame_l_calc.shape[1]
            disp_l = (int(center_l[1]), int(w_calc - center_l[0]))

            # under: ROTATE_90_COUNTERCLOCKWISE の逆変換 (cx,cy) → (h_calc-cy, cx)
            # 変更後
            w_calc = frame_r_calc.shape[1]
            disp_r = (int(center_r[1]), int(w_calc - center_r[0]))

            cv2.circle(frame_l, disp_l, 7, (0, 0, 255), -1)
            cv2.circle(frame_r, disp_r, 7, (0, 0, 255), -1)
        return current_pos, frame_l, frame_r

    def close(self):
        self.cap_top.release()
        self.cap_under.release()


class KalmanFilter3D:
    def __init__(self, process_noise=1e-4, measurement_noise=0.05):
        # 3軸分独立初期化
        self.q = process_noise       
        self.r = measurement_noise   
        
        self.x = np.zeros((3, 2))    
        self.p = np.eye(2) * 1.0     
        self.f = np.array([[1, 1], [0, 1]]) 
        self.h = np.array([[1, 0]])         
        self.q_mat = np.eye(2) * self.q
        self.r_mat = np.array([[self.r]])

    def update(self, measurements):
        filtered_pos = []
        for i in range(3):
            self.x[i] = np.dot(self.f, self.x[i])
            self.p = np.dot(np.dot(self.f, self.p), self.f.T) + self.q_mat
            
            z = np.array([[measurements[i]]])
            y = z - np.dot(self.h, self.x[i].reshape(2, 1))
            s = np.dot(self.h, np.dot(self.p, self.h.T)) + self.r_mat
            k = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(s))
            
            self.x[i] = self.x[i] + (k @ y).flatten()
            self.p = self.p - k @ self.h @ self.p
            
            filtered_pos.append(self.x[i, 0])
        return filtered_pos


if __name__ == "__main__":
    import time

    print("StereoTracker 動作確認モード")
    print("操作: [ESC] 終了  [r] CAM_OFFSETリセット")

    tracker = StereoTracker(cam_top_idx=1, cam_under_idx=0)
    kf = KalmanFilter3D(process_noise=1e-4, measurement_noise=0.05)

    while True:
        pos, frame_top, frame_under = tracker.get_3d_coordinates_and_frames()

        if frame_top is not None and frame_under is not None:

            # --- 座標テキストをフレームに描画 ---
            if pos is not None:
                filtered = kf.update(pos)
                coord_text = f"X:{filtered[0]:7.1f}  Y:{filtered[1]:7.1f}  Z:{filtered[2]:7.1f} mm"
                raw_text   = f"raw X:{pos[0]:7.1f}  Y:{pos[1]:7.1f}  Z:{pos[2]:7.1f} mm"
                color = (0, 255, 0)
            else:
                coord_text = "marker: NOT FOUND"
                raw_text   = ""
                color = (0, 0, 255)

            for frame in (frame_top, frame_under):
                cv2.putText(frame, coord_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                if raw_text:
                    cv2.putText(frame, raw_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

            # --- 表示 ---
            _, frame_top_calc, frame_under_calc = tracker.get_3d_coordinates_and_frames.__wrapped__ if hasattr(tracker.get_3d_coordinates_and_frames, '__wrapped__') else (None, None, None)

            cv2.imshow("Top Camera",         frame_top)
            cv2.imshow("Under Camera",       frame_under)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            print("\n終了します。")
            break

    tracker.close()
    cv2.destroyAllWindows()