import numpy as np
import os
import sys
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

def open_camera(idx, role):
    """
    config.toml [camera] の設定でカメラを開く（role = "top" / "under"）
      - DSHOW + YUY2（非圧縮）+ FPS 未指定だと 1280x720 で約4fps しか出ないため、形式・FPS を指定する
      - 自動露出だと照明の変化で露出・ゲインが変わり、マーカが検出できなくなる／露出時間が延びて fps が落ちるため、
        露出・ゲイン・ホワイトバランスを固定する（外部アプリで設定した値がカメラに残っていても上書きする）
    """
    c = config.load()["camera"]
    props = [(cv2.CAP_PROP_FRAME_WIDTH, c["width"]),
             (cv2.CAP_PROP_FRAME_HEIGHT, c["height"]),
             (cv2.CAP_PROP_FPS, c["fps"])]
    if c["fourcc"]:
        props.insert(0, (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c["fourcc"])))
    if c["backend"] == "DSHOW":
        # DSHOW はオープン時にまとめて指定しないと形式（MJPG）が反映されない
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW, [v for p in props for v in p])
    elif c["backend"] == "MSMF":
        # MSMF はオープン時の形式指定を受け付けないので、開いてから1項目ずつ設定する
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        for prop, val in props:
            cap.set(prop, val)
    else:
        raise ValueError(f"[camera] backend が不正です: {c['backend']}")

    # 露出・ゲイン・ホワイトバランスの固定（DSHOW は露出値を設定すると手動露出になる）
    # 映像の取得開始時にカメラ側の自動設定に戻されることがあるため、1フレーム読んでから設定する
    # 設定が反映されないことがあるため（C920e）、読み戻して確認し、違っていれば最大5回設定し直す
    if c["fix_exposure"]:
        cap.read()
        fixed = [(cv2.CAP_PROP_AUTO_WB, 0, "WB自動", 0),
                 (cv2.CAP_PROP_WB_TEMPERATURE, c["wb_temperature"], "WB温度", 10),
                 (cv2.CAP_PROP_EXPOSURE, c[f"exposure_{role}"], "露出", 0),
                 (cv2.CAP_PROP_GAIN, c[f"gain_{role}"], "ゲイン", 2)]       # ゲインは内部で ±1 程度丸められる
        for attempt in range(5):
            for prop, val, _, _ in fixed:
                cap.set(prop, val)
            for _ in range(5):
                cap.read()
            mismatch = [f"{name}={cap.get(prop):g}（設定 {val}）" for prop, val, name, tol in fixed
                        if abs(cap.get(prop) - val) > tol]
            if not mismatch:
                break
        if mismatch:
            print(f"[Warning] {role} カメラの設定が反映されていません: {', '.join(mismatch)}")
        elif attempt > 0:
            print(f"  {role} カメラ: 露出・ゲインの設定が {attempt + 1} 回目で反映されました")
    return cap


class StereoTracker:
    def __init__(self, cam_top_idx=None, cam_under_idx=None):
        cfg = config.load()
        tri = cfg["triangulation"]
        mk  = cfg["marker"]
        self.cam_cfg = cfg["camera"]
        if cam_top_idx is None:
            cam_top_idx = self.cam_cfg["top_idx"]
        if cam_under_idx is None:
            cam_under_idx = self.cam_cfg["under_idx"]

        # 三角測量の方式（"calibrated" = R,T を使った三角測量 / "legacy" = 実測 B_eff による旧方式）
        self.method = tri["method"]
        if self.method not in ("calibrated", "legacy"):
            raise ValueError(f"[triangulation] method が不正です: {self.method}")

        # 左カメラの絶対座標 (X=奥行方向が正, Y=水平左が正, Z=鉛直下が正) [mm]（方式ごと）
        self.FILTER_ALPHA = tri["ema_alpha"]
        self.smoothed_pos = None

        # 最新のハンドアイキャリブレーション値を適用
        HAND_EYE_CAL = np.array(tri["hand_eye_cal"])
        self.CAM_OFFSETS = {
            "legacy":     np.array(tri["cam_offset"]) + HAND_EYE_CAL,
            "calibrated": np.array(tri["cam_offset_calibrated"]) + HAND_EYE_CAL,
        }
        self.CAM_OFFSET = self.CAM_OFFSETS[self.method]
        # 実行時フレームをキャリブ時フレームに合わせるための時計回り90°回転の回数
        self.calib_rot_top   = tri["calib_rot_top"]
        self.calib_rot_under = tri["calib_rot_under"]
        self.max_epipolar_px = tri["max_epipolar_px"]

        # 三角測量の実測パラメータ（legacy 用）
        self.actual_baseline   = tri["actual_baseline"]
        self.natural_disp_norm = tri["natural_disp_norm"]
        self.natural_depth     = tri["natural_depth"]
        self.min_disp_norm     = tri["min_disp_norm"]

        # 診断用：直近フレームの両方式の結果（フィルタ前、ロボット座標）とエピポーラ誤差
        self.last_raw = {"legacy": None, "calibrated": None}
        self.last_epipolar_px = None
        self.last_centers = (None, None)

        # マーカ検出パラメータ
        self.lower_green = np.array(mk["hsv_lower"], dtype=np.uint8)
        self.upper_green = np.array(mk["hsv_upper"], dtype=np.uint8)
        self.morph_kernel     = mk["morph_kernel"]
        self.min_contour_area = mk["min_contour_area"]

        self._load_params()
        self._init_cameras(cam_top_idx, cam_under_idx)

    def _load_params(self):
        try:
            params = np.load(config.path("stereo_params"))
            self.M1, self.D1 = params['cameraMatrix1'], params['distCoeffs1']
            self.M2, self.D2 = params['cameraMatrix2'], params['distCoeffs2']
            self.R = params['R']
            self.T = params['T']
            self.R1 = params['R1'] 
            self.R2 = params['R2']
            self.Q  = params['Q']

            # 基線長からスケール誤差を自動補正 (test_track.py と同一)
            # 変更後（28〜42行目）
            ACTUAL_BASELINE = self.actual_baseline
            calc_baseline = np.linalg.norm(self.T)
            self.scale_factor = ACTUAL_BASELINE / calc_baseline
            print(f"Calibration parameters loaded successfully. スケール係数 = {self.scale_factor:.3f}")

            # ステレオキャリブレーションで得られたP1, P2を直接使用
            self.P1_calib = params['P1']
            self.P2_calib = params['P2']

            # calibrated 方式用：正規化座標での投影行列と基本行列
            T = self.T.reshape(3, 1)
            self.P_left  = np.hstack([np.eye(3), np.zeros((3, 1))])
            self.P_right = np.hstack([self.R, T])
            tx = np.array([[0, -T[2, 0], T[1, 0]],
                           [T[2, 0], 0, -T[0, 0]],
                           [-T[1, 0], T[0, 0], 0]])
            self.E = tx @ self.R
        except Exception as e:
            print("Error loading stereo_params.npz. Please recalibrate.")
            raise e

    def _init_cameras(self, idx_l, idx_r):
        # バックエンド・形式・FPS・露出などは config.toml [camera]
        self.cap_top = open_camera(idx_l, "top")
        self.cap_under = open_camera(idx_r, "under")

# 変更後（55〜67行目）
    def _transform_to_robot_coords(self, cam_x, cam_y, cam_z, method=None):
        """カメラローカル座標をロボットの絶対座標（右手系）に変換"""
        offset = self.CAM_OFFSETS[method] if method else self.CAM_OFFSET

        # R1はrectified画像用のためここでは使用しない
        # 直接三角測量結果をそのまま使用
        rect_x = cam_x * self.scale_factor
        rect_y = cam_y * self.scale_factor
        rect_z = cam_z * self.scale_factor

        # 軸の入れ替え
        rob_x_base = rect_z
        rob_y_base = -rect_y   
        rob_z_base = -rect_x
        
        # カメラの位置オフセットを加算
        rob_x = rob_x_base + offset[0]
        rob_y = rob_y_base + offset[1]
        rob_z = rob_z_base + offset[2]
        
        return np.array([rob_x, rob_y, rob_z])

    def _get_marker_center(self, frame):
        """test_track.py と完全に同一の重心取得アルゴリズム"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > self.min_contour_area:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    # サブピクセル精度の重心（legacy 方式では従来どおり int に切り捨てて使う）
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    return (cX, cY)
        return None

    @staticmethod
    def _rotate_pt_cw(pt, shape, k):
        """画像を時計回りに90°×k 回転したときの画素座標を返す（shape は回転前の画像）"""
        u, v = pt
        h, w = shape[:2]
        for _ in range(k % 4):
            u, v = h - 1 - v, u
            h, w = w, h
        return np.array([u, v], dtype=np.float64)

    @staticmethod
    def _rotate_cam_cw(p, k):
        """画像を時計回りに90°×k 回転したときのカメラ座標 (x, y, z) を返す"""
        x, y, z = p
        for _ in range(k % 4):
            x, y = -y, x
        return np.array([x, y, z])

    def _triangulate_legacy(self, center_l, center_r):
        """旧方式：実行時フレームの正規化座標の視差と実測 B_eff から奥行きを求める（カメラ座標を返す）"""
        pt_l_raw = np.array([[[int(center_l[0]), int(center_l[1])]]], dtype=np.float64)
        pt_r_raw = np.array([[[int(center_r[0]), int(center_r[1])]]], dtype=np.float64)

        # 正規化座標（歪み補正のみ）
        pt_l_norm = cv2.undistortPoints(pt_l_raw, self.M1, self.D1)[0][0]
        pt_r_norm = cv2.undistortPoints(pt_r_raw, self.M2, self.D2)[0][0]

        disparity_norm = pt_l_norm[0] - pt_r_norm[0]

        # 視差が小さすぎる場合はスキップ
        if abs(disparity_norm) < 1e-4 or abs(disparity_norm) < self.min_disp_norm:
            return None

        # 実行時フレームにキャリブ時の内部パラメータを当てているため視差にスケール誤差が出る。
        # 自然状態の実測値（config.toml [triangulation]）で補正する
        B_eff = self.natural_disp_norm * self.natural_depth
        cam_z = B_eff / abs(disparity_norm)
        cam_x = pt_l_norm[0] * cam_z
        cam_y = pt_l_norm[1] * cam_z
        return np.array([cam_x, cam_y, cam_z])

    def _triangulate_calibrated(self, center_l, center_r, shape_l, shape_r):
        """
        新方式：重心をキャリブ時フレームの画素座標に戻し、R,T を使って三角測量する。
        返り値は実行時フレームの向きに合わせたカメラ座標（legacy と同じ軸）。
        """
        ql = self._rotate_pt_cw(center_l, shape_l, self.calib_rot_top)
        qr = self._rotate_pt_cw(center_r, shape_r, self.calib_rot_under)
        nl = cv2.undistortPoints(ql.reshape(1, 1, 2), self.M1, self.D1).reshape(2)
        nr = cv2.undistortPoints(qr.reshape(1, 1, 2), self.M2, self.D2).reshape(2)

        # エピポーラ誤差（Sampson 距離を画素換算）：左右で別の物体を拾った場合などを除外
        xl, xr = np.array([nl[0], nl[1], 1.0]), np.array([nr[0], nr[1], 1.0])
        Ex, Etx = self.E @ xl, self.E.T @ xr
        denom = np.sqrt(Ex[0]**2 + Ex[1]**2 + Etx[0]**2 + Etx[1]**2)
        self.last_epipolar_px = abs(xr @ self.E @ xl) / denom * self.M1[0, 0]
        if self.last_epipolar_px > self.max_epipolar_px:
            return None

        X4 = cv2.triangulatePoints(self.P_left, self.P_right,
                                   nl.reshape(2, 1), nr.reshape(2, 1))
        X = (X4[:3] / X4[3]).ravel()
        if X[2] <= 0:
            return None
        # キャリブ時フレーム → 実行時フレームの軸（キャリブ時 = 実行時を k 回回転 → 逆に 4-k 回）
        return self._rotate_cam_cw(X, 4 - self.calib_rot_top)

    def get_3d_coordinates_and_frames(self):
        self.cap_top.grab()
        self.cap_under.grab()
        ret_l, frame_l = self.cap_top.retrieve()
        ret_r, frame_r = self.cap_under.retrieve()

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
        self.last_centers = (center_l, center_r)   # 診断用

        current_pos = None

        self.last_raw = {"legacy": None, "calibrated": None}
        self.last_epipolar_px = None

        if center_l and center_r:
            # 両方式で計算（診断用に last_raw に残す）。採用するのは self.method の方
            cam_legacy = self._triangulate_legacy(center_l, center_r)
            cam_calib  = self._triangulate_calibrated(center_l, center_r,
                                                      frame_l_calc.shape, frame_r_calc.shape)
            if cam_legacy is not None:
                self.last_raw["legacy"] = self._transform_to_robot_coords(*cam_legacy, method="legacy")
            if cam_calib is not None:
                self.last_raw["calibrated"] = self._transform_to_robot_coords(*cam_calib, method="calibrated")

            raw_pos = self.last_raw[self.method]
            if raw_pos is None:
                return None, frame_l, frame_r

            # 指数移動平均フィルタ
            if self.smoothed_pos is None:
                self.smoothed_pos = raw_pos
            else:
                self.smoothed_pos = self.FILTER_ALPHA * raw_pos + (1.0 - self.FILTER_ALPHA) * self.smoothed_pos
                
            current_pos = self.smoothed_pos

            # マーカーへの赤円描画は表示用フレームに（回転前座標に変換）
            h, w = frame_l.shape[:2]
            # 90度時計回り回転の逆変換: (cx, cy) → (cy, w_calc - cx)
            
            # top: ROTATE_90_CLOCKWISE の逆変換 (cx,cy) → (cy, w_calc-cx)
            w_calc = frame_l_calc.shape[1]
            disp_l = (int(center_l[1]), int(w_calc - int(center_l[0])))

            # under: 同上（frame_r は180°回転済みの表示用フレーム）
            w_calc = frame_r_calc.shape[1]
            disp_r = (int(center_r[1]), int(w_calc - int(center_r[0])))

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
        
        self.x = np.zeros((3, 2))                    # 各軸 [位置, 速度]
        self.p = np.stack([np.eye(2) * 1.0] * 3)     # 各軸の共分散 (3, 2, 2)
        self.f = np.array([[1, 1], [0, 1]])
        self.h = np.array([[1, 0]])
        self.q_mat = np.eye(2) * self.q
        self.r_mat = np.array([[self.r]])
        self.initialized = False

    def update(self, measurements):
        # 初回は観測値で位置を初期化（0から立ち上がる過渡応答を防ぐ）
        if not self.initialized:
            self.x[:, 0] = measurements[:3]
            self.x[:, 1] = 0.0
            self.initialized = True
            return [self.x[i, 0] for i in range(3)]

        filtered_pos = []
        for i in range(3):
            self.x[i] = np.dot(self.f, self.x[i])
            self.p[i] = np.dot(np.dot(self.f, self.p[i]), self.f.T) + self.q_mat

            z = np.array([[measurements[i]]])
            y = z - np.dot(self.h, self.x[i].reshape(2, 1))
            s = np.dot(self.h, np.dot(self.p[i], self.h.T)) + self.r_mat
            k = np.dot(np.dot(self.p[i], self.h.T), np.linalg.inv(s))

            self.x[i] = self.x[i] + (k @ y).flatten()
            self.p[i] = self.p[i] - k @ self.h @ self.p[i]

            filtered_pos.append(self.x[i, 0])
        return filtered_pos


if __name__ == "__main__":
    import time

    print("StereoTracker 動作確認モード")
    print("操作: [ESC] 終了  [r] CAM_OFFSETリセット")

    kcfg = config.load()["kalman"]
    tracker = StereoTracker()
    kf = KalmanFilter3D(process_noise=kcfg["process_noise_realtime"],
                        measurement_noise=kcfg["measurement_noise"])

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