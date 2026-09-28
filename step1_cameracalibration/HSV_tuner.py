# hsv_tuner.py
# 操作: [ESC] 確定して config.toml [marker] に保存して終了 / [q] 保存せず終了
import cv2
import numpy as np
import os
import sys
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

def nothing(x):
    pass


cfg = config.load()
mk  = cfg["marker"]
lo, hi = mk["hsv_lower"], mk["hsv_upper"]   # 現在の設定値を初期値にする

cap = cv2.VideoCapture(cfg["camera"]["top_idx"], cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H_low",  "HSV Tuner", lo[0], 179, nothing)
cv2.createTrackbar("H_high", "HSV Tuner", hi[0], 179, nothing)
cv2.createTrackbar("S_low",  "HSV Tuner", lo[1], 255, nothing)
cv2.createTrackbar("S_high", "HSV Tuner", hi[1], 255, nothing)
cv2.createTrackbar("V_low",  "HSV Tuner", lo[2], 255, nothing)
cv2.createTrackbar("V_high", "HSV Tuner", hi[2], 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 変更後
    frame = cv2.rotate(frame, cv2.ROTATE_180)          # TOP物理反転補正
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # キャリブレーション合わせ

    h_l  = cv2.getTrackbarPos("H_low",  "HSV Tuner")
    h_h  = cv2.getTrackbarPos("H_high", "HSV Tuner")
    s_l  = cv2.getTrackbarPos("S_low",  "HSV Tuner")
    s_h  = cv2.getTrackbarPos("S_high", "HSV Tuner")
    v_l  = cv2.getTrackbarPos("V_low",  "HSV Tuner")
    v_h  = cv2.getTrackbarPos("V_high", "HSV Tuner")

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([h_l, s_l, v_l], dtype=np.uint8),
                       np.array([h_h, s_h, v_h], dtype=np.uint8))

    kernel = np.ones((mk["morph_kernel"], mk["morph_kernel"]), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 重心計算
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > mk["min_contour_area"]:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(frame, f"area={area:.0f} cx={cx} cy={cy}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # マスクをカラー表示
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 変更後
    scale = 0.4  # 表示倍率（小さくしたい場合はさらに下げる）
    disp_frame    = cv2.resize(frame,    (0, 0), fx=scale, fy=scale)
    disp_mask_bgr = cv2.resize(mask_bgr, (0, 0), fx=scale, fy=scale)
    combined = np.hstack([disp_frame, disp_mask_bgr])
    cv2.imshow("HSV Tuner", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print(f"\n確定値:")
        print(f"hsv_lower = [{h_l}, {s_l}, {v_l}]")
        print(f"hsv_upper = [{h_h}, {s_h}, {v_h}]")
        config.update("marker", {"hsv_lower": [h_l, s_l, v_l],
                                 "hsv_upper": [h_h, s_h, v_h]})
        break
    elif key == ord('q'):
        print("\n保存せずに終了します。")
        break

cap.release()
cv2.destroyAllWindows()