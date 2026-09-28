# hsv_tuner.py
import cv2
import numpy as np
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def nothing(x):
    pass



cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H_low",  "HSV Tuner",  83, 179, nothing)
cv2.createTrackbar("H_high", "HSV Tuner", 104, 179, nothing)
cv2.createTrackbar("S_low",  "HSV Tuner",  12, 255, nothing)
cv2.createTrackbar("S_high", "HSV Tuner", 255, 255, nothing)
cv2.createTrackbar("V_low",  "HSV Tuner", 121, 255, nothing)
cv2.createTrackbar("V_high", "HSV Tuner", 255, 255, nothing)

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

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 重心計算
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 30:
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

    if cv2.waitKey(1) & 0xFF == 27:
        print(f"\n確定値:")
        print(f"lower_green = np.array([{h_l}, {s_l}, {v_l}], dtype=np.uint8)")
        print(f"upper_green = np.array([{h_h}, {s_h}, {v_h}], dtype=np.uint8)")
        break

cap.release()
cv2.destroyAllWindows()