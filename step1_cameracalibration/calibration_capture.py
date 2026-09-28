
import os
import sys
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

cfg = config.load()

print("start")
# 保存先ディレクトリの設定
save_dir = config.path("calib_images")
# os.makedirs(save_dir, exist_ok=True)

# カメラの初期化（インデックスは config.toml [camera]）
cap_top = cv2.VideoCapture(cfg["camera"]["top_idx"])
cap_under = cv2.VideoCapture(cfg["camera"]["under_idx"])

# 解像度の設定（config.toml [camera]）
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
cap_under.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
cap_under.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

# チェスボードの内側の交点数 (列, 行)（config.toml [chessboard]）
pattern_size = (cfg["chessboard"]["cols"], cfg["chessboard"]["rows"])

count = 0
print("操作方法: [スペースキー] 撮影, [ESCキー] 終了")

while True:
    ret_l, frame_l = cap_top.read()
    ret_r, frame_r = cap_under.read()


    if not ret_l or not ret_r:
        print("カメラからの映像取得に失敗しました．接続を確認してください．")
        break

    # 上カメラ（left）は物理的に上下反転設置のため180度回転して正立に戻す
    frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)

    # 下カメラ180度回転（物理反転の補正）＋両カメラ90度回転
    frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
    frame_l = cv2.rotate(frame_l, cv2.ROTATE_90_CLOCKWISE)
    frame_r = cv2.rotate(frame_r, cv2.ROTATE_90_CLOCKWISE)

    # プレビュー表示用に画像をコピー（保存用には線を描画しないため）
    display_l = frame_l.copy()
    display_r = frame_r.copy()

    # グレースケール変換
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

# チェスボードのコーナー検出（FAST_CHECKを追加）
    flags = cv2.CALIB_CB_FAST_CHECK
    found_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, flags)
    found_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, flags)

    # 認識できた場合はプレビュー画像にカラフルな線を描画
    if found_l:
        cv2.drawChessboardCorners(display_l, pattern_size, corners_l, found_l)
    if found_r:
        cv2.drawChessboardCorners(display_r, pattern_size, corners_r, found_r)

    # 映像の表示
    cv2.imshow('Left Camera', display_l)
    cv2.imshow('Right Camera', display_r)

    key = cv2.waitKey(1) & 0xFF

# 変更後
    if key == 32: 
        if found_l and found_r:
            # 3フレーム連続で両カメラ検出できた最後のフレームを保存（手ブレ軽減）
            stable_count = 0
            stable_l, stable_r = frame_l, frame_r
            for _ in range(3):
                ret_l2, f_l2 = cap_top.read()
                ret_r2, f_r2 = cap_under.read()
                if not ret_l2 or not ret_r2:
                    break
                f_r2 = cv2.rotate(f_r2, cv2.ROTATE_180)
                f_l2 = cv2.rotate(f_l2, cv2.ROTATE_90_CLOCKWISE)
                f_r2 = cv2.rotate(f_r2, cv2.ROTATE_90_CLOCKWISE)
                g_l2 = cv2.cvtColor(f_l2, cv2.COLOR_BGR2GRAY)
                g_r2 = cv2.cvtColor(f_r2, cv2.COLOR_BGR2GRAY)
                ok_l, _ = cv2.findChessboardCorners(g_l2, pattern_size, cv2.CALIB_CB_FAST_CHECK)
                ok_r, _ = cv2.findChessboardCorners(g_r2, pattern_size, cv2.CALIB_CB_FAST_CHECK)
                if ok_l and ok_r:
                    stable_count += 1
                    stable_l, stable_r = f_l2, f_r2

            if stable_count == 3:
                filename_l = os.path.join(save_dir, f"left_{count:02d}.jpg")
                filename_r = os.path.join(save_dir, f"right_{count:02d}.jpg")
                cv2.imwrite(filename_l, stable_l)
                cv2.imwrite(filename_r, stable_r)
                print(f"[{count:02d}] 成功（安定確認済み）: 保存しました．")
                count += 1
            else:
                print("警告: チェスボードが安定していません。静止させてから撮影してください。")
        else:
            print("警告: 両方のカメラでチェスボード全体が認識されていません．")            
    # ESCキーで終了
    elif key == 27: 
        print("撮影を終了します．")
        break

cap_top.release()
cap_under.release()
cv2.destroyAllWindows()