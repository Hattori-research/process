import cv2
import numpy as np
import glob

# チェスボードの設定
pattern_size = (10, 7)
square_size = 12.00  # mm

# 3D空間のコーナー座標を準備 (0,0,0), (23.2,0,0), (46.4,0,0) ...
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# 画像上の2Dポイントと3Dポイントを保存するリスト
objpoints = []   # 現実世界の3Dポイント
imgpoints_l = [] # 左カメラの画像平面上の2Dポイント
imgpoints_r = [] # 右カメラの画像平面上の2Dポイント

# 撮影した画像のパス（保存先に合わせて調整してください）
images_left = sorted(glob.glob('calibration_images/left_*.jpg'))
images_right = sorted(glob.glob('calibration_images/right_*.jpg'))

print(f"見つかった画像ペア: {len(images_left)}セット")

img_shape = None

# 画像を1枚ずつ読み込んでコーナーを検出
for img_l_path, img_r_path in zip(images_left, images_right):

    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    
    if img_l is None or img_r is None:
        continue

# 変更後
    # 上カメラ180度回転（物理反転の補正）
    img_l = cv2.rotate(img_l, cv2.ROTATE_180)

    # img_lは時計回り、img_rは反時計回りで試す
    img_l = cv2.rotate(img_l, cv2.ROTATE_90_CLOCKWISE)
    img_r = cv2.rotate(img_r, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 変更後
    if img_shape is None:
        # gray_l.shape = (height, width)
        # img_shapeはOpenCVの(width, height)形式で渡す
        h, w = gray_l.shape[:2]
        img_shape = (w, h)  # 縦長画像なら(720, 1280)になる
        print(f"img_shape = {img_shape}  (w={w}, h={h})")

    # チェスボードのコーナー検出
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

    # 両方のカメラで認識できた場合のみリストに追加
    if ret_l and ret_r:
        objpoints.append(objp)

        # サブピクセル精度に細線化（より高精度な座標計算のため）
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

print(f"有効なキャリブレーションペア: {len(objpoints)}セット")

if len(objpoints) > 0:
    print("ステレオキャリブレーションを実行中...")
    
    # 1. 個別のカメラキャリブレーション（ステレオ計算の初期値として使用）
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    # 2. ステレオキャリブレーション
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_shape, criteria=criteria_stereo, flags=flags)

    print(f"ステレオキャリブレーションRMS誤差: {ret_stereo:.4f} ピクセル")

    # 3. ステレオ平行化（Rectification）パラメータの計算
# 変更後
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=(1280, 720))

    # キャリブレーション結果としてTベクトルを表示（垂直配置の確認用）
    print(f"並進ベクトルT (mm): {T.flatten()}")
    print(f"  Tx={T[0][0]:.2f}  Ty={T[1][0]:.2f}  Tz={T[2][0]:.2f}")
    print(f"  ※上下配置の場合 Ty≈-31mm, Tx≈0, Tz≈0 になれば正常")

    # 4. 後続のトラッキングで使用するためにパラメータを保存
    np.savez("stereo_params.npz",
             cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
             cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             roi_left=roi_left, roi_right=roi_right)
             
    print("すべてのパラメータを 'stereo_params.npz' に保存しました．")
else:
    print("有効な画像ペアがありませんでした．撮影データを確認してください．")