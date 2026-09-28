#撮影誤差の確認
import numpy as np
import cv2
import glob

params = np.load('stereo_params.npz')
M1, D1 = params['cameraMatrix1'], params['distCoeffs1']
M2, D2 = params['cameraMatrix2'], params['distCoeffs2']
R, T   = params['R'], params['T']

pattern_size = (7, 10)
square_size  = 12.0
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

images_left  = sorted(glob.glob('calibration_images/left_*.jpg'))
images_right = sorted(glob.glob('calibration_images/right_*.jpg'))

print(f"{'No':>3} {'RMS_L':>7} {'RMS_R':>7}  ファイル名")
print("-" * 55)

errors = []
for i, (pl, pr) in enumerate(zip(images_left, images_right)):
    img_l = cv2.rotate(cv2.imread(pl), cv2.ROTATE_180)  # 反転修正済み
    img_r = cv2.imread(pr)
    gl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rl, cl = cv2.findChessboardCorners(gl, pattern_size, None)
    rr, cr = cv2.findChessboardCorners(gr, pattern_size, None)
    if not rl or not rr:
        continue

    cl = cv2.cornerSubPix(gl, cl, (11,11), (-1,-1), criteria)
    cr = cv2.cornerSubPix(gr, cr, (11,11), (-1,-1), criteria)

    # 左カメラ再投影誤差
    _, rvec, tvec, _ = cv2.solvePnPRansac(objp, cl, M1, D1)
    proj_l, _ = cv2.projectPoints(objp, rvec, tvec, M1, D1)
    err_l = np.sqrt(((cl - proj_l)**2).mean())

    # 右カメラ再投影誤差
    _, rvec, tvec, _ = cv2.solvePnPRansac(objp, cr, M2, D2)
    proj_r, _ = cv2.projectPoints(objp, rvec, tvec, M2, D2)
    err_r = np.sqrt(((cr - proj_r)**2).mean())

    errors.append((err_l + err_r)/2, )
    flag = " <<<" if (err_l > 1.0 or err_r > 1.0) else ""
    print(f"{i:>3} {err_l:>7.3f} {err_r:>7.3f}  {pl}{flag}")

print(f"\n平均再投影誤差: {np.mean(errors):.3f} px")
print(f"1.0px超の枚数: {sum(1 for e in errors if e > 1.0)} / {len(errors)}")