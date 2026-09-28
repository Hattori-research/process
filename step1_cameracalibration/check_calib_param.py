# check_calib_param.py
import cv2
import glob
import numpy as np

images = sorted(glob.glob('calibration_images/left_*.jpg'))
img = cv2.imread(images[0])
print(f"保存済みキャリブ画像 shape: {img.shape}\n")

params = np.load('stereo_params.npz')
print("=== stereo_params.npz の内容 ===")
for key in params.files:
    print(f"\n[{key}]  shape={params[key].shape}")
    print(params[key])