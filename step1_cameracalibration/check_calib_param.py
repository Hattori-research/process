# check_calib_param.py
import cv2
import glob
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

images = sorted(glob.glob(os.path.join(config.path("calib_images"), 'left_*.jpg')))
img = cv2.imread(images[0])
print(f"保存済みキャリブ画像 shape: {img.shape}\n")

params = np.load(config.path("stereo_params"))
print("=== stereo_params.npz の内容 ===")
for key in params.files:
    print(f"\n[{key}]  shape={params[key].shape}")
    print(params[key])