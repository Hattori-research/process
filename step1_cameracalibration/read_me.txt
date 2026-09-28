process/
  step1_cameracalibration/
    stereo_calibrate.py      ← キャリブレーション実行
    calibration_capture.py   ← キャリブ画像撮影
    hsv_tuner.py             ← HSV閾値調整
    check_calibration.py     ← 再投影誤差確認
    stereo_triangulate.py    ← トラッキング本体
    stereo_params.npz        ← キャリブパラメータ