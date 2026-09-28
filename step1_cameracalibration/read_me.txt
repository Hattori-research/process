process/
  config.toml                ← 実測値・パラメータの一括管理（手で編集可）
  config.py                  ← config.toml の読み込み・書き換え
  common/
    stereo_triangulate.py    ← トラッキング本体（StereoTracker, KalmanFilter3D）
    motor_control.py         ← モータ制御（MotorController）
  step1_cameracalibration/
    calibration_capture.py   ← キャリブ画像撮影
    stereo_calibrate.py      ← キャリブレーション実行 → stereo_params.npz 保存、config.toml [calibration] 更新
    HSV_tuner.py             ← HSV閾値調整 → ESC で config.toml [marker] 更新（q で保存せず終了）
    check_calibration.py     ← 再投影誤差確認
    check_calib_param.py     ← stereo_params.npz の中身表示
    stereo_params.npz        ← キャリブパラメータ
