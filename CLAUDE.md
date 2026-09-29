# プロジェクト概要
宇都宮大学 バイオニクス研究室 修士研究
「ユーグレナの表皮帯を規範とした複合滑走構造による高自由度変形機構」

## 実機
- ワイヤアンカーを備えた繊維を放射状に4本配置し、アンカー内のワイヤで繊維に力を伝達
- 各繊維のガイドパーツが隣接繊維との間で滑走して変形する
- 繊維長200mm、フレーム 350×350×445.5mm、機構先端にマーカ
- ステレオカメラ2台でマーカ位置を取得、モータ4つでワイヤを引張、エンコーダ値を取得
- サンプリングレート 16Hz（2026-09-29 に 20Hz から変更。カメラ取得が約16fps のため）

## 制御手法（参考: Thuruthel et al., IEEE T-RO, 2019, doi:10.1109/TRO.2018.2878318）
1. データ取得
2. RNNで順モデルを学習
   入力: 位置 x(i-0〜39), エンコーダ値 A(i-0〜39), 引張量 W(i+1〜20)
   ※ 16Hz 化にあたり秒数を維持（過去2秒・未来1秒）→ 実装は x,A(i-0〜31), W(i+1〜16)
   出力: 位置 x(i+1〜10)
3. 順モデルとランダムシューティング法（2000シューティング×5000回）で逆方向のデータセットを5000個作成
4. MLPコントローラを学習
   入力: 位置変化量Δx と エンコーダ値 A（計7次元）
   構造: 
   出力: 4本のワイヤ引張量 W（4次元）
システム: 目標位置 → MLP → モータ指令/エンコーダ値 → カメラで現在位置取得 → フィードバック

# ディレクトリ規則
process/stepN_<内容>/<スクリプト>.py の形でステップごとに分ける。
例: step1_cameracalibration/stereo_calibrate.py
新しいステップを作るときは、命名案を提示して確認を取ってから作成する。
複数ステップで使うモジュールは process/common/ に置く（stereo_triangulate.py, motor_control.py, models.py）。

# パラメータ管理（config.toml）
- 実測値・ハードコード値は process/config.toml に集約し、コードに直書きしない
- 読み込み: `config.load()`（dict）/ パス: `config.path("<[paths]のキー>")`（process/ 基準の絶対パス）
- 書き換え: `config.update("<section>", {key: value})`（コメント・並び順を保持し、TOMLとして正しいか検証してから保存）
- スクリプトが自動で書き換える項目にはコメントに `[auto]` を付ける（例: HSV_tuner.py → [marker]、stereo_calibrate.py → [calibration]）
- config.toml は今後も頻繁に書き換わるので、編集前に必ず最新の内容を読み直す。キー名を変えたら参照箇所をすべて検索して揃える
- 各スクリプトの冒頭で `sys.path.insert(0, <process/>)` → `import config` / `from common.xxx import ...` とする
- 移行状況: 全ステップ（step1〜5）と common は移行済み
- [model] [narx] [controller] の構造に関わる値（past_seq, lstm_hidden, hidden_size 等）を変えると、既存の学習済み重みは読み込めなくなる（再学習が必要）

# 作業ルール
- 応答は日本語
- コードを変更したら、ファイル名と変更した行番号（何行目〜何行目）を報告する
- カメラ・モータを動かすスクリプトは勝手に実行しない。実行は服部が行う

# 現在の実装状況（2026-09-28 時点のコードから整理）

## 共通事項
- パスはすべて config.toml [paths] から解決するので、カレントディレクトリに依存しない
- `Data/` は .gitignore 対象。構成: `rnn_csv/{train,test}/`, `Weights/`, `mp4/`, `png/`, `control_results/{,video}/`
- ハードウェア設定（config.toml [camera] [motor]）: モータ `COM3` / 115200bps、カメラ index 上=1・下=0、1280×720
- `TrajectoryNet` / `ControllerMLP` は common/models.py に一本化（step3・4・5・check_narx から import）
- 共通ハイパーパラメータ（config.toml [model]）: `past_seq=32`（2秒）, `future_seq=16`（1秒）, `w_min/w_max=0/16mm`, `cut_initial_steps=80`（5秒）, `movement_limit=120mm`, `jump_thresh=150mm`

## step1_cameracalibration（カメラ校正・3次元計測）
| ファイル | 内容 |
|---|---|
| calibration_capture.py | チェスボード(10×7)画像を撮影し `calibration_images/` に保存（3フレーム安定確認） |
| stereo_calibrate.py | ステレオ校正（マス12mm）→ `stereo_params.npz` 保存、RMS・基線長・日時を config.toml [calibration] に記録 |
| check_calibration.py / check_calib_param.py | 再投影誤差の確認 / npz の中身表示 |
| HSV_tuner.py | マーカ（緑）のHSV閾値調整。ESC で config.toml [marker] に保存、q で保存せず終了 |
| common/stereo_triangulate.py | `StereoTracker`（マーカ重心→3次元座標）と `KalmanFilter3D`。step2/5 から import される |

- 座標系: X=奥行が正, Y=水平左が正, Z=鉛直下が正 [mm]。方式ごとのカメラオフセット（`cam_offset` / `cam_offset_calibrated`）を加算
- 三角測量の方式は config.toml `[triangulation] method` で切り替え（2026-09-28〜 既定は calibrated）
  - **calibrated**: 実行時フレーム（縦長）の重心をキャリブ時フレーム（横長）に戻し（`calib_rot_top/under`=3、エピポーラ誤差 0.17px で実測判定）、R,T で `cv2.triangulatePoints`。重心はサブピクセル。エピポーラ誤差 > `max_epipolar_px` の検出は破棄
  - **legacy**: 実行時フレームにキャリブ時の内部パラメータをそのまま当て、実測 `B_eff = natural_disp_norm(0.1808) * natural_depth(277.0)` で奥行きを補正する旧方式。主点ずれにより奥行きで数十mm、軸間の混ざりも大きい（Z を 10mm 動かすと X が 8mm 動く等）。2026-09-28 以前に取得したデータはこの方式
  - **ロボット座標の原点 = 自然状態のマーカ先端**（2026-09-28 決定）。`cam_offset_calibrated` は check_triangulation.py の [o] で自然状態から自動設定する
  - 比較・定規検証ツール: step1_cameracalibration/check_triangulation.py（[SPACE] 記録、[o] 原点設定、`--headless N` で統計のみ）。平均中の std > `[triangulation_check] max_std` なら「動いている」として記録しない
  - 実機確認（2026-09-28）: legacy は奥行きを縮めて出す（奥行き 310mm で −15mm、370mm 付近で約 −48mm）。calibrated の静止時 std は奥行き 0.5mm・他 0.04mm 以下
  - チェスボード検証（2026-09-28, check_chessboard_accuracy.py, 奥行き 266〜387mm, 45フレーム）:
    calibrated は隣接間隔 12.05〜12.14mm（真 12）、端から端 108.1〜108.6 / 71.0〜72.3mm（真 108 / 72）、平面RMS 0.6〜1.1mm。
    legacy は形状が大きく歪む（端から端 105〜140 / 80〜94mm、平面RMS 18〜23mm）→ **calibrated が正しい。基線長 31.73mm のままで精度は十分（治具の作り直しは不要）**
  - 動いている物体では左右カメラの取得タイミング差でエピポーラ誤差が 1〜数px に増える（静止時 0.1〜0.3px）
- カメラ取得レート（2026-09-28 計測、config.toml [camera] backend/fourcc/fps）:
  - 旧設定 DSHOW + YUY2 + FPS未指定 → **約3.8fps**（2026-09-28 以前のデータは 20Hz 記録でも位置の更新は約4Hz）
  - DSHOW + MJPG + FPS30（現設定）→ 約16fps、エピポーラ誤差 0.23px、外れなし。形式による位置の偏りなし
  - MSMF + MJPG → 24〜30fps だが、静止マーカでもエピポーラ外れが約1割、オープン時に固まることがあり不採用
- 自然状態のマーカ位置は、原点設定後の数分で奥行き方向に数mm ずれる（機構のクリープ）
- 指数移動平均（α=0.3）を内部でかけ、呼び出し側でさらにカルマンフィルタをかけている（二重平滑化）
- 注意点:
  - 画像の回転処理はスクリプト間で異なる（capture は保存前に回転済み、calibrate は読込後さらに回転、triangulate は上カメラに180°回転なし 等）。**試行錯誤の結果この組み合わせで動作しているので、指示がない限り変更しない**
  - check_calibration.py は pattern_size=(7,10) で他と逆
  - `KalmanFilter3D` は軸ごとに独立した共分散を持ち、初回の観測値で位置を初期化する（2026-09-28 修正。以前は共分散を3軸で共有しており、起動直後に Z が最大 +230mm ほど行き過ぎていた。定常状態の出力は修正前と同一）
  - `__main__` 内の `__wrapped__` を使う行は意味のない式（動作確認用の残骸）

## step2_datasampling（データ取得）
- common/motor_control.py: `MotorController`。プーリ径16mmで引張量[mm]→角度[deg]変換、`MAX_PULL=60mm`、電流0.1。送信 `"電流×4,角度×4e\n"`、受信 `"time (角度 電流 目標)×4"`
- data_sampling.py: 2秒ごとにランダムな1本のワイヤへ U(0,16)mm を**累積加算**し、どれかが累積30mm以上になると全ワイヤを0に戻す
  - 目標 57600 サンプル（16Hz×60分）。通信途絶時は自動で再接続して継続
  - マーカ検出時のみ CSV に記録（未検出フレームは欠落 → 時系列に隙間ができうる）
  - CSV列: `Sample_Count, Time, Target_W0-3, Angle_W0-3, Cur_W0-3, X, Y, Z` → `Data/rnn_csv/<下限>_<上限>_<累積上限>_<日時>.csv`
  - 取得したCSVは split_train_test.py で `Data/rnn_csv/train/` と `test/` に振り分ける
- split_train_test.py: 1本の長いCSVを時間ブロック（config.toml [split]、既定 5分）に分け、4ブロックに1つを test にする（約 3:1、時間的に偏らない）
  - 各ブロックの先頭に元ファイルの1行目（自然状態 = ゼロ点）を付ける（step3/4 が各CSVの1行目をゼロ点に使うため）
  - 分割前に Time 列をもとに sampling_rate の等間隔へ補間（Target_W は直前値保持、他は線形）。0.5秒超の途切れは補間せず区切る
  - 分割した元ファイルは `Data/rnn_csv/raw/` に移動。`--dry-run` で振り分けの確認のみ
  - 2026-09-28 取得の60分データ（約18Hz 記録）を 16Hz に補間して分割済み: train 10ファイル 46142行 / test 3ファイル 17555行
- `Data/rnn_csv/legacy/`: 2026-09-28 以前の旧方式（legacy 三角測量・約4fps）のデータ。学習には使わない

## step3_narx（順モデル学習） train_rnn.py
- 入力: 過去32ステップの [エンコーダ相対値 A(4) + 相対位置 x(3)] = (32,7)、未来16ステップの目標引張量 W (16,4)
- 構造: LSTM(7→64, 2層) の最終出力 + W平坦化 → FC(128-128) → 出力
- 出力: 未来**16**ステップ（1秒）の相対位置 (16,3)（仕様の「x(i+1〜10)」とは異なる）
- 位置は時刻 t-1 の現在位置を原点とする相対座標。エンコーダは各CSVの1行目基準
- StandardScaler（target / angle / rel_coord）を `Data/Weights/*.pkl` に保存、モデルは `best_trajectory_model.pth`
- 8:2 で train/val 分割、Adam, MSE, early stopping(20)。test で RMSE[mm] を表示
  - 注意: val は重なり合うウィンドウのランダム分割なので train とほぼ同じ中身になり、val loss は楽観的（early stopping が効きにくい）。精度評価は時間ブロックで分けた test で行う
- 学習結果（2026-09-29, 16Hz, train 44871 窓 / test 17174 窓, CPU で200エポック約25分, 早期終了せず val はまだ微減）:
  - test 3D RMSE **2.38mm**（X 1.61 / Y 1.57 / Z 0.79）、1秒先（+16ステップ）の誤差 1.95mm
  - 比較: 「動かない」予測 11.54mm、「等速で動く」予測 10.98mm → NARX は誤差を約 1/5 に削減（1秒間の実移動量は平均 9.5mm）
  - ログ: `Data/Weights/train_rnn_log_20260929.txt`

## step4_controller（コントローラ学習） Train_controller.py
- **仕様との差異**: ランダムシューティングによる逆データセット生成は未実装。代わりに以下2手法を比較
  - 手法A（NARX間接）: 重み固定のNARXにコントローラ出力を通し、予測軌道と実データの未来軌道のMSEで逆伝播 → `best_controller_narx.pth`
  - 手法B（直接逆モデル）: 実データの未来引張量を教師に直接回帰 → `best_controller_direct.pth`
- `ControllerMLP` 入力: 現在位置(3, 常に0) + 目標相対位置(3, t+15時点) + 過去W(32×4) + 過去A(32×4) = **262次元**（仕様の7次元とは異なる）
- 構造: [Linear(256)-LayerNorm-ReLU-Dropout(0.1)]×3 → Linear → Sigmoid
- 出力: 未来16ステップ × 4本 の引張量（[0,1] を 0〜16mm にスケール）
- AdamW, ReduceLROnPlateau, grad clip 1.0, early stopping(20)
- check_narx.py: ゼロ入力＋一定8mm引張でのNARX出力確認用

## step5_realtime（実機リアルタイム制御） realtime_controll.py
- 起動時にコントローラ（A/B）とモード（[1] testデータから20点サンプルした目標へのランダム追従 / [2] キーボードでXYZ入力）を選択
- カメラは別スレッドで取得（動画を `Data/control_results/video/` に保存）
- 各ステップでコントローラ出力16ステップのうち**先頭1ステップのみ**を指令（receding horizon 的運用）
- 1目標あたり `HOLD_STEPS=32`（2秒）制御後、自然長へ戻す。結果CSVを `Data/control_results/` に保存
- エンコーダ基準値は起動時の値（学習時は各CSVの1行目）
- `POSITION_THRESH` は動画上の誤差表示の色分けにのみ使用（過去コードの名残だった `STABLE_STEPS` は削除済み）
- 起動時のウォームアップでは、マーカを初めて検出するまで待ってから状態バッファを貯める（見失い中は直前の位置を保持）

## 仕様（制御手法）と実装の主な差分まとめ
1. 順モデル出力: 仕様 10ステップ → 実装 16ステップ（16Hz で 1秒）
2. 逆データセット: 仕様 ランダムシューティング(2000×5000) → 未実装（NARX経由の勾配学習／直接逆モデルで代替）
3. コントローラ入力: 仕様 Δx+A の7次元 → 実装 262次元（過去32ステップ分のW・Aを含む）
4. コントローラ出力: 仕様 W 4次元 → 実装 16×4（実機では先頭のみ使用）
