"""
check_triangulation.py

三角測量の新旧方式（calibrated / legacy）を並べて表示し、定規での実測値と比較するためのツール

操作:
  [SPACE] N フレームを平均して記録（コンソールで実測値やメモを入力）→ CSV に保存
  [o]     自然状態で押す：現在位置が原点 (0,0,0) になるよう
          config.toml [triangulation] cam_offset_calibrated を書き換える
  [ESC]   終了
  ※ 平均中にマーカが動いていた（std > max_std）場合は記録・原点設定をしない

--headless N : ウィンドウを出さず、N フレーム計測して統計だけ表示して終了
"""
import os
import sys
import time
import argparse
import datetime
import csv
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from common.stereo_triangulate import StereoTracker

_ccfg = config.load()["triangulation_check"]
AVG_FRAMES = _ccfg["avg_frames"]   # 記録・原点設定で平均するフレーム数
MAX_STD    = _ccfg["max_std"]      # これを超えたら「動いている」とみなす [mm]


def is_still(cal):
    """calibrated の各軸 std が MAX_STD 以下なら True。超えていれば警告を出す"""
    std = cal.std(0)
    if (std > MAX_STD).any():
        print(f"  [!] マーカが動いています（std {std.round(2)} > {MAX_STD} mm）。静止させてからやり直してください")
        return False
    return True


def set_origin(tracker, cal):
    """現在位置（calibrated）が (0,0,0) になるよう cam_offset_calibrated を更新"""
    old = tracker.CAM_OFFSETS["calibrated"] - np.array(config.load()["triangulation"]["hand_eye_cal"])
    new = (old - cal.mean(0)).round(2)
    print(f"  cam_offset_calibrated: {old.round(2)} → {new}")
    ans = input("  config.toml に書き込みますか？ [y/N] > ").strip().lower()
    if ans != "y":
        print("  → 書き込みませんでした")
        return
    config.update("triangulation", {"cam_offset_calibrated": [float(v) for v in new]})
    tracker.CAM_OFFSETS["calibrated"] = new + np.array(config.load()["triangulation"]["hand_eye_cal"])
    if tracker.method == "calibrated":
        tracker.CAM_OFFSET = tracker.CAM_OFFSETS["calibrated"]
    print("  → 原点を更新しました")


def collect(tracker, n, timeout=10.0):
    """両方式が有効なフレームを n 個集めて (legacy, calibrated, epipolar) の配列を返す"""
    leg, cal, epi = [], [], []
    t0 = time.time()
    while len(cal) < n and time.time() - t0 < timeout:
        tracker.get_3d_coordinates_and_frames()
        r = tracker.last_raw
        if r["legacy"] is not None and r["calibrated"] is not None:
            leg.append(r["legacy"]); cal.append(r["calibrated"]); epi.append(tracker.last_epipolar_px)
    return np.array(leg), np.array(cal), np.array(epi)


def summarize(leg, cal, epi):
    print(f"  有効フレーム数: {len(cal)}")
    if len(cal) == 0:
        return
    print(f"  legacy     X,Y,Z = {leg.mean(0).round(1)}  (std {leg.std(0).round(2)})")
    print(f"  calibrated X,Y,Z = {cal.mean(0).round(1)}  (std {cal.std(0).round(2)})")
    print(f"  差 (calibrated - legacy) = {(cal.mean(0) - leg.mean(0)).round(1)}")
    print(f"  エピポーラ誤差 = {epi.mean():.2f} px (max {epi.max():.2f})")
    print(f"  カメラ奥行き（calibrated, X - cam_offset_calibrated[0]）= "
          f"{cal.mean(0)[0] - config.load()['triangulation']['cam_offset_calibrated'][0]:.1f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, default=0)
    args = ap.parse_args()

    # 平滑化は比較の邪魔になるので、フィルタ前の値（last_raw）を使う
    tracker = StereoTracker()
    time.sleep(2.0)

    if args.headless:
        summarize(*collect(tracker, args.headless))
        tracker.close()
        return

    out_dir = config.path("triangulation_check")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv")
    header = ["time", "memo",
              "leg_x", "leg_y", "leg_z", "cal_x", "cal_y", "cal_z",
              "cal_std_x", "cal_std_y", "cal_std_z", "epipolar_px", "n_frames"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)
    print(f"記録先: {out_path}")
    print("[SPACE] 記録  [o] 原点設定（自然状態で）  [ESC] 終了")

    while True:
        _, frame_top, frame_under = tracker.get_3d_coordinates_and_frames()
        if frame_top is None or frame_under is None:
            continue
        r = tracker.last_raw
        lines = []
        for key, color in (("calibrated", (0, 255, 0)), ("legacy", (0, 200, 255))):
            p = r[key]
            text = f"{key:10s} X:{p[0]:7.1f} Y:{p[1]:7.1f} Z:{p[2]:7.1f}" if p is not None else f"{key:10s} ---"
            lines.append((text, color))
        if tracker.last_epipolar_px is not None:
            lines.append((f"epipolar {tracker.last_epipolar_px:5.2f} px", (255, 255, 255)))
        for frame in (frame_top, frame_under):
            for i, (text, color) in enumerate(lines):
                cv2.putText(frame, text, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        disp = np.hstack([cv2.resize(frame_top, (0, 0), fx=0.5, fy=0.5),
                          cv2.resize(frame_under, (0, 0), fx=0.5, fy=0.5)])
        cv2.imshow("check_triangulation", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('o'):
            leg, cal, epi = collect(tracker, AVG_FRAMES)
            summarize(leg, cal, epi)
            if len(cal) > 0 and is_still(cal):
                set_origin(tracker, cal)
        if key == 32:
            leg, cal, epi = collect(tracker, AVG_FRAMES)
            summarize(leg, cal, epi)
            if len(cal) == 0 or not is_still(cal):
                continue
            memo = input("  実測値やメモ（例: 'ruler X=0 Y=0 Z=0'、空欄可）> ")
            row = [datetime.datetime.now().isoformat(timespec="seconds"), memo,
                   *leg.mean(0).round(2), *cal.mean(0).round(2), *cal.std(0).round(3),
                   round(float(epi.mean()), 3), len(cal)]
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            print("  → 記録しました")

    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
