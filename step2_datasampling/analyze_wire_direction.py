"""
analyze_wire_direction.py

取得データ（data_sampling.py の CSV）から「どのワイヤを引くと、どの向きに動くか」を集計する（読み取りのみ）

集計方法:
  - 指令が変わった時刻のうち、1本だけが増えた指令（incremental の「引き足し」）を取り出す
    （どれかが減った指令＝全解放などは除外。random_multi のデータでは 1本だけ増えた指令のみ対象）
  - 指令が変わる直前の位置から、settle_sec 秒後（次の指令の前）の位置までの移動を、引き足した量 [mm] で割る
    → 1mm 引いたときの移動 (dX, dY, dZ) [mm/mm]
  - 水平面での向き = atan2(dY, dX)（X=奥行が正, Y=水平左が正）
  - 向きの一貫性 = 各回の移動方向（単位ベクトル）の平均の長さ（1 = 毎回同じ向き、0 = ばらばら）
  - あわせて、全サンプルの位置が水平面のどの向きに分布しているかを 45°刻みで数える

使い方:
  python step2_datasampling/analyze_wire_direction.py [CSV ...]   # 省略時は Data/rnn_csv/raw/ の最新ファイル
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config


def analyze(path, settle_sec=1.8):
    df = pd.read_csv(path)
    W = df[[f"Target_W{i}" for i in range(4)]].values
    P = df[["X", "Y", "Z"]].values
    T = df["Time"].values

    changes = np.where(np.abs(np.diff(W, axis=0)).sum(1) > 0)[0] + 1
    res = {i: [] for i in range(4)}
    n_skip = 0
    for k, c in enumerate(changes):
        d = W[c] - W[c - 1]
        if (d < 0).any() or (d > 0).sum() != 1:          # 1本だけ増えた指令のみ
            n_skip += 1
            continue
        i = int(np.argmax(d))
        end = np.searchsorted(T, T[c] + settle_sec)
        if end >= len(P) or (k + 1 < len(changes) and changes[k + 1] < end):
            n_skip += 1
            continue
        res[i].append(np.r_[P[end] - P[c - 1], d[i]])

    print(f"== {os.path.basename(path)}  （{len(df)} 行、指令変更 {len(changes)} 回、うち集計対象外 {n_skip} 回）")
    print("| 引いたワイヤ | 回数 | 1mm 引いたときの移動 (dX, dY, dZ) [mm/mm] | 水平面での向き | 向きの一貫性 |")
    print("|---|---|---|---|---|")
    for i in range(4):
        if not res[i]:
            print(f"| W{i} | 0 | - | - | - |")
            continue
        a = np.array(res[i])
        d = a[:, :3] / a[:, 3:4]
        m = d.mean(0)
        ang = np.degrees(np.arctan2(m[1], m[0]))
        unit = d / np.linalg.norm(d, axis=1, keepdims=True)
        consistency = np.linalg.norm(unit.mean(0))
        print(f"| W{i} | {len(a)} | ({m[0]:+.2f}, {m[1]:+.2f}, {m[2]:+.2f}) | {ang:+.0f}° | {consistency:.2f} |")

    ang_all = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    hist, edges = np.histogram(ang_all, bins=np.arange(-180, 181, 45))
    print("\n位置の分布: " + "  ".join(f"{c} {np.percentile(P[:, j], 5):.1f}〜{np.percentile(P[:, j], 95):.1f}mm"
                                  for j, c in enumerate("XYZ")) + "（5〜95%）")
    print("水平面での位置の向き（原点から、45°刻み）:")
    for lo, hi, h in zip(edges[:-1], edges[1:], hist):
        print(f"  {lo:+4.0f}〜{hi:+4.0f}°: {h / len(P):6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="集計する CSV（省略時は raw/ の最新ファイル）")
    ap.add_argument("--settle", type=float, default=1.8, help="指令変更から何秒後の位置で移動を測るか [s]")
    args = ap.parse_args()
    files = args.files or sorted(glob.glob(os.path.join(config.path("raw_csv_dir"), "*.csv")), key=os.path.getmtime)[-1:]
    for f in files:
        analyze(f, args.settle)


if __name__ == "__main__":
    main()
