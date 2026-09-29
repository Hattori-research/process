"""
split_train_test.py

data_sampling.py で取得した1本の長いCSVを、時間ブロックごとに train / test に振り分ける
（カメラ・モータは使わない）

  - [split] resample = true のとき、分割前に Time 列をもとに sampling_rate の等間隔へ補間し直す
      Target_W（指令値）は直前値保持、それ以外（Angle, Cur, X, Y, Z）は線形補間
      記録が max_gap_sec より長く途切れた所（マーカ見失い・再接続など）は補間せず、そこで区切る（セグメント）
  - ブロック長・test の割合は config.toml [split]
  - 例: block_minutes=5, test_every=4, test_index=2
        → ブロック 0,1,[2],3,4,5,[6],... の [ ] が test（約 3:1、時間的に偏らない）
  - step3/4 は各CSVの1行目を「エンコーダ値・座標のゼロ点（自然状態）」として使うため、
    各ブロックの先頭に元ファイルの1行目（自然状態）を付けて保存する
  - ウィンドウ（過去+未来）は各ファイル内でしか作られないので、ブロックをまたぐリークは起きない
    （さらに step3/4 は各ファイル先頭 cut_initial_steps 行を捨てるため、境目に隙間もできる）
  - 分割した元ファイルは [paths] raw_csv_dir に移動する（二重に分割しないため）

使い方:
  python step2_datasampling/split_train_test.py              # rnn_csv_dir 直下の CSV をすべて分割
  python step2_datasampling/split_train_test.py a.csv b.csv  # 指定したファイルだけ
  python step2_datasampling/split_train_test.py --dry-run    # 振り分け結果の表示のみ
"""
import os
import sys
import glob
import shutil
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

HOLD_COLS = [f"Target_W{i}" for i in range(4)]   # 指令値：直前値保持で補間


def split_segments(df, max_gap):
    """Time が巻き戻る所・max_gap 秒より空く所で区切ったセグメントのリストを返す"""
    dt = np.diff(df["Time"].values)
    cuts = [0, *(np.where((dt <= 0) | (dt > max_gap))[0] + 1), len(df)]
    segs = [df.iloc[a:b].reset_index(drop=True) for a, b in zip(cuts[:-1], cuts[1:])]
    return [seg for seg in segs if len(seg) > 1]


def resample_segment(seg, rate):
    """1セグメントを rate [Hz] の等間隔に補間する"""
    t = seg["Time"].values
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * rate)) + 1) / rate
    out = {"Sample_Count": None, "Time": grid}
    idx_prev = np.searchsorted(t, grid, side="right") - 1          # 直前の実サンプル
    for col in seg.columns:
        if col in ("Sample_Count", "Time"):
            continue
        v = seg[col].values
        out[col] = v[idx_prev] if col in HOLD_COLS else np.interp(grid, t, v)
    res = pd.DataFrame(out)[seg.columns]
    return res


def split_file(path, train_dir, test_dir, cfg, dry_run=False):
    scfg = cfg["split"]
    mcfg = cfg["model"]
    rate = cfg["system"]["sampling_rate"]
    block_rows = int(scfg["block_minutes"] * 60 * rate)
    # 学習で使える最低行数（先頭カット + 過去 + 未来 + 1）
    min_usable = mcfg["cut_initial_steps"] + mcfg["past_seq"] + mcfg["future_seq"] + 1

    df = pd.read_csv(path)
    if len(df) < 2:
        print(f"[skip] {os.path.basename(path)}: データなし")
        return None
    base_row = df.iloc[[0]].copy()     # 自然状態（ゼロ点）

    # --- 等間隔化（セグメントごと）---
    if scfg["resample"]:
        segs = split_segments(df, scfg["max_gap_sec"])
        segs = [resample_segment(s, rate) for s in segs]
        dt = np.diff(df["Time"].values)
        print(f"{os.path.basename(path)}: {len(df)} 行（平均 {1 / dt[dt > 0].mean():.1f}Hz）"
              f" → {rate:g}Hz に補間、{len(segs)} セグメント（{scfg['max_gap_sec']}s 超の途切れで区切り）")
    else:
        segs = [df]
        print(f"{os.path.basename(path)}: {len(df)} 行（補間なし）")

    # --- ブロックに分けて振り分け ---
    stem = os.path.splitext(os.path.basename(path))[0]
    summary = {"train": 0, "test": 0, "dropped": 0}
    count = 0                           # Sample_Count を通し番号で振り直す
    k = 0                               # 全セグメント通しのブロック番号
    for si, seg in enumerate(segs):
        seg = seg.copy()
        seg["Sample_Count"] = np.arange(count, count + len(seg))
        count += len(seg)
        # 末尾の端数が小さければ直前に結合
        starts = list(range(0, len(seg), block_rows))
        if len(starts) > 1 and len(seg) - starts[-1] < block_rows * scfg["min_block_ratio"]:
            starts.pop()
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(seg)
            block = seg.iloc[s:e]
            if not (si == 0 and j == 0):
                block = pd.concat([base_row, block])   # 先頭にゼロ点（自然状態）を付ける
            if len(block) < min_usable:
                print(f"  seg {si:02d} block {k:02d}: {e - s:5d} 行 → 破棄（{min_usable} 行未満）")
                summary["dropped"] += e - s
                continue
            split = "test" if k % scfg["test_every"] == scfg["test_index"] else "train"
            print(f"  seg {si:02d} block {k:02d}: {e - s:5d} 行 → {split}")
            summary[split] += e - s
            if not dry_run:
                out_dir = test_dir if split == "test" else train_dir
                block.to_csv(os.path.join(out_dir, f"{stem}_b{k:02d}.csv"), index=False)
            k += 1
    total = summary["train"] + summary["test"]
    print(f"  → train {summary['train']} 行 ({summary['train'] / total:.0%}) / "
          f"test {summary['test']} 行 ({summary['test'] / total:.0%}) / 破棄 {summary['dropped']} 行")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="分割するCSV（省略時は rnn_csv_dir 直下のすべて）")
    ap.add_argument("--dry-run", action="store_true", help="振り分け結果を表示するだけで保存しない")
    args = ap.parse_args()

    cfg = config.load()
    src_dir   = config.path("rnn_csv_dir")
    train_dir = config.path("train_csv_dir")
    test_dir  = config.path("test_csv_dir")
    raw_dir   = config.path("raw_csv_dir")

    files = args.files or sorted(glob.glob(os.path.join(src_dir, "*.csv")))
    if not files:
        print(f"分割するCSVがありません: {src_dir}")
        return

    if not args.dry_run:
        for d in (train_dir, test_dir, raw_dir):
            os.makedirs(d, exist_ok=True)

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        existing = glob.glob(os.path.join(train_dir, f"{stem}_b*.csv")) + \
                   glob.glob(os.path.join(test_dir, f"{stem}_b*.csv"))
        if existing and not args.dry_run:
            print(f"[skip] {os.path.basename(f)}: 分割済みのファイルがすでにあります（{len(existing)} 個）")
            continue
        if split_file(f, train_dir, test_dir, cfg, dry_run=args.dry_run) is None:
            continue
        if not args.dry_run:
            dst = os.path.join(raw_dir, os.path.basename(f))
            if os.path.abspath(f) == os.path.abspath(dst):
                continue                  # raw/ のファイルを分割し直した場合は移動しない
            shutil.move(f, dst)
            print(f"  元ファイルを移動: {dst}")


if __name__ == "__main__":
    main()
