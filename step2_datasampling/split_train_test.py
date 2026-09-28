"""
split_train_test.py

data_sampling.py で取得した1本の長いCSVを、時間ブロックごとに train / test に振り分ける
（カメラ・モータは使わない）

  - ブロック長・test の割合は config.toml [split]
  - 例: block_minutes=5, test_every=5, test_index=2
        → ブロック 0,1,[2],3,4,5,6,[7],... の [ ] が test（約 8:2、時間的に偏らない）
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
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config


def split_file(path, train_dir, test_dir, cfg, dry_run=False):
    scfg = cfg["split"]
    mcfg = cfg["model"]
    block_rows = int(scfg["block_minutes"] * 60 * cfg["system"]["sampling_rate"])
    # 学習で使える最低行数（先頭カット + 過去 + 未来 + 1）
    min_usable = mcfg["cut_initial_steps"] + mcfg["past_seq"] + mcfg["future_seq"] + 1

    df = pd.read_csv(path)
    if len(df) < 2:
        print(f"[skip] {os.path.basename(path)}: データなし")
        return None
    base_row = df.iloc[[0]]            # 自然状態（ゼロ点）

    # ブロックの区切り（末尾の端数が小さければ直前に結合）
    starts = list(range(0, len(df), block_rows))
    if len(starts) > 1 and len(df) - starts[-1] < block_rows * scfg["min_block_ratio"]:
        starts.pop()
    bounds = [(s, starts[i + 1] if i + 1 < len(starts) else len(df)) for i, s in enumerate(starts)]

    stem = os.path.splitext(os.path.basename(path))[0]
    summary = {"train": 0, "test": 0}
    print(f"{os.path.basename(path)}: {len(df)} 行 → {len(bounds)} ブロック（{block_rows} 行/ブロック）")
    for i, (s, e) in enumerate(bounds):
        split = "test" if i % scfg["test_every"] == scfg["test_index"] else "train"
        block = df.iloc[s:e]
        if i > 0:
            block = pd.concat([base_row, block])   # 先頭にゼロ点（自然状態）を付ける
        n = len(block)
        note = "" if n >= min_usable else f"  [!] {min_usable} 行未満のため学習では使われない"
        print(f"  block {i:02d}: 行 {s:6d}〜{e - 1:6d} → {split:5s}{note}")
        summary[split] += e - s
        if not dry_run:
            out_dir = test_dir if split == "test" else train_dir
            block.to_csv(os.path.join(out_dir, f"{stem}_b{i:02d}.csv"), index=False)
    total = summary["train"] + summary["test"]
    print(f"  → train {summary['train']} 行 ({summary['train'] / total:.0%}) / "
          f"test {summary['test']} 行 ({summary['test'] / total:.0%})")
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
            shutil.move(f, dst)
            print(f"  元ファイルを移動: {dst}")


if __name__ == "__main__":
    main()
