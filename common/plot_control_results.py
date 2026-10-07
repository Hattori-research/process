"""
common/plot_control_results.py

step5 の結果CSV（Data/control_results/*.csv）を選んで、到達精度を比べるグラフを作る

  1. 時間ごとの誤差：試行の平均（線）と 25〜75% の範囲（帯）
  2. 4秒時点（最終ステップ）の誤差の分布：箱ひげ図＋各試行の点（平均・中央値を表示）
  3. 2ファイルを選んだとき：同じ目標どうしの最終誤差の散布図（y=x より下なら後者が良い）と Wilcoxon の符号順位検定
     （step5 の目標点は target_seed で固定なので、同じ試行番号は同じ目標）

使い方:
  python common/plot_control_results.py            # ファイル選択ダイアログで CSV を選ぶ（複数可）
  python common/plot_control_results.py a.csv b.csv
凡例のラベルはファイル名から自動で付け（NARX間接 → 手法A、直接逆モデル → 手法B、積分補正の有無は corr 列から判定）、
確認ダイアログで書き換えられる。図は [paths] control_plots に PNG で保存する
"""
import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
SAMPLING_RATE = config.load()["system"]["sampling_rate"]
METHOD_NAMES = {"NARX間接": "手法A", "直接逆モデル": "手法B"}


def choose_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    paths = filedialog.askopenfilenames(
        title="グラフにする結果CSVを選択（複数可）",
        initialdir=config.path("control_results"),
        filetypes=[("CSV", "*.csv"), ("すべて", "*.*")])
    root.destroy()
    return list(paths)


def default_label(path, df):
    name = os.path.splitext(os.path.basename(path))[0]
    method, _, ts = name.partition("_")
    label = METHOD_NAMES.get(method, method)
    if {"corr_x", "corr_y", "corr_z"} <= set(df.columns):
        label += "・補正あり" if df[["corr_x", "corr_y", "corr_z"]].abs().to_numpy().max() > 0 else "・補正なし"
    return f"{label}（{ts[4:8]}_{ts[9:13]}）" if len(ts) >= 13 else label


def ask_labels(labels):
    """自動で付けたラベルを確認・書き換え（キャンセルならそのまま）"""
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk(); root.withdraw()
    out = []
    for i, lb in enumerate(labels):
        s = simpledialog.askstring("凡例のラベル", f"{i + 1}/{len(labels)} 本目のラベル", initialvalue=lb, parent=root)
        out.append(s.strip() if s and s.strip() else lb)
    root.destroy()
    return out


def load(path):
    df = pd.read_csv(path)
    err = df.pivot(index="trial", columns="step", values="error_3d")   # 行: 試行, 列: ステップ
    return df, err


def plot(paths, labels, out_dir):
    data = [load(p) for p in paths]
    n = len(data)
    pair = n == 2
    fig, axes = plt.subplots(1, 3 if pair else 2, figsize=(17 if pair else 12, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1. 時間ごとの誤差
    ax = axes[0]
    for i, ((df, err), lb) in enumerate(zip(data, labels)):
        t = (err.columns.to_numpy() + 1) / SAMPLING_RATE
        ax.plot(t, err.mean(), color=colors[i % len(colors)], label=f"{lb}（n={len(err)}）")
        ax.fill_between(t, err.quantile(0.25), err.quantile(0.75), color=colors[i % len(colors)], alpha=0.2)
    ax.set_xlabel("制御開始からの時間 [s]"); ax.set_ylabel("目標との誤差 [mm]")
    ax.set_title("時間ごとの誤差（線: 平均、帯: 25〜75%）")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend()

    # 2. 最終誤差の分布
    ax = axes[1]
    finals = [err.ffill(axis=1).iloc[:, -1].to_numpy() for _, err in data]
    ax.boxplot(finals, showfliers=False, widths=0.5)
    rng = np.random.default_rng(0)
    for i, f in enumerate(finals):
        ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(f)), f, s=14, alpha=0.6, color=colors[i % len(colors)])
        ax.text(i + 1, f.max() * 1.04, f"平均 {f.mean():.1f}\n中央値 {np.median(f):.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(1, n + 1)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("最終ステップの誤差 [mm]")
    ax.set_title(f"最終誤差の分布（{(data[0][1].columns.max() + 1) / SAMPLING_RATE:.0f}秒時点）")
    ax.set_ylim(0, max(f.max() for f in finals) * 1.25); ax.grid(alpha=0.3, axis="y")

    # 3. 同じ目標どうしの比較（2ファイルのとき）
    if pair:
        ax = axes[2]
        (_, e1), (_, e2) = data
        common = e1.index.intersection(e2.index)
        f1 = e1.ffill(axis=1).iloc[:, -1].loc[common].to_numpy()
        f2 = e2.ffill(axis=1).iloc[:, -1].loc[common].to_numpy()
        lim = max(f1.max(), f2.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.scatter(f1, f2, s=18, alpha=0.7)
        p = wilcoxon(f1, f2).pvalue if len(common) > 0 and np.any(f1 != f2) else float("nan")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"{labels[0]} の最終誤差 [mm]"); ax.set_ylabel(f"{labels[1]} の最終誤差 [mm]")
        ax.set_title(f"同じ目標どうしの比較（n={len(common)}）\n"
                     f"差の平均（後者−前者）{np.mean(f2 - f1):+.2f}mm、後者が小さい {np.mean(f2 < f1):.0%}、Wilcoxon p={p:.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "compare_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    fig.savefig(out, dpi=150)
    print(f"保存: {out}")
    for lb, f in zip(labels, finals):
        print(f"  {lb}: n={len(f)} 最終誤差 平均 {f.mean():.2f} / 中央値 {np.median(f):.2f} / 5mm未満 {np.mean(f < 5):.0%} / 10mm未満 {np.mean(f < 10):.0%}")
    plt.show()


def main():
    paths = sys.argv[1:] or choose_files()
    if not paths:
        print("CSV が選ばれませんでした"); return
    labels = [default_label(p, pd.read_csv(p)) for p in paths]
    if len(sys.argv) <= 1:
        labels = ask_labels(labels)
    plot(paths, labels, config.path("control_plots"))


if __name__ == "__main__":
    main()
