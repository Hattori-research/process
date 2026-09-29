"""
step4_controller/train_controller.py

2種類のMLPコントローラを学習・比較する

[手法A] モデルベース間接学習（NARX使用）
  コントローラ出力 → NARX → 予測位置 → 損失（目標位置との差）
  → コントローラの重みのみ更新

[手法B] 直接逆モデル学習
  実データの（状態, 引張量）ペアから直接コントローラを学習
  損失 = ||予測引張量 - 実際の引張量||²
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from common.models import TrajectoryNet, ControllerMLP   # NARX（重み固定）とMLPコントローラ

cfg  = config.load()
mcfg = cfg["model"]
ccfg = cfg["controller"]

# ========================================================
# パス設定（config.toml [paths]）
# ========================================================
TRAIN_CSV_DIR   = config.path("train_csv_dir")
TEST_CSV_DIR    = config.path("test_csv_dir")
WEIGHTS_DIR     = config.path("weights_dir")
NARX_MODEL_PATH = config.path("narx_model")
CTRL_A_PATH     = config.path("ctrl_narx")
CTRL_B_PATH     = config.path("ctrl_direct")

# ========================================================
# ハイパーパラメータ（config.toml [model] [controller]）
# ========================================================
PAST_SEQ   = mcfg["past_seq"]
FUTURE_SEQ = mcfg["future_seq"]
BATCH_SIZE = ccfg["batch_size"]
EPOCHS     = ccfg["epochs"]
LR         = ccfg["learning_rate"]
WEIGHT_DECAY = ccfg["weight_decay"]
PATIENCE   = ccfg["patience"]
GRAD_CLIP  = ccfg["grad_clip"]
LAMBDA_SMOOTH = ccfg["lambda_smooth"]   # 手法A: 引張量の変化の罰則
LAMBDA_PRIOR  = ccfg["lambda_prior"]    # 手法A: 記録引張量からのずれの罰則
W_MIN, W_MAX = mcfg["w_min"], mcfg["w_max"]

CUT_INITIAL_STEPS = mcfg["cut_initial_steps"]
MOVEMENT_LIMIT    = mcfg["movement_limit"]
JUMP_THRESH       = mcfg["jump_thresh"]


# ========================================================
# データセット（手法A・B共通）
# ========================================================
class ControllerDataset(Dataset):
    """
    各サンプル:
      cur_pos     : (3,)           現在位置（正規化）
      tgt_pos     : (3,)           目標位置（正規化）
      past_w      : (PAST_SEQ,4)   過去引張量（正規化）
      past_a      : (PAST_SEQ,4)   過去エンコーダ（正規化）
      x_narx      : (PAST_SEQ,7)   NARXへの入力（手法Aで使用）
      future_pos  : (FUTURE_SEQ,3) 目標相対位置（手法A損失用、正規化）
      future_w    : (FUTURE_SEQ,4) 実際の引張量（手法B損失用、正規化）
    """
    def __init__(self, csv_paths, past_seq, future_seq,
                 target_scaler, angle_scaler, rel_coord_scaler):
        self.data = []
        print(f"[Dataset] {len(csv_paths)}ファイルを読み込みます...")

        for fpath in csv_paths:
            df = pd.read_csv(fpath)
            if len(df) <= CUT_INITIAL_STEPS + past_seq + future_seq:
                continue

            base_angle = df[['Angle_W0','Angle_W1','Angle_W2','Angle_W3']].values[0]
            base_coord = df[['X','Y','Z']].values[0]

            targets = df[['Target_W0','Target_W1','Target_W2','Target_W3']].values[CUT_INITIAL_STEPS:]
            angles  = df[['Angle_W0', 'Angle_W1', 'Angle_W2', 'Angle_W3']].values[CUT_INITIAL_STEPS:]
            coords  = df[['X','Y','Z']].values[CUT_INITIAL_STEPS:]

            angles_rel = angles - base_angle
            coords_c   = np.clip(coords,
                                 base_coord - MOVEMENT_LIMIT,
                                 base_coord + MOVEMENT_LIMIT)

            t_sc = target_scaler.transform(targets).astype(np.float32)
            a_sc = angle_scaler.transform(angles_rel).astype(np.float32)

            for t in range(past_seq, len(t_sc) - future_seq):
                cur_p = coords_c[t - 1]
                tgt_p = coords_c[t + future_seq - 1]

                past_rel  = coords_c[t - past_seq : t] - cur_p
                fut_rel   = coords_c[t : t + future_seq] - cur_p

                # ジャンプ対策
                if (np.abs(np.diff(past_rel, axis=0)).max() > JUMP_THRESH or
                    np.abs(np.diff(fut_rel,  axis=0)).max() > JUMP_THRESH):
                    continue

                past_rel_sc = rel_coord_scaler.transform(past_rel).astype(np.float32)
                fut_rel_sc  = rel_coord_scaler.transform(fut_rel).astype(np.float32)

                xp_a      = a_sc[t - past_seq : t]
                narx_in   = np.hstack((xp_a, past_rel_sc))

                tgt_sc = rel_coord_scaler.transform(
                    (tgt_p - cur_p).reshape(1, -1)).flatten().astype(np.float32)

                self.data.append({
                    "cur_pos"   : np.zeros(3, dtype=np.float32),  # 常に原点基準
                    "tgt_pos"   : tgt_sc,
                    "past_w"    : t_sc[t - past_seq : t],
                    "past_a"    : xp_a,
                    "x_narx"    : narx_in,
                    "future_pos": fut_rel_sc,
                    "future_w"  : t_sc[t : t + future_seq],       # 手法B用
                })

        print(f"  有効サンプル数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return tuple(torch.from_numpy(d[k]) for k in
                     ["cur_pos","tgt_pos","past_w","past_a",
                      "x_narx","future_pos","future_w"])


# ========================================================
# 手法A: NARX間接学習
# ========================================================
def narx_loss(ctrl, narx, batch, device, w_mean, w_std):
    """
    手法Aの損失 = 位置誤差 + λ_smooth × 引張量の変化 + λ_prior × 記録引張量からのずれ
      位置誤差 : NARX(コントローラ出力) の予測軌道と実データの未来軌道の MSE（正規化座標）
      変化     : 直前の実引張量 → 出力16ステップの各ステップ間の差の2乗平均（[0,1] スケール）
                 → 0/16mm を往復するような急峻な指令を抑える
      ずれ     : 記録された未来引張量との MSE（[0,1] スケール）
                 → NARX の学習範囲から大きく外れた入力を使って予測誤差だけを下げるのを抑える
    返り値: (合計損失, 位置誤差, 変化, ずれ)
    """
    cur, tgt, pw, pa, x_narx, fut_pos, fut_w = (t.to(device) for t in batch)

    # コントローラ出力 [0,1] → 物理スケール → NARXスケール
    w_norm = ctrl(cur, tgt, pw, pa)                      # (B, F, 4) [0,1]
    w_phys = w_norm * (W_MAX - W_MIN) + W_MIN            # [mm]
    w_sc   = (w_phys - w_mean) / w_std                    # 勾配グラフ維持

    # NARX予測（NARXパラメータへの勾配は不要だがグラフは維持）
    pred = narx(x_narx, w_sc)                            # (B, F, 3)
    # 全ステップの損失（最終ステップだけでなく全体を使う）
    pos_loss = nn.functional.mse_loss(pred, fut_pos)

    # 直前の実引張量（[0,1]）から出力の各ステップへの変化
    to01 = lambda w_std_sc: ((w_std_sc * w_std + w_mean - W_MIN) / (W_MAX - W_MIN)).clamp(0.0, 1.0)
    prev = to01(pw[:, -1:, :])                           # (B, 1, 4)
    smooth = torch.diff(torch.cat([prev, w_norm], dim=1), dim=1).pow(2).mean()
    prior  = nn.functional.mse_loss(w_norm, to01(fut_w))

    loss = pos_loss + LAMBDA_SMOOTH * smooth + LAMBDA_PRIOR * prior
    return loss, pos_loss, smooth, prior


def train_narx_epoch(ctrl, narx, loader, opt, device, w_mean, w_std):
    ctrl.train()
    total = 0.0
    for batch in loader:
        opt.zero_grad()
        loss, *_ = narx_loss(ctrl, narx, batch, device, w_mean, w_std)
        loss.backward()
        nn.utils.clip_grad_norm_(ctrl.parameters(), GRAD_CLIP)
        opt.step()
        total += loss.item() * batch[0].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_narx(ctrl, narx, loader, device, w_mean, w_std, detail=False):
    ctrl.eval()
    sums = np.zeros(4)
    for batch in loader:
        vals = narx_loss(ctrl, narx, batch, device, w_mean, w_std)
        sums += np.array([v.item() for v in vals]) * batch[0].size(0)
    means = sums / len(loader.dataset)
    return means if detail else means[0]


# ========================================================
# 手法B: 直接逆モデル学習
# ========================================================
def train_direct_epoch(ctrl, loader, opt, device, w_mean, w_std):
    ctrl.train()
    total = 0.0
    for cur, tgt, pw, pa, _, _, fut_w in loader:
        cur, tgt = cur.to(device), tgt.to(device)
        pw, pa   = pw.to(device),  pa.to(device)
        fut_w    = fut_w.to(device)

        opt.zero_grad()

        w_norm = ctrl(cur, tgt, pw, pa)          # (B, F, 4) [0,1]

        # fut_w はStandardScaler正規化済み → 物理スケール[mm] → [0,1]に変換
        fut_w_phys = fut_w * w_std + w_mean      # mm
        fut_w_01   = ((fut_w_phys - W_MIN) / (W_MAX - W_MIN)).clamp(0.0, 1.0)

        loss = nn.functional.mse_loss(w_norm, fut_w_01)
        loss.backward()
        nn.utils.clip_grad_norm_(ctrl.parameters(), GRAD_CLIP)
        opt.step()
        total += loss.item() * cur.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_direct(ctrl, loader, device, w_mean, w_std):
    ctrl.eval()
    total = 0.0
    for cur, tgt, pw, pa, _, _, fut_w in loader:
        cur, tgt = cur.to(device), tgt.to(device)
        pw, pa   = pw.to(device),  pa.to(device)
        fut_w    = fut_w.to(device)
        w_norm   = ctrl(cur, tgt, pw, pa)
        fut_w_phys = fut_w * w_std + w_mean
        fut_w_01   = ((fut_w_phys - W_MIN) / (W_MAX - W_MIN)).clamp(0.0, 1.0)
        total += nn.functional.mse_loss(w_norm, fut_w_01).item() * cur.size(0)
    return total / len(loader.dataset)


# ========================================================
# 学習ループ共通関数
# ========================================================
def run_training(method, ctrl, narx, train_loader, val_loader,
                 device, save_path, w_mean, w_std):
    opt = torch.optim.AdamW(ctrl.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=ccfg["lr_factor"], patience=ccfg["lr_patience"])

    best_val, patience_cnt = float("inf"), 0
    print(f"\n=== [{method}] 学習開始 ===")

    for epoch in range(1, EPOCHS + 1):
        if method == "NARX間接":
            tr = train_narx_epoch(ctrl, narx, train_loader, opt, device, w_mean, w_std)
            vl = eval_narx(ctrl, narx, val_loader, device, w_mean, w_std)
        else:
            tr = train_direct_epoch(ctrl, train_loader, opt, device, w_mean, w_std)
            vl = eval_direct(ctrl, val_loader, device, w_mean, w_std)

        sch.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            detail = ""
            if method == "NARX間接":
                _, pos, smo, pri = eval_narx(ctrl, narx, val_loader, device, w_mean, w_std, detail=True)
                detail = f"  (Val 位置:{pos:.6f} 変化:{smo:.5f} ずれ:{pri:.5f})"
            print(f"Epoch [{epoch:3d}/{EPOCHS}]  "
                  f"Train:{tr:.6f}  Val:{vl:.6f}  "
                  f"LR:{opt.param_groups[0]['lr']:.2e}{detail}")

        if vl < best_val:
            best_val, patience_cnt = vl, 0
            torch.save(ctrl.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[Early Stopping] best val={best_val:.6f}")
                break

    print(f"  → 保存: {save_path}")
    return best_val


# ========================================================
# テスト評価（引張量の統計 + 位置誤差mm換算）
# ========================================================
@torch.no_grad()
def test_evaluation(method, ctrl, narx, loader, device,
                    rel_coord_scaler, w_mean=None, w_std=None):
    ctrl.eval()
    print(f"\n=== [{method}] テスト評価 ===")

    w_list, pred_pos_list, true_pos_list = [], [], []

    for cur, tgt, pw, pa, x_narx, fut_pos, fut_w in loader:
        cur, tgt  = cur.to(device),  tgt.to(device)
        pw, pa    = pw.to(device),   pa.to(device)
        x_narx    = x_narx.to(device)

        w_norm = ctrl(cur, tgt, pw, pa)
        w_phys = (w_norm * (W_MAX - W_MIN) + W_MIN).cpu().numpy()
        w_list.append(w_phys)

        if method == "NARX間接":
            w_sc = (w_norm * (W_MAX - W_MIN) + W_MIN - w_mean) / w_std
            pred = narx(x_narx, w_sc).cpu().numpy()
            true = fut_pos.numpy()
            pred_pos_list.append(pred)
            true_pos_list.append(true)

    w_all = np.concatenate(w_list, axis=0)
    print("引張量出力統計 [mm]:")
    for i in range(4):
        wi = w_all[:, :, i]
        print(f"  W{i}: mean={wi.mean():.2f} std={wi.std():.2f} "
              f"min={wi.min():.2f} max={wi.max():.2f}")
    print(f"  ステップ間の変化 |ΔW| 平均 {np.abs(np.diff(w_all, axis=1)).mean():.3f} mm")

    if method == "NARX間接" and pred_pos_list:
        preds = np.concatenate(pred_pos_list, axis=0)
        trues = np.concatenate(true_pos_list, axis=0)

        # 逆正規化してmm換算
        N, F, C = preds.shape
        preds_mm = rel_coord_scaler.inverse_transform(
            preds.reshape(-1, C)).reshape(N, F, C)
        trues_mm = rel_coord_scaler.inverse_transform(
            trues.reshape(-1, C)).reshape(N, F, C)

        err = preds_mm - trues_mm
        rmse = np.sqrt((err**2).mean(axis=(0, 1)))
        print(f"NARX予測位置誤差 [mm]: X={rmse[0]:.2f}  Y={rmse[1]:.2f}  Z={rmse[2]:.2f}  "
              f"3D={np.sqrt((err**2).sum(axis=-1).mean()):.2f}")


# ========================================================
# メイン
# ========================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B"], help="片方の手法だけ学習する（省略時は両方）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # スケーラ読み込み
    target_scaler    = joblib.load(config.path("target_scaler"))
    angle_scaler     = joblib.load(config.path("angle_scaler"))
    rel_coord_scaler = joblib.load(config.path("rel_coord_scaler"))

    # NARXモデル読み込み・全パラメータ固定
    narx = TrajectoryNet(PAST_SEQ, FUTURE_SEQ).to(device)
    narx.load_state_dict(torch.load(NARX_MODEL_PATH, map_location=device,
                                    weights_only=True))
    narx.eval()
    for p in narx.parameters():
        p.requires_grad = False
    print(f"NARX読み込み完了: {NARX_MODEL_PATH}")

    # w_mean/w_std をデバイステンソルに（勾配グラフ維持用）
    w_mean = torch.tensor(target_scaler.mean_,  dtype=torch.float32, device=device)
    w_std  = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)

    # fut_wはStandardScaler正規化済みなので、手法Bでは[0,1]比較のために
    # fut_wを物理スケールに戻してSigmoidと比較できるよう修正が必要
    # → シンプルに fut_w を [0,1] に変換してデータセットに格納する方が明快
    # （ここではtarget_scaler の逆変換 → [W_MIN,W_MAX] で割る）

    # データセット
    train_csv = glob.glob(os.path.join(TRAIN_CSV_DIR, "*.csv"))
    test_csv  = glob.glob(os.path.join(TEST_CSV_DIR,  "*.csv"))
    if not train_csv:
        raise FileNotFoundError(TRAIN_CSV_DIR)
    if not test_csv:
        raise FileNotFoundError(TEST_CSV_DIR)

    print(f"\n学習: {len(train_csv)}ファイル  テスト: {len(test_csv)}ファイル")

    ds_train = ControllerDataset(train_csv, PAST_SEQ, FUTURE_SEQ,
                                  target_scaler, angle_scaler, rel_coord_scaler)
    ds_test  = ControllerDataset(test_csv,  PAST_SEQ, FUTURE_SEQ,
                                  target_scaler, angle_scaler, rel_coord_scaler)

    n_tr = int(len(ds_train) * ccfg["train_ratio"])
    n_vl = len(ds_train) - n_tr
    ds_tr, ds_vl = torch.utils.data.random_split(
        ds_train, [n_tr, n_vl], generator=torch.Generator().manual_seed(ccfg["split_seed"]))

    kw = dict(batch_size=BATCH_SIZE, num_workers=0)
    tr_loader   = DataLoader(ds_tr,   shuffle=True,  **kw)
    vl_loader   = DataLoader(ds_vl,   shuffle=False, **kw)
    test_loader = DataLoader(ds_test, shuffle=False, **kw)

    print(f"Train:{len(ds_tr)}  Val:{len(ds_vl)}  Test:{len(ds_test)}")
    print(f"手法A 正則化: lambda_smooth={LAMBDA_SMOOTH}  lambda_prior={LAMBDA_PRIOR}")

    # ==========================================
    # 手法A: NARX間接学習
    # ==========================================
    if args.only != "B":
        ctrl_a = ControllerMLP(PAST_SEQ, FUTURE_SEQ).to(device)
        run_training("NARX間接", ctrl_a, narx, tr_loader, vl_loader,
                     device, CTRL_A_PATH, w_mean, w_std)
        ctrl_a.load_state_dict(torch.load(CTRL_A_PATH, map_location=device,
                                          weights_only=True))
        test_evaluation("NARX間接", ctrl_a, narx, test_loader, device,
                        rel_coord_scaler, w_mean, w_std)

    # ==========================================
    # 手法B: 直接逆モデル学習
    # ==========================================
    if args.only != "A":
        ctrl_b = ControllerMLP(PAST_SEQ, FUTURE_SEQ).to(device)
        run_training("直接逆モデル", ctrl_b, narx, tr_loader, vl_loader,
                     device, CTRL_B_PATH, w_mean, w_std)
        ctrl_b.load_state_dict(torch.load(CTRL_B_PATH, map_location=device,
                                          weights_only=True))
        test_evaluation("直接逆モデル", ctrl_b, narx, test_loader, device,
                        rel_coord_scaler, w_mean, w_std)

    print("\n=== 全学習完了 ===")
    print(f"  手法A: {CTRL_A_PATH}")
    print(f"  手法B: {CTRL_B_PATH}")


if __name__ == "__main__":
    main()