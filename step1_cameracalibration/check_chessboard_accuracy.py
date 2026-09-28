"""
check_chessboard_accuracy.py

キャリブ用チェスボード（既知寸法）を両カメラに映し、三角測量の精度を新旧方式で評価する

  - コーナー検出は stereo_triangulate.py と同じ前処理（回転）をかけた実行時フレームで行う
  - 各コーナーを calibrated / legacy の両方式で三角測量し、
      隣接コーナー間隔（真値 = square_size）
      端から端の長さ（真値 = (cols-1)*square, (rows-1)*square）
      平面からのずれ（真値 = 0）
    を比較する

使い方:
  python step1_cameracalibration/check_chessboard_accuracy.py [--seconds 60]
  → 実行中にボードの距離（奥行き）や傾きをゆっくり変える。検出できたフレームごとに結果を表示し、
    最後に奥行き帯ごとのまとめを表示する
"""
import os
import sys
import time
import argparse
import datetime
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from common.stereo_triangulate import StereoTracker


def grab_calc_frames(tracker):
    """stereo_triangulate.get_3d_coordinates_and_frames と同じ回転をかけた計算用フレーム"""
    tracker.cap_top.grab()
    tracker.cap_under.grab()
    ok_l, fl = tracker.cap_top.retrieve()
    ok_r, fr = tracker.cap_under.retrieve()
    if not (ok_l and ok_r):
        return None, None
    fr = cv2.rotate(fr, cv2.ROTATE_180)
    return cv2.rotate(fl, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(fr, cv2.ROTATE_90_CLOCKWISE)


def find_corners(gray, patterns):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    for pat in patterns:
        ok, c = cv2.findChessboardCorners(gray, pat, flags)
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
            return pat, c.reshape(-1, 2)
    return None, None


def triangulate_all(tracker, cl, cr, shape):
    """全コーナーを両方式で三角測量（カメラ座標, mm）。失敗したコーナーは NaN"""
    cal, leg, epi = [], [], []
    for pl, pr in zip(cl, cr):
        c = tracker._triangulate_calibrated(pl, pr, shape, shape)
        epi.append(tracker.last_epipolar_px)
        cal.append(c if c is not None else [np.nan] * 3)
        g = tracker._triangulate_legacy(pl, pr)
        leg.append(g if g is not None else [np.nan] * 3)
    return np.array(cal), np.array(leg), np.array(epi)


def board_metrics(P, pat, square):
    """P: (rows*cols, 3)。隣接間隔・端から端・平面度を返す"""
    cols, rows = pat
    G = P.reshape(rows, cols, 3)
    d_h = np.linalg.norm(np.diff(G, axis=1), axis=2).ravel()   # 横方向の隣接
    d_v = np.linalg.norm(np.diff(G, axis=0), axis=2).ravel()   # 縦方向の隣接
    adj = np.concatenate([d_h, d_v])
    len_h = np.linalg.norm(G[:, -1] - G[:, 0], axis=1)          # 各行の端から端
    len_v = np.linalg.norm(G[-1, :] - G[0, :], axis=1)          # 各列の端から端
    # 平面度：最小二乗平面からの距離
    Q = P[~np.isnan(P).any(1)]
    ctr = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - ctr)
    plane = np.abs((Q - ctr) @ vt[2])
    return {
        "adj_mean": np.nanmean(adj), "adj_std": np.nanstd(adj),
        "len_h": np.nanmean(len_h), "len_h_true": (cols - 1) * square,
        "len_v": np.nanmean(len_v), "len_v_true": (rows - 1) * square,
        "plane_rms": np.sqrt(np.mean(plane ** 2)),
        "depth": np.nanmean(P[:, 2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    cb = config.load()["chessboard"]
    square = cb["square_size"]
    patterns = [(cb["cols"], cb["rows"]), (cb["rows"], cb["cols"])]

    tracker = StereoTracker()
    time.sleep(2.0)
    print(f"チェスボード {cb['cols']}x{cb['rows']}（{square}mm）を両カメラに映し、距離・傾きをゆっくり変えてください"
          f"（{args.seconds:.0f}秒）")

    results = []
    saved = False
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        fl, fr = grab_calc_frames(tracker)
        if fl is None:
            continue
        gl, gr = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        pat_l, cl = find_corners(gl, patterns)
        if cl is None:
            continue
        pat_r, cr = find_corners(gr, [pat_l])
        if cr is None:
            continue

        # 左右でコーナーの並び順が逆になることがあるので、エピポーラ誤差が小さい方を採用
        best = None
        for cand in (cr, cr[::-1]):
            cal, leg, epi = triangulate_all(tracker, cl, cand, fl.shape)
            score = np.nanmedian(epi)
            if best is None or score < best[0]:
                best = (score, cal, leg, epi)
        _, cal, leg, epi = best
        if np.isnan(cal).any():
            print(f"  [skip] エピポーラ誤差が大きいコーナーあり（median {np.nanmedian(epi):.2f}px）")
            continue

        mc = board_metrics(cal, pat_l, square)
        ml = board_metrics(leg, pat_l, square)
        results.append((mc, ml, np.median(epi)))
        print(f"  奥行 {mc['depth']:6.1f}mm | 隣接間隔 cal {mc['adj_mean']:6.2f}±{mc['adj_std']:.2f}  "
              f"leg {ml['adj_mean']:6.2f}±{ml['adj_std']:.2f} (真 {square}) | "
              f"横長 cal {mc['len_h']:6.1f} leg {ml['len_h']:6.1f} (真 {mc['len_h_true']:.0f}) | "
              f"縦長 cal {mc['len_v']:6.1f} leg {ml['len_v']:6.1f} (真 {mc['len_v_true']:.0f}) | "
              f"平面RMS cal {mc['plane_rms']:.2f} leg {ml['plane_rms']:.2f} | epi {np.median(epi):.2f}px")

        if not saved:
            out = os.path.join(config.path("triangulation_check"),
                               "chessboard_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".jpg")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            vis_l, vis_r = fl.copy(), fr.copy()
            cv2.drawChessboardCorners(vis_l, pat_l, cl.reshape(-1, 1, 2).astype(np.float32), True)
            cv2.drawChessboardCorners(vis_r, pat_l, cr.reshape(-1, 1, 2).astype(np.float32), True)
            cv2.imwrite(out, np.hstack([cv2.resize(vis_l, (0, 0), fx=0.5, fy=0.5),
                                        cv2.resize(vis_r, (0, 0), fx=0.5, fy=0.5)]))
            print(f"  検出画像を保存: {out}")
            saved = True

    tracker.close()

    if not results:
        print("チェスボードを両カメラで検出できませんでした")
        return

    print(f"\n=== まとめ（{len(results)} フレーム）===")
    depths = np.array([r[0]["depth"] for r in results])
    bins = [(0, 290), (290, 330), (330, 370), (370, 1000)]
    for lo, hi in bins:
        idx = [i for i, d in enumerate(depths) if lo <= d < hi]
        if not idx:
            continue
        def avg(which, key):
            return np.mean([results[i][which][key] for i in idx])
        print(f"奥行 {lo}-{hi}mm（{len(idx)}フレーム）")
        for which, name in ((0, "calibrated"), (1, "legacy    ")):
            print(f"  {name}: 隣接間隔 {avg(which, 'adj_mean'):6.2f}mm（真 {square}）  "
                  f"横長 {avg(which, 'len_h'):6.1f}（真 {results[0][0]['len_h_true']:.0f}）  "
                  f"縦長 {avg(which, 'len_v'):6.1f}（真 {results[0][0]['len_v_true']:.0f}）  "
                  f"平面RMS {avg(which, 'plane_rms'):.2f}mm")


if __name__ == "__main__":
    main()
