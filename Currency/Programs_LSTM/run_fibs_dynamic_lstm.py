import time
import datetime
import argparse
import pandas as pd
import json
import warnings
import numpy as np

from fibs_main_lstm_opt_ga import learn

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        choices=[],
        default="",
        help="予測対象",
    )

    parser.add_argument(
        "--rolling",
        type=str,
        default="2022-01-01",
        help="予測開始時期（ローリングスタート時期）",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="予測終了時期（ローリング完了タイミング）",
    )

    parser.add_argument("--val", type=int, default=60, help="評価期間（検証期間）")

    parser.add_argument(
        "--dim",
        type=int,
        default=150,
        help="特徴量数（スタート台）, -999: 全て,  -1:（特定の）特徴量の指定",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="results/" + datetime.datetime.today().strftime("%Y%m%d_%H%M"),
        help="保存ディレクトリ",
    )
    parser.add_argument(
        "--pos",
        type=str,
        choices=["y", "n"],
        default="y",
        help="ポジション・損益計算フラグ",
    )

    # optuna_flag
    parser.add_argument(
        "--optuna_flag",
        type=str,
        choices=["y", "n"],
        default="n",
        help="ポジション・損益計算フラグ",
    )

    args = parser.parse_args()

    with open("config.json", "r") as f:
        pre_hyperparameters = json.load(f)

    start_time = time.time()

    if args.target_d != 0:
        target = f"{args.target_d}D"
    else:
        target = args.target

    X = np.random.rand(100, 50)
    y = np.random.rand(100, 50)
    y_col = np.random.rand(100, 50)

    # バックテスト期間の指定に応じたデータ抽出
    if args.end_date is not None:
        X = X[X.index <= args.end_date]
        y = y[y.index <= args.end_date]

    # 最適特徴量の計算（ローリング）
    learn(
        X,
        y,
        target,
        pre_hyperparameters,
        opt_flag=args.optuna_flag,
        rolling_start=args.rolling,
        val_period=args.val,
        out_dir=args.dir,
    )
