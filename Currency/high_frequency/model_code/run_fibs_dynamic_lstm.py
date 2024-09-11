import time
import datetime
import argparse
import pandas as pd
import json
import warnings
import numpy as np

from fibs_main_lstm_opt_ga import learn, load_files

# from meta_labeling import quantabeling

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        choices=[],
        default="",
        help="予測対象:何分後のレートを予測するか",
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

    with open("variable_folder.json", "r") as f:
        valiable_folder = json.load(f)

    start_time = time.time()

    X, y, y_orig = load_files()

    # 最適特徴量の計算（ローリング）
    _precision = learn(
        X,
        y,
        y_orig,
        args.target,
        pre_hyperparameters,
        opt_flag=args.optuna_flag,
        rolling_start=args.rolling,
        val_period=args.val,
        out_dir=args.dir,
    )
