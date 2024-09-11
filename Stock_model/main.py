import os
import json
import warnings
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
import datetime
from train import module, TrainParams

# シード値の設定
seed = 777
# PyTorchの乱数生成器
torch.manual_seed(seed)
# CUDAの乱数生成器（GPU使用の場合）
torch.cuda.manual_seed(seed)

warnings.filterwarnings("ignore")


class Config:

    start_period: int = 1000  # バックテスト開始タイミング
    end_period: int = 2000  # バックテスト終了タイミング
    train_period: int = 1000  # 学習期間
    hidden_size = 20  # 隠れ層
    learning_rate = 0.001  # 学習率
    M = 100  # 選択銘柄数
    gamma = 0.01  # 割引率
    decay_rate = 0.01  # 減衰係数
    epoch_number = 50  # 学習回数

    learning_type = "all"  # 学習タイプ
    test_type = "backtest_cross_sectional_momentum"  # テストタイプ

    # expert_data_param
    create_expert_data = True  # 専門家データを模倣するか
    expert_train_ratio = 0  # 専門家データの損失関数の学習率
    # expert_type=1 #1 :cross_sectional_momentum、2 :Buying-Loser Selling-Winner 専門家データの種類

    folder_dir = "results/base/" + datetime.datetime.today().strftime(
        "%Y%m%d_%H%M"
    )  # help="保存ディレクトリ"


if __name__ == "__main__":

    config = Config()
    with open("variable_folder.json", "r") as f:
        valiable_folder = json.load(f)

    if not os.path.exists(config.folder_dir):
        os.makedirs(config.folder_dir)

    asset_input = np.random.rand(100, 50, 10)
    market_input = np.random.rand(100, 50)
    stock_returns = np.random.rand(100, 50)
    expert_dataset = np.random.rand(100, 50)
    date_time = np.random.rand(100, 50)

    # total_length=asset_input.shape[1]
    output_file = pd.DataFrame()
    portfolio_weights = []

    if config.learning_type == "all" or config.create_expert_data == True:

        expert_train_ratio = config.expert_train_ratio
        learning_type = "all"

    else:
        expert_train_ratio = 0
        learning_type = "not_all"

        params = TrainParams(
            num_assets=asset_input.shape[0],
            market_input_shape=market_input.shape[1],
            time_length=asset_input.shape[1],
            model_params=[],
            hidden_size=asset_input.shape[2],
            learning_type=learning_type,
            M=config.M,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            decay_rate=config.decay_rate,
            expert_train_ratio=expert_train_ratio,
            epoch_number=config.epoch_number,
            test_type="train",
        )

        model_params, portfolio_weight = module(
            asset_input,
            market_input,
            stock_returns,
            expert_dataset,
            params,
            last_portfolio=None,
        )

        # テンソルをリストに変換して新しいディクショナリを作成
        model_parameters_list = {k: v.tolist() for k, v in model_params.items()}

        # 保存ファイルのパスを指定
        save_path = "model_parameters_1.json"
        # JSONファイルに保存
        with open(save_path, "w") as f:
            json.dump(model_parameters_list, f, indent=4)

        # パラメータをファイルに保存
        # torch.save(model_params, save_path)

        # 予測フェーズ
        for ind in range(config.start_period, config.end_period, 1):
            asset_input_test = asset_input[:, ind - config.train_period : ind, :]
            market_input_test = market_input[:][ind - config.train_period : ind]
            stock_returns_test = stock_returns[:][ind - config.train_period : ind]

            if config.learning_type == "all" or config.create_expert_data == True:
                expert_dataset_test = expert_dataset[:][ind - config.train_period : ind]
            else:
                expert_dataset_test = []

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            params = TrainParams(
                num_assets=asset_input_test.shape[0],
                market_input_shape=market_input_test.shape[1],
                time_length=asset_input_test.shape[1],
                model_params=model_params,
                hidden_size=asset_input_test.shape[2],
                learning_type=learning_type,
                M=config.M,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                decay_rate=config.decay_rate,
                expert_train_ratio=expert_train_ratio,
                epoch_number=config.epoch_number,
                test_type="test",
            )

            portfolio_weight = module(
                asset_input_test,
                market_input_test,
                stock_returns_test,
                expert_dataset_test,
                params,
                last_portfolio=portfolio_weight,
            )
            portfolio_weight_cpu = portfolio_weight[:, -1].to("cpu").tolist()
            portfolio_weight = [
                round(float(num), 5) if isinstance(num, (int, float)) else num
                for num in portfolio_weight_cpu
            ]
