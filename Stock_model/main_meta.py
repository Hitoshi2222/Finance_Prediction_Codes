import os
import json
import warnings
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
import datetime
from meta_policy import DQN, TrainParams, QNetwork
from Trading_env import TradingEnvironment as trading_env

# シード値の設定
seed = 777
# PyTorchの乱数生成器
torch.manual_seed(seed)
# CUDAの乱数生成器（GPU使用の場合）
torch.cuda.manual_seed(seed)

warnings.filterwarnings("ignore")


class Config:

    start_period: int = 1000  # 2086
    end_period: int = 2000  # 3391
    train_period: int = 60
    retrain_period: int = 40

    model_number = 3
    portfolio_size = 3
    action_size = 3
    episodes = 5

    rolling_reward_start = 0

    learning_type = "reinforcement"
    test_type = "meta_backtest"

    folder_dir = "results/meta/" + datetime.datetime.today().strftime(
        "%Y%m%d_%H%M"
    )  # help="保存ディレクトリ"


if __name__ == "__main__":

    config = Config()
    with open("variable_folder_meta.json", "r") as f:
        valiable_folder = json.load(f)

    if not os.path.exists(config.folder_dir):
        os.makedirs(config.folder_dir)

    market_input = np.random.rand(100, 50)
    stock_returns = np.random.rand(100, 50)
    model_rewards = np.random.rand(100, 50)
    portfolio_weights = np.random.rand(100, 50)
    date_time = np.random.rand(100, 50)

    # total_length=asset_input.shape[1]
    output_file = pd.DataFrame()
    actions = []
    portfolio_weights_selected = []

    params = TrainParams(
        market_feature_size=market_input.shape[1],
        reward_size=model_rewards.shape[0],
        action_size=config.action_size,
        episodes=config.episodes,
    )

    env = trading_env(
        market_input.shape[0],
        config.portfolio_size,
        market_input,
        model_rewards,
        config.action_size,
    )
    DQN_model = DQN(env, params)
    model_params = DQN_model.train(market_input, model_rewards, params)

    # 予測フェーズ
    for ind in range(config.start_period, config.end_period, 1):
        t = ind + +1
        t_reward = ind + config.rolling_reward_start + config.train_period + 1

        market_input_test = market_input[:][t]
        model_rewards_test = model_rewards[:, t_reward]
        portfolio_weights_test = portfolio_weights[:, t_reward]

        action = DQN.pred(
            market_input_test, model_rewards_test, config.action_size, model_params
        )
        actions.append(action)
        portfolio_weights_selected.append(portfolio_weights_test[action])
