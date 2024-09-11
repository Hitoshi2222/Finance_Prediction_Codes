from dataclasses import dataclass
from model import diverse_Trader_Policy
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

# シード値の設定
seed = 777
# PyTorchの乱数生成器
torch.manual_seed(seed)
# CUDAの乱数生成器（GPU使用の場合）
torch.cuda.manual_seed(seed)


@dataclass
class TrainParams:
    num_assets: int
    market_input_shape: int
    time_length: int
    model_params: any
    hidden_size: int
    learning_type: str
    M: int
    learning_rate: float
    gamma: float
    decay_rate: float
    expert_train_ratio: float
    epoch_number: int
    test_type: str


def generate_last_portfolio(num_rows, num_cols, device):
    # 正の要素を生成
    positive_elements = torch.rand(num_rows, num_cols // 2).to(device)

    # 負の要素を生成
    negative_elements = -torch.rand(num_rows, num_cols - num_cols // 2).to(device)

    # 正の要素の合計を1に正規化
    positive_elements /= positive_elements.sum(dim=1, keepdim=True)

    # 負の要素の合計を-1に正規化
    negative_elements /= -negative_elements.sum(dim=1, keepdim=True)

    # 正の要素と負の要素を結合
    last_portfolio = torch.cat([positive_elements, negative_elements], dim=1).to(device)

    # 要素をシャッフルしてランダムな順序にする
    indices = torch.randperm(last_portfolio.size(1)).to(device)
    last_portfolio = last_portfolio[:, indices]

    return last_portfolio


def module(
    asset_input,
    market_input,
    stock_returns,
    expert_actions,
    params: TrainParams,
    last_portfolio=None,
):

    output_size = 2 * params.num_assets

    # Initialize the model

    model = diverse_Trader_Policy(
        num_assets=params.num_assets,
        num_market=params.market_input_shape,
        hidden_size=params.hidden_size,
        time_length=params.time_length,
        output_size=output_size,
        M=params.M,
    )

    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習時のポートフォリオ初期化用に
    if params.test_type == "train":
        last_portfolio = generate_last_portfolio(
            asset_input.shape[0], asset_input.shape[1], device
        )

    # 正規化されたデータを格納する配列を準備
    asset_input_normalized = np.empty_like(asset_input)

    # 1次元目（バッチ）ごとにループ
    for i in range(asset_input.shape[0]):
        scaler = MinMaxScaler()
        # i番目のバッチを取り出し、2次元に変形
        batch = asset_input[i]

        # 特徴量ごとに正規化
        batch_normalized = scaler.fit_transform(batch)

        # 正規化されたバッチを戻す
        asset_input_normalized[i] = batch_normalized

    scaler = MinMaxScaler()
    market_input_normalized = scaler.fit_transform(market_input)

    # NumPy配列をPyTorchテンソルに変換
    asset_input_normalized = torch.tensor(
        asset_input_normalized, dtype=torch.float32
    ).to(device)
    asset_input_normalized = asset_input_normalized.permute(0, 2, 1)
    market_input_normalized = torch.tensor(
        market_input_normalized, dtype=torch.float32
    ).to(device)

    if params.test_type == "train":

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

        # Training loop
        for epoch in range(params.epoch_number):  # Number of epochs

            optimizer.zero_grad()

            portfolio_weight = model.forward(
                asset_input_normalized, market_input_normalized, last_portfolio
            )

            # Calculate reward
            rewards = model.reward(portfolio_weight, stock_returns, params.decay_rate)

            # Calculate loss
            loss = model.calculate_loss_function(
                portfolio_weight,
                rewards,
                expert_actions,
                params.gamma,
                params.expert_train_ratio,
                params.learning_type,
            )

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if epoch % 10 == 0:

                print(f"Epoch {epoch}, Loss: {loss.item()}")

        return model.state_dict(), portfolio_weight  # Return the model parameters

    elif params.test_type == "test":

        model.load_state_dict(params.model_params)

        # 予測フェーズ
        with torch.no_grad():
            portfolio_weight = model.forward(
                asset_input_normalized, market_input_normalized, last_portfolio
            )

        return portfolio_weight
