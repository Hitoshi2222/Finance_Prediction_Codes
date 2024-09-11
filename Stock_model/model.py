import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# シード値の設定
seed = 777

# Pythonの組み込み乱数生成器
random.seed(seed)

# Numpyの乱数生成器
np.random.seed(seed)

# PyTorchの乱数生成器
torch.manual_seed(seed)

# CUDAの乱数生成器（GPU使用の場合）
torch.cuda.manual_seed(seed)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # パディングの修正
            padding = (kernel_size - 1) * dilation_size // 2
            conv_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,  # ストライドが1であることを明示的に指定
                dilation=dilation_size,
                padding=padding,
            )
            # He初期化を適用
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity="relu")

            layers.append(conv_layer)
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SpatialAttention(nn.Module):
    def __init__(self, N, F, T):
        super().__init__()
        self.W1 = nn.Linear(T, 1, bias=False)
        self.W2 = nn.Linear(F, T, bias=False)
        self.W3 = nn.Linear(F, 1, bias=False)
        self.V_s = nn.Parameter(torch.rand(N, N))
        self.bias = nn.Parameter(torch.rand(N, N))

    def forward(self, H):
        # H is expected to have dimensions [N, F, T] (batch, features, time)
        H_N, H_F, H_T = H.size()
        H_transposed = H.transpose(1, 2)  # Transpose to [N, T, F]

        H_W1 = self.W1(H)
        H_W1 = H_W1.view(H_N, H_F)

        H_w1_W2 = self.W2(H_W1)  # Transform result along feature dimension

        H_W3 = self.W3(H_transposed)  # Transform original along feature dimension
        H_W3 = H_W3.view(H_N, H_T).T
        H_W1_W2_W3 = torch.matmul(H_w1_W2, H_W3)

        # Calculate attention scores
        H_W1_W2_W3_b = torch.sigmoid(H_W1_W2_W3 + self.bias)
        S_hat = torch.matmul(self.V_s, H_W1_W2_W3_b)

        # Normalize attention scores (row-wise normalization)
        attention_scores = F.softmax(S_hat, dim=1)

        return attention_scores


class MetaTraderLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_assets):
        super(MetaTraderLSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Attention Parameters
        self.W5 = nn.Parameter(torch.randn(hidden_dim, 2 * hidden_dim))
        self.W6 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.ve = nn.Parameter(torch.randn(hidden_dim, 1))
        self.W7 = nn.Parameter(torch.randn(hidden_dim))
        self.bm = nn.Parameter(torch.randn(1))

        # LSTMの初期化関数の呼び出し
        self.init_lstm_weights()

        # 初期化後の重みを表示して確認
        self.print_lstm_weights()

    def init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def print_lstm_weights(self):
        print("Initialized LSTM Weights:")
        for name, param in self.lstm.named_parameters():
            print(f"{name}: {param.data}")

    def forward(self, market_features):
        # LSTMで市場特徴をエンコード
        h_t, _ = self.lstm(
            market_features.unsqueeze(0)
        )  # 簡易のためバッチサイズ = 1と仮定

        # h_t = self.layer_norm(h_t)  # Layer Normalizationの適用

        # 時系列の最後の隠れ状態を時間次元全体で繰り返し
        h_T = h_t[:, -1, :].repeat(
            1, h_t.size(1), 1
        )  # シーケンス全体に最後の隠れ状態を繰り返す

        # 時系列のアテンション計算
        combined_features = torch.cat([h_t, h_T], dim=2)

        et = self.ve.t() @ torch.tanh(
            self.W5 @ combined_features.transpose(1, 2)
            + self.W6 @ market_features.unsqueeze(0).transpose(1, 2)
        )

        alpha_t = F.softmax(et.squeeze(), dim=0).unsqueeze(
            1
        )  # シーケンス次元に沿ってSoftmax
        h_t = h_t.view(h_t.shape[1], h_t.shape[2])

        # ショートレシオ（ρ）を計算
        context_vector = torch.sum(
            alpha_t * h_t, dim=0, keepdim=True
        )  # LSTM出力の重み付き合計
        context_vector = context_vector.squeeze()
        rho_t = (
            0.5 * torch.sigmoid(self.W7 @ context_vector + self.bm) + 0.5
        )  # ショートレシオの計算

        return rho_t


class diverse_Trader_Policy(nn.Module):
    def __init__(
        self,
        num_assets,
        num_market,
        hidden_size,
        time_length,
        output_size,
        M,
    ):
        super().__init__()

        # Example usage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tcn = TemporalConvNet(
            hidden_size, [hidden_size, hidden_size, hidden_size]
        ).to(device)
        self.spatial_attention = SpatialAttention(
            num_assets, hidden_size, time_length
        ).to(device)
        self.fc_v = nn.Linear(hidden_size, 1).to(
            device
        )  # Fully connected layer to get v^
        self.fc_final = nn.Linear(num_assets * 2, num_assets).to(
            device
        )  # Combine v^ and xa
        self.market_lstm = MetaTraderLSTMAttention(
            num_market, num_market, num_assets
        ).to(device)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, output_size).to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)
        self.M = M
        self.num_assets = num_assets

    def forward(self, asset_input, market_input, last_portfolio):

        original = asset_input
        asset_output = self.tcn(asset_input)

        attention_out = self.spatial_attention(asset_output)
        asset_output_transformed = asset_output.transpose(0, 2)
        res_output = (
            torch.matmul(asset_output_transformed, attention_out).transpose(0, 2)
            + original
        )

        res_output = res_output.transpose(1, 2)

        # Additional fully connected layer to create v^
        v_hat = self.fc_v(res_output)
        v_hat = v_hat.view(v_hat.shape[0], v_hat.shape[1])

        # Example placeholder for last portfolio vector
        combined_v = torch.cat([v_hat, last_portfolio], dim=0).transpose(0, 1)

        v = torch.sigmoid(self.fc_final(combined_v)).transpose(0, 1)

        portfolio_weights = []

        ## Cacl Long Postion based

        ## Calc Short Postion based

        return portfolio_weights

    def reward(self, args):

        return

    def port_reward(model_returns, args):

        return

    def calculate_loss_function(self, args):

        return
