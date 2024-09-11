import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # デバイス情報を出力

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

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class QNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.temporal_attention = TemporalAttention(hidden_size)
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        lstm_out, _ = self.lstm(state)
        context_vector, _ = self.temporal_attention(lstm_out)
        context_vector = context_vector.unsqueeze(1)
        x = torch.relu(self.fc1(context_vector)).T
        q_values = self.fc2(x.T)
        return q_values

class DQNAgent:
    def __init__(self, market_feature_size, reward_size, action_size):
        self.state_size = market_feature_size + reward_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_every = 5

        self.q_network = QNetwork(self.state_size, action_size).to(device)
        self.target_network = QNetwork(self.state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).T.to(device)
        act_values = self.q_network(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, reward):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in minibatch]).squeeze(-1).to(device)
        actions = torch.LongTensor([e[1] for e in minibatch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in minibatch]).squeeze(-1).to(device)
        dones = torch.FloatTensor([e[4] for e in minibatch]).to(device)
        reward = torch.FloatTensor(reward).to(device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = reward + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.q_network.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.q_network.state_dict(), name)

@dataclass
class TrainParams:
    market_feature_size: int
    reward_size: int
    action_size: int
    episodes: int
    
class DQN:
    def __init__(self, env, params:TrainParams):
        self.env = env
        self.market_feature_size = params.market_feature_size
        self.reward_size = params.reward_size
        self.action_size = params.action_size
        self.agent = DQNAgent(params.market_feature_size, params.reward_size, params.action_size)

    def train(self, market_feature, reward, params:TrainParams):
        for e in range(params.episodes):
            state = self.env.reset()

            for t in range(len(market_feature)):
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                
                if done:
                    self.agent.update_target_network()
                    print(f"episode: {e}/{params.episodes}, score: {t}, e: {self.agent.epsilon:.2}")
                    break

                self.agent.replay(reward)
                
                state = next_state
        
        return self.agent.q_network.state_dict()
    
    def pred(market_feature, reward, action_size, model_params):
        market_feature = market_feature.reshape(-1, 1)
        state = np.concatenate((market_feature, reward))
        q_network = QNetwork(state.shape[0], action_size).to(device)
        q_network.load_state_dict(model_params)
        state = torch.FloatTensor(state).T.to(device)
        act_values = q_network(state)
        action = torch.argmax(act_values[0]).item()
        
        return action
