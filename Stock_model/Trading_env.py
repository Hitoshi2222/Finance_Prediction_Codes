import numpy as np

class TradingEnvironment:
    def __init__(self, market_feature_size, portfolio_size,market_feature,performance_all,action_size):
        self.market_feature_size = market_feature_size
        self.portfolio_size = portfolio_size
        self.action_size = action_size
        self.current_step = 0
        self.done = False
        self.market_data = market_feature  # 仮の市場データを生成
        self.performance = performance_all

    def reset(self):
        self.current_step = 0
        self.done = False
        initial_state= self._get_state()
        return initial_state

    #def _generate_market_data(self):
        # 仮の市場データを生成（実際のデータに置き換え可能）
    #    return np.random.rand(1000, self.market_feature_size)

    def _get_state(self):
        market_features = self.market_data[self.current_step][:].reshape(-1, 1)
        performance = self.performance[:,self.current_step]
        state = np.concatenate((market_features, performance))
        return state

    def step(self, action):
        reward = self._calculate_reward(action)
        self.current_step += 1

        if self.current_step >= len(self.market_data) - 1:
            self.done = True

        next_state = self._get_state()

        return next_state, reward, self.done, {}

    def _calculate_reward(self, action):
        reward = self.performance[action,self.current_step]
        return reward
