import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from market_regime_detector import MarketRegimeDetector, MarketRegime

class TradingEnv(gym.Env):
    """
    Custom Gym environment for XAUUSD trading with trailing stops and dynamic exits.
    """
    def __init__(self, df, initial_balance=1000, transaction_cost=0, leverage=50, stop_loss_pct=0.02, scalping_mode=False):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.scalping_mode = scalping_mode

        # Trailing stop parameters
        self.trailing_stop_pct = 0.025
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0

        # Profit taking parameters
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]
        self.take_profit_pct = 0.10
        self.partial_take_profit_pct = 0.02

        # Breakeven stop parameters
        self.breakeven_trigger_pct = 0.015
        self.breakeven_activated = False

        # Dynamic exit parameters
        self.max_holding_period = 24
        self.entry_time = None

        # Calculate technical indicators
        self._calculate_indicators()

        # Action space: continuous action between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: price history (last 10 closes), RSI, MACD, MACD_signal, position, balance
        self.lookback = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback + 5,), dtype=np.float32
        )

        self.trades = []
        
        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector(scalping_mode=scalping_mode)

        self.reset()

    def _calculate_indicators(self):
        """Calculate technical indicators for the dataset"""
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        self.df.fillna(0, inplace=True)

    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.done = False
        self.trades = []

        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0
        self.entry_time = None
        self.breakeven_activated = False

        return self._get_observation()

    def _update_regime_parameters(self):
        """Update trading parameters based on current market regime"""
        current_regime, regime_params = self.regime_detector.detect_regime(self.df, self.current_step)

        self.profit_targets = regime_params['profit_targets']
        self.trailing_stop_pct = regime_params['trailing_stop_pct']
        self.take_profit_pct = regime_params['profit_targets'][-1]
        self.partial_take_profit_pct = regime_params['profit_targets'][1]
        self.breakeven_trigger_pct = regime_params['breakeven_trigger']
        self.max_holding_period = regime_params['max_holding_time']
        self.current_regime = current_regime

    def _get_observation(self):
        prices = self.df.loc[self.current_step - self.lookback:self.current_step - 1, 'Close'].values
        rsi = self.df.loc[self.current_step - 1, 'RSI']
        macd = self.df.loc[self.current_step - 1, 'MACD']
        macd_signal = self.df.loc[self.current_step - 1, 'MACD_signal']
        return np.concatenate([prices, [rsi, macd, macd_signal, self.position, self.balance]])

    def step(self, action, confidence=1.0):
        self._update_regime_parameters()
        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0

        if isinstance(action, np.ndarray):
            action = action[0]

        min_confidence = 0.3
        if confidence < min_confidence:
            action = 0

        effective_action = action * confidence

        exit_reason = self._check_dynamic_exits(current_price)
        if exit_reason:
            profit = (current_price - self.entry_price) * self.position
            self.balance += profit
            self.total_profit += profit

            if exit_reason == 'trailing_stop':
                reward = profit / self.initial_balance * 30
            elif exit_reason == 'take_profit':
                reward = profit / self.initial_balance * 100
            elif 'partial' in exit_reason:
                reward = profit / self.initial_balance * 50
            elif exit_reason == 'max_time':
                reward = profit / self.initial_balance * 10
            else:
                reward = profit / self.initial_balance * 20

            self.trades.append({
                'step': self.current_step,
                'action': 'exit',
                'reason': exit_reason,
                'price': current_price,
                'profit': profit,
                'confidence': confidence
            })
            self._reset_position_state()

        elif self.position == 0:
            if abs(effective_action) > 0.1:
                side = 1 if effective_action > 0 else -1
                position_multiplier = min(abs(effective_action), 1.0)
                confidence_multiplier = confidence ** 0.5
                self.position = side * self.balance * self.leverage * position_multiplier * confidence_multiplier / current_price
                self.entry_price = current_price
                self.entry_time = self.current_step
                self.highest_price_since_entry = current_price
                
                if side > 0:
                    self.trailing_stop_distance = current_price * (1 - self.trailing_stop_pct)
                else:
                    self.trailing_stop_distance = current_price * (1 + self.trailing_stop_pct)
                
                self.breakeven_activated = False
                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy' if side > 0 else 'sell_short',
                    'price': current_price,
                    'position': self.position,
                    'confidence': confidence
                })
        else:
            self._update_trailing_stops(current_price)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def _check_dynamic_exits(self, current_price):
        if self.position == 0:
            return None

        if self.position > 0:
            profit_pct = (current_price - self.entry_price) / self.entry_price
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price

        if not self.breakeven_activated and profit_pct >= self.breakeven_trigger_pct:
            self.breakeven_activated = True
            buffer_pct = 0.0005 # 0.05% buffer for scalping
            if self.position > 0:
                self.trailing_stop_distance = self.entry_price * (1 + buffer_pct)
            else:
                self.trailing_stop_distance = self.entry_price * (1 - buffer_pct)

        for target_pct in sorted(self.profit_targets, reverse=True):
            if profit_pct >= target_pct:
                if target_pct < self.profit_targets[-1]:
                    profit_portion = 0.5
                    exit_reason = f'profit_{target_pct}_partial'
                    self.position *= (1 - profit_portion)
                    return exit_reason
                else:
                    exit_reason = 'take_profit'
                    self.position = 0
                    self._reset_position_state()
                    return exit_reason

        if self.position > 0:
            if current_price <= self.trailing_stop_distance:
                self.position = 0
                self._reset_position_state()
                return 'trailing_stop'
        else:
            if current_price >= self.trailing_stop_distance:
                self.position = 0
                self._reset_position_state()
                return 'trailing_stop'

        if self.entry_time and (self.current_step - self.entry_time) >= self.max_holding_period:
            self.position = 0
            self._reset_position_state()
            return 'max_time'

        stop_loss_limit = -0.005 if self.scalping_mode else -0.02
        if profit_pct <= stop_loss_limit:
            self.position = 0
            self._reset_position_state()
            return 'stop_loss'

        return None

    def _reset_position_state(self):
        self.position = 0
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0
        self.entry_price = 0
        self.entry_time = None
        self.breakeven_activated = False

    def _update_trailing_stops(self, current_price):
        if self.position > 0:
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 - self.trailing_stop_pct)
        else:
            if current_price < self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 + self.trailing_stop_pct)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}, Total Profit: {self.total_profit}")
