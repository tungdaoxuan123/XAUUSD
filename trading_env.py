import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    LONG-ONLY Gym environment for XAUUSD trading.

    Fixed 2:1 risk-reward ratio with ATR-derived dynamic pip distances.
    Discrete action space: 0 = WAIT/CLOSE, 1 = ENTER/HOLD LONG.
    Reward: +2 for TP hit, -1 for SL hit, 0 for waiting.
    """
    def __init__(self, df, initial_balance=1000, transaction_cost=0, leverage=50,
                 atr_mult_sl=1.0):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.atr_mult_sl = atr_mult_sl

        self._calculate_indicators()

        self.action_space = spaces.Discrete(2)

        self.lookback = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback + 5,), dtype=np.float32
        )

        self.trades = []
        self.reset()

    def _calculate_indicators(self):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        ema12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        # ATR for dynamic SL/TP distances
        hl = self.df['High'] - self.df['Low'] if 'High' in self.df.columns else self.df['Close'] * 0.001
        self.df['ATR'] = pd.Series(
            np.maximum(hl.abs() if not isinstance(hl, pd.Series) else hl,
                       np.maximum(
                           (self.df['High'] - self.df['Close'].shift()).abs() if 'High' in self.df.columns else hl,
                           (self.df['Low'] - self.df['Close'].shift()).abs() if 'Low' in self.df.columns else hl,
                       )
                      )
        ).rolling(14).mean()

        self.df.fillna(0, inplace=True)

    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0
        self.total_profit = 0
        self.done = False
        self.trades = []
        return self._get_observation()

    def _get_observation(self):
        prices = self.df.loc[self.current_step - self.lookback:self.current_step - 1, 'Close'].values
        rsi = self.df.loc[self.current_step - 1, 'RSI']
        macd = self.df.loc[self.current_step - 1, 'MACD']
        macd_signal = self.df.loc[self.current_step - 1, 'MACD_signal']
        return np.concatenate([prices, [rsi, macd, macd_signal, self.position, self.balance]])

    def step(self, action):
        current_price = float(self.df.loc[self.current_step, 'Close'])
        atr = float(self.df.loc[self.current_step, 'ATR']) if 'ATR' in self.df.columns else current_price * 0.01
        reward = 0.0

        if isinstance(action, (list, np.ndarray)):
            action = int(action[0])

        if self.position != 0:
            if self.position > 0:
                high = float(self.df.loc[self.current_step, 'High']) if 'High' in self.df.columns else current_price
                low  = float(self.df.loc[self.current_step, 'Low']) if 'Low' in self.df.columns else current_price

                if high >= self.tp_price:
                    profit = (self.tp_price - self.entry_price) * self.position
                    self.balance += profit
                    self.total_profit += profit
                    reward = 2.0
                    self.trades.append({
                        'step': self.current_step, 'action': 'tp_hit',
                        'price': self.tp_price, 'profit': profit,
                    })
                    self._reset_position()
                elif low <= self.sl_price:
                    profit = (self.sl_price - self.entry_price) * self.position
                    self.balance += profit
                    self.total_profit += profit
                    reward = -1.0
                    self.trades.append({
                        'step': self.current_step, 'action': 'sl_hit',
                        'price': self.sl_price, 'profit': profit,
                    })
                    self._reset_position()
            else:
                self._reset_position()

        elif action == 1:
            sl_dist = max(atr * self.atr_mult_sl, current_price * 0.001)
            tp_dist = 2.0 * sl_dist
            self.entry_price = current_price
            self.sl_price = current_price - sl_dist
            self.tp_price = current_price + tp_dist
            position_size = self.balance * self.leverage / current_price
            self.position = position_size
            self.trades.append({
                'step': self.current_step, 'action': 'long_entry',
                'price': current_price, 'sl': self.sl_price, 'tp': self.tp_price,
            })

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, False, {}

    def _reset_position(self):
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
              f"Position: {self.position:.2f}, Total Profit: {self.total_profit:.2f}")
