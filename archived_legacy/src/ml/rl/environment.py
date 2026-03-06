import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

class TradingEnv:
    """Gym-style environment for RL trading agent."""
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0, 
                 transaction_fee: float = 0.0001, window_size: int = 60):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.reset()

    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.inventory = 0
        self.current_step = self.window_size
        self.history = []
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        # Window of price data + position info
        obs = self.data.iloc[self.current_step - self.window_size:self.current_step].values
        # Normalize window (simple z-score or min-max)
        obs_norm = (obs - np.mean(obs, axis=0)) / (np.std(obs, axis=0) + 1e-8)
        
        # Append position info: [inventory, normalized_balance]
        state = np.append(obs_norm.flatten(), [self.inventory, self.balance / self.initial_balance])
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # action: 0=Hold, 1=Buy, 2=Sell
        current_price = self.data.iloc[self.current_step]['close']
        prev_value = self.balance + self.inventory * current_price
        
        if action == 1: # Buy
            shares_to_buy = self.balance // (current_price * (1 + self.transaction_fee))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                self.inventory += shares_to_buy
        elif action == 2: # Sell
            if self.inventory > 0:
                revenue = self.inventory * current_price * (1 - self.transaction_fee)
                self.balance += revenue
                self.inventory = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        new_price = self.data.iloc[self.current_step]['close']
        current_value = self.balance + self.inventory * new_price
        reward = (current_value - prev_value) / prev_value # Percentage return as reward
        
        obs = self._get_observation()
        info = {'total_value': current_value, 'inventory': self.inventory}
        
        return obs, reward, done, info

