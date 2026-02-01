import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import torch
import torch.nn as nn

try:
    from ripser import ripser
    TDA_AVAILABLE = True
except:
    TDA_AVAILABLE = False

class MarketRegime(Enum):
    TRENDING_UP = "up"
    TRENDING_DOWN = "down"
    MEAN_REVERTING = "mr"
    HIGH_VOLATILITY = "hv"
    CHAOTIC = "ch"

@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    direction: int
    confidence: float
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    regime: MarketRegime

class TDAFeatureExtractor:
    def __init__(self, dim=10):
        self.dim = dim
        self.scaler = StandardScaler()
        
    def embed(self, series, tau=1):
        n = len(series) - (self.dim - 1) * tau
        if n <= 0: return np.array([])
        return np.array([series[i:i + self.dim * tau:tau] for i in range(n)])
    
    def extract_features(self, prices):
        ret = np.diff(np.log(prices + 1e-10))
        if len(ret) < self.dim * 2:
            return {'h0_mean': 0, 'h0_max': 0, 'h0_std': 0, 'h1_mean': 0, 'h1_count': 0,
                    'mean_reversion': 0, 'volatility': 0, 'momentum': 0, 'trend': 0}
        pc = self.embed(ret)
        if len(pc) == 0:
            return {'h0_mean': 0, 'h0_max': 0, 'h0_std': 0, 'h1_mean': 0, 'h1_count': 0,
                    'mean_reversion': 0, 'volatility': 0, 'momentum': 0, 'trend': 0}
        from scipy.spatial.distance import pdist
        d = pdist(pc)
        return {
            'h0_mean': np.mean(d), 'h0_max': np.max(d), 'h0_std': np.std(d),
            'h1_mean': np.median(d), 'h1_count': len(d) // 10,
            'mean_reversion': np.tanh(np.std(d) * 10),
            'volatility': np.std(ret), 'momentum': np.mean(ret[-5:]) if len(ret) >= 5 else 0,
            'trend': (prices[-1] - prices[0]) / (prices[0] + 1e-10)
        }

class SimpleNN(nn.Module):
    def __init__(self, inp=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3)
        )
    def forward(self, x): return self.net(x)
    def predict_proba(self, x):
        with torch.no_grad(): return torch.softmax(self.forward(x), dim=1)

class TDANeuralStrategy:
    def __init__(self, lookback=50, dim=10):
        self.lookback = lookback
        self.tda = TDAFeatureExtractor(dim=dim)
        self.model = SimpleNN()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.names = ['h0_mean', 'h0_max', 'h0_std', 'h1_mean', 'h1_count',
                      'mean_reversion', 'volatility', 'momentum', 'trend']
    
    def prepare_features(self, prices):
        f = self.tda.extract_features(prices)
        return np.array([f[n] for n in self.names]).reshape(1, -1)
    
    def generate_signal(self, prices, current_price):
        if len(prices) < self.lookback:
            return self._no_signal(current_price)
        feat = self.prepare_features(prices[-self.lookback:])
        if not self.is_trained:
            return self._rule_signal(feat, current_price)
        fs = self.scaler.transform(feat)
        probs = self.model.predict_proba(torch.FloatTensor(fs)).numpy()[0]
        direction = int(np.argmax(probs)) - 1
        conf = float(probs[np.argmax(probs)])
        if conf < 0.55: direction = 0
        return self._make_signal(direction, conf, current_price, feat[0])
    
    def _rule_signal(self, feat, price):
        f = dict(zip(self.names, feat[0]))
        if f['mean_reversion'] > 0.5 and f['momentum'] < -0.005:
            return self._make_signal(1, 0.6, price, feat[0])
        elif f['mean_reversion'] > 0.5 and f['momentum'] > 0.005:
            return self._make_signal(-1, 0.6, price, feat[0])
        elif f['trend'] > 0.01: return self._make_signal(1, 0.55, price, feat[0])
        elif f['trend'] < -0.01: return self._make_signal(-1, 0.55, price, feat[0])
        return self._make_signal(0, 0.5, price, feat[0])
    
    def _make_signal(self, d, conf, price, feat):
        vol = max(feat[6], 0.01)
        sl = price * (1 - 2 * vol) if d == 1 else price * (1 + 2 * vol) if d == -1 else price
        tp = price * (1 + 3 * vol) if d == 1 else price * (1 - 3 * vol) if d == -1 else price
        ps = max(0, min(0.1, conf * 0.15 - 0.05)) if d != 0 else 0
        return TradeSignal(datetime.now(), "", d, conf, ps, price, sl, tp, MarketRegime.CHAOTIC)
    
    def _no_signal(self, price):
        return TradeSignal(datetime.now(), "", 0, 0, 0, price, price, price, MarketRegime.CHAOTIC)
    
    def train(self, df, returns, epochs=50):
        print("Preparing training data...")
        X, y = [], []
        for i in range(self.lookback, len(df) - 1):
            prices = df['close'].values[i-self.lookback:i]
            X.append(self.prepare_features(prices)[0])
            r = returns[i] if i < len(returns) else 0
            y.append(2 if r > 0.003 else (0 if r < -0.003 else 1))
        X, y = np.array(X), np.array(y)
        print(f"Samples: {len(X)}, Long: {sum(y==2)}, Flat: {sum(y==1)}, Short: {sum(y==0)}")
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        Xt, yt = torch.FloatTensor(Xs), torch.LongTensor(y)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for e in range(epochs):
            opt.zero_grad()
            out = self.model(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()
            if (e + 1) % 10 == 0:
                acc = (out.argmax(1) == yt).float().mean()
                print(f"Epoch {e+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")
        self.is_trained = True
        print("Training complete!")
    
    def save(self, path="models/tda_neural_model.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'scaler': self.scaler, 'trained': self.is_trained}, path)
        print(f"Saved to {path}")
    
    def load(self, path="models/tda_neural_model.pt"):
        if os.path.exists(path):
            c = torch.load(path)
            self.model.load_state_dict(c['model'])
            self.scaler = c['scaler']
            self.is_trained = c['trained']
            return True
        return False
