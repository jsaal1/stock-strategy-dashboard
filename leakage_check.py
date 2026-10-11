#!/usr/bin/env python3
"""
Reproduces the leakage finding described in the README.

Runs the same Random Forest walk-forward backtest four ways:
  1. original setup: raw price-level features, no purge (the June 2025 version)
  2. raw features with a 21-day purge
  3. scale-free features, no purge
  4. scale-free features with a 21-day purge (the current project.py)

Usage: python leakage_check.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TICKERS = ['AAPL', 'NVDA', 'WMT', 'KO', 'XOM', 'CVX', 'JPM', 'BRK-B', 'JNJ', 'LLY']
DATA_DIR = "stock_data_csv"
HORIZON, TRAIN_WINDOW, TEST_SIZE = 21, 252, 21


def load(ticker):
    df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv", header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    return df.sort_index()


def raw_features(df):
    """The original feature set: OHLCV plus indicators, most of them in price units."""
    p = df['Close']
    delta = p.diff()
    out = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    out['RSI'] = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean())
    out['MACD'] = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    out['MACDsignal'] = out['MACD'].ewm(span=9, adjust=False).mean()
    out['MACDhist'] = out['MACD'] - out['MACDsignal']
    out['Volatility'] = p.rolling(21).std()
    out['SMA20'], out['SMA50'] = p.rolling(20).mean(), p.rolling(50).mean()
    out['EMA20'], out['EMA50'] = p.ewm(span=20, adjust=False).mean(), p.ewm(span=50, adjust=False).mean()
    out['BBupper'] = out['SMA20'] + 2 * p.rolling(20).std()
    out['BBlower'] = out['SMA20'] - 2 * p.rolling(20).std()
    out['Momentum10'] = p - p.shift(10)
    r = p.pct_change()
    out['Sharpe21'] = r.rolling(21).mean() / r.rolling(21).std()
    return out


def scale_free_features(df):
    """The current feature set (same definitions as project.py)."""
    p = df['Close']
    r = p.pct_change()
    delta = p.diff()
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    sma20, std20 = p.rolling(20).mean(), p.rolling(20).std()
    return pd.DataFrame({
        'RSI14': 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()),
        'MACD': macd / p,
        'MACDhist': (macd - macd.ewm(span=9, adjust=False).mean()) / p,
        'Volatility21': r.rolling(21).std(),
        'DistSMA20': p / sma20 - 1,
        'DistSMA50': p / p.rolling(50).mean() - 1,
        'EMA20vsEMA50': p.ewm(span=20, adjust=False).mean() / p.ewm(span=50, adjust=False).mean() - 1,
        'BollingerPctB': (p - (sma20 - 2 * std20)) / (4 * std20),
        'Return10': p.pct_change(10),
        'Sharpe21': r.rolling(21).mean() / r.rolling(21).std(),
    })


def backtest(df, make_features, purge):
    X = make_features(df)
    fwd = df['Close'].shift(-HORIZON) / df['Close'] - 1
    keep = X.notna().all(axis=1) & fwd.notna()
    X, y, close = X[keep], (fwd[keep] > 0).astype(int), df['Close'][keep]

    y_true, y_pred, idx = [], [], []
    for start in range(0, len(X) - TRAIN_WINDOW - TEST_SIZE + 1, TEST_SIZE):
        end = start + TRAIN_WINDOW
        m = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        m.fit(X.iloc[start:end - purge], y.iloc[start:end - purge])
        y_pred.extend(m.predict(X.iloc[end:end + TEST_SIZE]))
        y_true.extend(y.iloc[end:end + TEST_SIZE])
        idx.extend(X.index[end:end + TEST_SIZE])
    y_true, y_pred, c = np.array(y_true), np.array(y_pred), close.loc[idx]

    # Same non-overlapping 21-day trade rule as project.py, before costs
    wealth, i = 1.0, 0
    while i < len(c) - HORIZON:
        if y_pred[i] == 1:
            wealth *= c.iloc[i + HORIZON] / c.iloc[i]
            i += HORIZON
        else:
            i += 1
    return dict(accuracy=(y_true == y_pred).mean(), always_up=y_true.mean(),
                strategy=wealth - 1, buy_hold=c.iloc[-1] / c.iloc[0] - 1)


if __name__ == '__main__':
    data = {t: load(t) for t in TICKERS}
    setups = [("Raw features, no purge (original)", raw_features, 0),
              ("Raw features, 21-day purge", raw_features, HORIZON),
              ("Scale-free features, no purge", scale_free_features, 0),
              ("Scale-free features, 21-day purge (current)", scale_free_features, HORIZON)]
    rows = []
    for name, feats, purge in setups:
        res = pd.DataFrame([backtest(data[t], feats, purge) for t in TICKERS], index=TICKERS)
        beats = (res['strategy'] > res['buy_hold']).sum()
        rows.append({'setup': name, **res.mean().to_dict(), 'beats_buy_hold': f"{beats}/{len(TICKERS)}"})
        print(f"\n{name}\n{res.round(3).to_string()}")
    print("\nMean over tickers:")
    print(pd.DataFrame(rows).set_index('setup').round(3).to_string())
