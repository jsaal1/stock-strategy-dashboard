#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:40:53 2025
Revised Oct 2026: purged walk-forward validation, scale-free features,
baseline comparison, transaction costs and risk metrics.

@author: jacob
"""

# %%
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# %%
# Configuration
tickers = ['AAPL', 'NVDA', 'WMT', 'KO', 'XOM', 'CVX', 'JPM', 'BRK-B', 'JNJ', 'LLY']
DATA_DIR = "stock_data_csv"
HORIZON = 21          # prediction horizon in trading days (~1 month)
TRAIN_WINDOW = 252    # rolling training window (~1 year)
TEST_SIZE = 21        # rows predicted per walk-forward step
PURGE = HORIZON       # rows dropped from the end of each training window (see README)
COST_BPS = 5          # transaction cost per side, in basis points
DOWNLOAD = False      # True re-downloads data from Yahoo Finance (results will change)
SHOW_PLOTS = False    # True shows matplotlib plots when run as a script

# %%
# Download the past 5 years of data for each ticker (only if requested or missing)
if DOWNLOAD or not all(os.path.exists(f"{DATA_DIR}/{t}.csv") for t in tickers):
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    ohlcv_keywords = ['Open', 'High', 'Low', 'Close', 'Volume']
    for ticker in tickers:
        df = yf.download(ticker, period='5y')
        df.dropna(inplace=True)
        ohlcv_cols = [col for col in df.columns if any(key in col for key in ohlcv_keywords)]
        df[ohlcv_cols].to_csv(f"{DATA_DIR}/{ticker}.csv", index=True)

# %%
# Load data from saved CSV files
data = {}
for ticker in tickers:
    df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv", header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [col[0] for col in df.columns.values]   # 'Close', 'High', ...
    data[ticker] = df.sort_index()

# %%
# Feature engineering
# Every feature is scale-free (a ratio, return or bounded oscillator). Raw price
# levels such as Close or SMA50 trend over time, so a tree model trained on last
# year's price range cannot extrapolate to this year's; they are deliberately excluded.
FEATURES = ['RSI14', 'MACD', 'MACDhist', 'Volatility21', 'DistSMA20', 'DistSMA50',
            'EMA20vsEMA50', 'BollingerPctB', 'Return10', 'Sharpe21']

def add_features(df, price_col='Close'):
    p = df[price_col]
    ret = p.pct_change()

    # RSI (14)
    delta = p.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI14'] = 100 - 100 / (1 + avg_gain / avg_loss)

    # MACD, normalised by price
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd / p
    df['MACDhist'] = (macd - signal) / p

    # Volatility of daily returns (21-day)
    df['Volatility21'] = ret.rolling(21).std()

    # Distance from moving averages
    df['DistSMA20'] = p / p.rolling(20).mean() - 1
    df['DistSMA50'] = p / p.rolling(50).mean() - 1
    df['EMA20vsEMA50'] = p.ewm(span=20, adjust=False).mean() / p.ewm(span=50, adjust=False).mean() - 1

    # Bollinger %B: 0 at the lower band, 1 at the upper band
    sma20, std20 = p.rolling(20).mean(), p.rolling(20).std()
    df['BollingerPctB'] = (p - (sma20 - 2 * std20)) / (4 * std20)

    # Momentum and short-term risk-adjusted return
    df['Return10'] = p.pct_change(10)
    df['Sharpe21'] = ret.rolling(21).mean() / ret.rolling(21).std()
    return df

# %%
# Apply features and generate the target: does the price rise over the next HORIZON days?
for ticker, df in data.items():
    df = add_features(df)
    df['FwdReturn'] = df['Close'].shift(-HORIZON) / df['Close'] - 1
    # The last HORIZON rows have no known future price. Drop them rather than
    # letting the comparison with NaN silently label them as "down".
    df = df.dropna(subset=FEATURES + ['FwdReturn'])
    df['Target'] = (df['FwdReturn'] > 0).astype(int)
    data[ticker] = df

# %%
# Purged walk-forward validation
def walk_forward_validation(X, y, window_size=TRAIN_WINDOW, test_size=TEST_SIZE, purge=PURGE):
    """
    Train on a rolling window and predict the next `test_size` rows.

    The label of row t depends on prices up to t + HORIZON. Without a purge, the
    last HORIZON training labels are computed from prices inside the test window,
    so the model is partly trained on the answers. Dropping the last `purge`
    rows of each training window removes that overlap.
    """
    y_true, y_pred, test_idx, importances = [], [], [], []
    for start in range(0, len(X) - window_size - test_size + 1, test_size):
        end_train = start + window_size
        end_test = end_train + test_size

        X_train = X.iloc[start:end_train - purge]
        y_train = y.iloc[start:end_train - purge]
        X_test, y_test = X.iloc[end_train:end_test], y.iloc[end_train:end_test]

        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                       random_state=42)
        model.fit(X_train, y_train)

        y_true.extend(y_test)
        y_pred.extend(model.predict(X_test))
        test_idx.extend(X_test.index)
        importances.append(model.feature_importances_)

    return np.array(y_true), np.array(y_pred), pd.DatetimeIndex(test_idx), np.array(importances)

# %%
# Run model on each stock and store predictions with their dates
results = {}
for ticker in tickers:
    df = data[ticker]
    y_true, y_pred, idx, importances = walk_forward_validation(df[FEATURES], df['Target'])
    results[ticker] = (y_true, y_pred, idx, importances)

# %%
# Backtest: daily equity curves with transaction costs
def strategy_equity(df, prediction, horizon=HORIZON, cost_bps=COST_BPS):
    """
    Non-overlapping trades: on a day the model predicts "up", buy at that close
    and hold for `horizon` days; otherwise stay in cash and check the next day.
    Returns the daily equity curve and the daily position (1 = invested).
    """
    position = np.zeros(len(df))
    i = 0
    n_trades = 0
    while i < len(df) - horizon:
        if prediction[i] == 1:
            position[i + 1:i + horizon + 1] = 1   # invested from close i to close i + horizon
            n_trades += 1
            i += horizon
        else:
            i += 1
    daily_ret = df['Close'].pct_change().fillna(0).values
    strat_ret = position * daily_ret
    # Charge costs on every change of position (entry and exit)
    trades = np.abs(np.diff(np.concatenate([[0], position])))
    strat_ret = strat_ret - trades * cost_bps / 1e4
    equity = pd.Series(np.cumprod(1 + strat_ret), index=df.index)
    return equity, pd.Series(position, index=df.index), n_trades

def buy_hold_equity(df):
    return df['Close'] / df['Close'].iloc[0]

def sharpe(equity):
    r = equity.pct_change().dropna()
    return np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan

def max_drawdown(equity):
    return (equity / equity.cummax() - 1).min()

# %%
# Evaluate and summarise performance per stock
summary_stats = []
for ticker, (y_true, y_pred, idx, importances) in results.items():
    df = data[ticker].loc[idx].copy()
    df['Prediction'] = y_pred

    strat, position, n_trades = strategy_equity(df, y_pred)
    bh = buy_hold_equity(df)
    feature_importance = sorted(zip(FEATURES, importances.mean(axis=0)), key=lambda x: x[1], reverse=True)

    summary_stats.append({
        "ticker": ticker,
        "accuracy": (y_true == y_pred).mean(),
        "baseline_accuracy": y_true.mean(),          # accuracy of always predicting "up"
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "top_features": feature_importance[:5],
        "strategy_return": strat.iloc[-1] - 1,
        "buy_hold_return": bh.iloc[-1] - 1,
        "strategy_sharpe": sharpe(strat),
        "buy_hold_sharpe": sharpe(bh),
        "strategy_max_dd": max_drawdown(strat),
        "buy_hold_max_dd": max_drawdown(bh),
        "time_in_market": position.mean(),
        "n_trades": n_trades,
        "strategy_curve": strat,
        "buy_hold_curve": bh,
    })

summary_table = pd.DataFrame([{k: v for k, v in s.items()
                               if not isinstance(v, (pd.Series, np.ndarray, list))}
                              for s in summary_stats]).set_index('ticker')

# %%
# Print and plot results
pd.set_option('display.width', 200)
print(summary_table.round(3).to_string())
print("\nMean over tickers:")
print(summary_table.mean(numeric_only=True).round(3).to_string())

if SHOW_PLOTS:
    for stat in summary_stats:
        plt.figure(figsize=(4, 3))
        sn.heatmap(stat['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title(f"{stat['ticker']} Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        stat['strategy_curve'].plot(label='Strategy')
        stat['buy_hold_curve'].plot(label='Buy & Hold')
        plt.title(f"{stat['ticker']} Strategy vs Buy & Hold")
        plt.ylabel('Growth of 1')
        plt.legend()
        plt.tight_layout()
        plt.show()

# %%

# =======================
# Dash App for Dashboard
# =======================

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px

app = Dash(__name__)
available_tickers = tickers + ['ALL']
stats_by_ticker = {s['ticker']: s for s in summary_stats}

app.layout = html.Div([
    html.H1("Stock Strategy Dashboard"),

    html.P(
        "A Random Forest trained on scale-free technical indicators (RSI, MACD, distance from moving "
        "averages, Bollinger %B, momentum, volatility) predicts whether a stock will be higher in 21 "
        "trading days. When it predicts an increase, the strategy buys and holds for 21 days, paying "
        f"{COST_BPS} bps per side. Validation is walk-forward with a {PURGE}-day purge so that no "
        "training label uses prices from the test period. The results are compared to buy-and-hold, "
        "and the model's accuracy to the baseline of always predicting \"up\".",
        style={'fontSize': '16px', 'maxWidth': '1000px', 'lineHeight': '1.6'}
    ),

    dcc.Dropdown(
        id='ticker-select',
        options=[{'label': t, 'value': t} for t in available_tickers],
        value='ALL'
    ),

    html.Div([
        dcc.Graph(id='strategy-plot', style={'width': '65%'}),
        dash_table.DataTable(
            id='summary-table',
            columns=[{'name': 'Metric', 'id': 'Metric'}, {'name': 'Value', 'id': 'Value'}],
            style_table={'width': '340px'},
            style_cell={'textAlign': 'left', 'width': '50%'},
            style_header={'fontWeight': 'bold'}
        )
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'flex-start',
              'alignItems': 'center', 'gap': '90px', 'paddingLeft': '40px'}),

    html.Div([
        dcc.Graph(id='confusion-matrix', style={'display': 'inline-block', 'width': '49%'}),
        dcc.Graph(id='confusion-matrix-cumulative', style={'display': 'inline-block', 'width': '49%'})
    ])
])

def metric_rows(s):
    return [
        {'Metric': 'Model accuracy', 'Value': f"{s['accuracy']:.1%}"},
        {'Metric': 'Always-up accuracy', 'Value': f"{s['baseline_accuracy']:.1%}"},
        {'Metric': 'Strategy return', 'Value': f"{s['strategy_return']:.1%}"},
        {'Metric': 'Buy & hold return', 'Value': f"{s['buy_hold_return']:.1%}"},
        {'Metric': 'Strategy Sharpe', 'Value': f"{s['strategy_sharpe']:.2f}"},
        {'Metric': 'Buy & hold Sharpe', 'Value': f"{s['buy_hold_sharpe']:.2f}"},
        {'Metric': 'Strategy max drawdown', 'Value': f"{s['strategy_max_dd']:.1%}"},
        {'Metric': 'Buy & hold max drawdown', 'Value': f"{s['buy_hold_max_dd']:.1%}"},
        {'Metric': 'Time in market', 'Value': f"{s['time_in_market']:.0%}"},
    ]

@app.callback(
    Output('strategy-plot', 'figure'),
    Output('confusion-matrix', 'figure'),
    Output('confusion-matrix-cumulative', 'figure'),
    Output('confusion-matrix-cumulative', 'style'),     # hidden for single stocks
    Output('summary-table', 'data'),
    Input('ticker-select', 'value')
)
def update_dashboard(selected_ticker):
    if selected_ticker == 'ALL':
        # Equal-weight average of the per-stock curves over their common dates
        strat_curves = pd.concat([s['strategy_curve'] for s in summary_stats], axis=1).dropna()
        bh_curves = pd.concat([s['buy_hold_curve'] for s in summary_stats], axis=1).dropna()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=strat_curves.index, y=strat_curves.mean(axis=1), name='Strategy'))
        fig1.add_trace(go.Scatter(x=bh_curves.index, y=bh_curves.mean(axis=1), name='Buy & Hold'))
        fig1.update_layout(title='Average Strategy vs Buy & Hold', yaxis_title='Growth of 1')

        total_cm = sum(s['confusion_matrix'] for s in summary_stats)
        fig2 = px.imshow(total_cm / len(summary_stats), text_auto='.0f', color_continuous_scale='Blues',
                         labels=dict(x="Predicted", y="Actual"), x=['Down', 'Up'], y=['Down', 'Up'])
        fig2.update_layout(title='Average Confusion Matrix')
        fig3 = px.imshow(total_cm, text_auto=True, color_continuous_scale='Blues',
                         labels=dict(x="Predicted", y="Actual"), x=['Down', 'Up'], y=['Down', 'Up'])
        fig3.update_layout(title='Cumulative Confusion Matrix')

        mean = summary_table.mean(numeric_only=True)
        rows = metric_rows(mean)

        # Average feature importance across tickers
        imp = pd.DataFrame([dict(s['top_features']) for s in summary_stats]).mean().sort_values(ascending=False)
        for i, (name, val) in enumerate(imp.head(3).items(), 1):
            rows.append({'Metric': f'Feature rank {i} (weight)', 'Value': f"{name} ({val:.3f})"})

        return fig1, fig2, fig3, {'display': 'inline-block', 'width': '49%'}, rows

    s = stats_by_ticker[selected_ticker]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=s['strategy_curve'].index, y=s['strategy_curve'].values, name='Strategy'))
    fig1.add_trace(go.Scatter(x=s['buy_hold_curve'].index, y=s['buy_hold_curve'].values, name='Buy & Hold'))
    fig1.update_layout(title=f'{selected_ticker} Strategy vs Buy & Hold', yaxis_title='Growth of 1')

    fig2 = px.imshow(s['confusion_matrix'], text_auto=True, color_continuous_scale='Blues',
                     labels=dict(x="Predicted", y="Actual"), x=['Down', 'Up'], y=['Down', 'Up'])
    fig2.update_layout(title=f'{selected_ticker} Confusion Matrix')

    rows = metric_rows(s)
    for i, (name, val) in enumerate(s['top_features'][:3], 1):
        rows.append({'Metric': f'Feature rank {i} (weight)', 'Value': f"{name} ({val:.3f})"})

    return fig1, fig2, go.Figure(), {'display': 'none'}, rows


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
