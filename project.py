#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:40:53 2025

@author: jacob
"""

# %%
# Import necessary libraries
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# %%
# Define list of tickers and prepare data dictionary
tickers = ['AAPL', 'NVDA', 'WMT', 'KO', 'XOM', 'CVX', 'JPM', 'BRK-B', 'JNJ', 'LLY']
data = {}

# %%
# Download the past 5 years of data for each ticker
for ticker in tickers:    
    df = yf.download(ticker, period='5y')
    df.dropna(inplace=True)
    data[ticker] = df

# %%
import os
os.makedirs("stock_data_csv", exist_ok=True)

# Save only OHLCV columns
ohlcv_keywords = ['Open', 'High', 'Low', 'Close', 'Volume']

for ticker, df in data.items():
    ohlcv_cols = [col for col in df.columns if any(key in col for key in ohlcv_keywords)]
    df_filtered = df[ohlcv_cols]
    df_filtered.to_csv(f"stock_data_csv/{ticker}.csv", index=True)

# %%

# Load data from saved CSV files
data = {}
for ticker in tickers:
    df = pd.read_csv(f"stock_data_csv/{ticker}.csv", header=[0, 1], index_col=0, parse_dates=True)
    df.columns = ['_'.join(col).strip() if '_' not in col[0] else col[0] for col in df.columns.values]
    data[ticker] = df

# %% 
# Function to compute technical indicators
def add_technical_indicators(df, price_col):
    # RSI (14)
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACDsignal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDhist'] = df['MACD'] - df['MACDsignal']

    # Volatility (21-day rolling standard deviation)
    df['Volatility'] = df[price_col].rolling(window=21).std()

    # Simple Moving Averages
    df['SMA20'] = df[price_col].rolling(window=20).mean()
    df['SMA50'] = df[price_col].rolling(window=50).mean()

    # Exponential Moving Averages
    df['EMA20'] = df[price_col].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df[price_col].ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20-day SMA ± 2 * std)
    sma20 = df[price_col].rolling(window=20).mean()
    std20 = df[price_col].rolling(window=20).std()
    df['BBupper'] = sma20 + (2 * std20)
    df['BBlower'] = sma20 - (2 * std20)

    # Momentum (10-day)
    df['Momentum10'] = df[price_col] - df[price_col].shift(10)

    df['Sharpe21'] = df[price_col].pct_change().rolling(21).mean() / df[price_col].pct_change().rolling(21).std()

    return df.dropna()

# %%
# Apply indicators and generate target variable
for ticker, df in data.items():
    price_col = f'Close_{ticker}'
    df = add_technical_indicators(df, price_col=price_col)
    df['Target'] = (df[price_col].shift(-21) > df[price_col]).astype(int)
    data[ticker] = df.dropna()

# %%
# Import machine learning modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
# Function for walk-forward validation
def walk_forward_validation(X, y, window_size=252, test_size=21):
    results = []
    all_y_true = []
    all_y_pred = []
    all_importances = []

    for start in range(0, len(X) - window_size - test_size + 1, test_size):
        end_train = start + window_size
        end_test = end_train + test_size

        X_train, y_train = X.iloc[start:end_train], y.iloc[start:end_train]
        X_test, y_test = X.iloc[end_train:end_test], y.iloc[end_train:end_test]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_importances.append(model.feature_importances_)

    return results, all_y_true, all_y_pred, all_importances

# %%
# Run model on each stock and store predictions and performance
results = {}

for ticker in tickers:
    df_ticker = data[ticker]
    feature_cols = [col for col in df_ticker.columns if col not in ['Target']]
    X = df_ticker[feature_cols]
    y = df_ticker['Target']
    df_ticker.sort_index(inplace=True)
    scores, y_true, y_pred, importances = walk_forward_validation(X, y, window_size=252, test_size=21)
    results[ticker] = (scores, y_true, y_pred, importances)

# %%

# Utility functions for strategy and buy-and-hold returns
def compute_strategy_returns(df, price_col, prediction_col='Prediction', horizon=21):
    wealth = 1.0
    returns = pd.Series(index=df.index, dtype=float)
    i = 0
    while i < len(df) - horizon:
        if df.iloc[i][prediction_col] == 1:
            buy_price = df.iloc[i][price_col]
            sell_price = df.iloc[i + horizon][price_col]
            if pd.notna(buy_price) and pd.notna(sell_price) and buy_price > 0:
                pct_return = sell_price / buy_price
                wealth *= pct_return
                returns.iloc[i] = wealth
            i += horizon
        else:
            i += 1
    returns.iloc[0] = 1.0 # Starts returns at 1
    returns.ffill(inplace=True) # Fills empty values with last known valid value
    return returns

def compute_buy_hold_returns(df, price_col):
    base_price = df[price_col].iloc[0]
    return df[price_col] / base_price

# %%
# Evaluate and summarize performance per stock
summary_stats = []

for ticker, (scores, y_true, y_pred, importances) in results.items():
    cm = confusion_matrix(y_true, y_pred)
    avg_importance = np.mean(importances, axis=0)
    feature_cols = [col for col in data[ticker].columns if col not in ['Target']]
    feature_importance = dict(zip(feature_cols, avg_importance))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Compute pseudo-PnL based on prediction
    df = data[ticker].copy()
    df = df.iloc[-len(y_true):]  # Align with predictions
    # df = df.reset_index()
    df['Prediction'] = y_pred
    df['Actual'] = y_true
    
    # Buy and hold benchmark
    initial_price = df[f'Close_{ticker}'].iloc[0]
    final_price = df[f'Close_{ticker}'].iloc[-1]
    buy_hold_return = final_price / initial_price - 1

    # Compute full strategy and buy-and-hold return series using utility functions
    strategy_returns = compute_strategy_returns(df, f'Close_{ticker}')
    buy_hold_curve = compute_buy_hold_returns(df, f'Close_{ticker}').loc[strategy_returns.index]
    
    cumulative_return = strategy_returns[-1] - 1

    summary_stats.append({
        "ticker": ticker,
        "mean_accuracy": np.mean(scores),
        "confusion_matrix": cm,
        "top_features": sorted_importance[:5],
        "cumulative_return": cumulative_return,
        "buy_hold_return": buy_hold_return,
        "strategy_returns": strategy_returns,
        "buy_hold_curve": buy_hold_curve
    })

# %%
# Print and plot results for each stock after processing
for stat in summary_stats:
    print(f"{stat['ticker']}: Mean Accuracy = {stat['mean_accuracy']:.3f}")
    print("Confusion Matrix:")
    print(stat['confusion_matrix'])
    print("Top Features:")
    for name, val in stat['top_features']:
        print(f"  {name}: {val:.4f}")
    print(f"Cumulative PnL from strategy (non-overlapping): {stat['cumulative_return']:.2%}")
    print(f"Buy and Hold Return: {stat['buy_hold_return']:.2%}")
    print()

    # Plotting
    plt.figure(figsize=(4, 3))
    sn.heatmap(stat['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f"{stat['ticker']} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    stat['strategy_returns'].plot(label='Strategy', marker='o')
    stat['buy_hold_curve'].plot(label='Buy & Hold')
    plt.title(f"{stat['ticker']} Strategy vs Buy & Hold")
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%


# =======================
# Dash App for Dashboard
# =======================

# Dash app for interactive dashboard visualization
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px

app = Dash(__name__)

available_tickers = tickers + ['ALL']

app.layout = html.Div([
    html.H1("Stock Strategy Dashboard"),
    
    html.P(
        "The strategy uses a machine learning model (Random Forest) trained on technical indicators "
        "like RSI, MACD, and moving averages to predict whether a stock's price will rise over the next 21 trading days. "
        "If the model predicts an increase, the strategy simulates buying the stock and selling it after 21 days. "
        "This is compared to a simple buy-and-hold approach as a baseline.",
        style={'fontSize': '16px', 'maxWidth': '1000px', 'lineHeight': '1.6'}
    ),
    
    dcc.Dropdown(
        id='ticker-select', 
        options=[{'label': t, 'value': t} for t in available_tickers], 
        value='ALL'
    ),

    html.Div([
        dcc.Graph(
            id='strategy-plot',
            style={'width': '65%'}
        ),
        dash_table.DataTable(
            id='summary-table',
            columns=[
                {'name': 'Metric', 'id': 'Metric'},
                {'name': 'Value', 'id': 'Value'}
            ],
            style_table={'width': '300px'},
            style_cell={'textAlign': 'left', 'width': '50%'},
            style_header={'fontWeight': 'bold'}
        )
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'justifyContent': 'flex-start',  
        'alignItems': 'center',          
        'gap': '90px',                   
        'paddingLeft': '40px'            
    }),

    html.Div([
        dcc.Graph(id='confusion-matrix', style={'display': 'inline-block', 'width': '49%'}),
        dcc.Graph(id='confusion-matrix-cumulative', style={'display': 'inline-block', 'width': '49%'})
    ])
])

@app.callback(
    Output('strategy-plot', 'figure'),
    Output('confusion-matrix', 'figure'),
    Output('confusion-matrix-cumulative', 'figure'),
    Output('confusion-matrix-cumulative', 'style'),     # Used to make the figure invisible for singe stocks
    Output('summary-table', 'data'),
    Input('ticker-select', 'value')
)
def update_dashboard(selected_ticker):
    if selected_ticker == 'ALL':
        avg_strategy = []
        avg_buyhold = []
        avg_cm = np.zeros((2, 2))
        count = 0
        
        # Goes through each ticker and calculates the cumulative/average of the stats
        for ticker in tickers:
            _, y_true, y_pred, _ = results[ticker]
            df = data[ticker].copy().iloc[-len(y_true):]
            df['Prediction'] = y_pred
            strategy_series = compute_strategy_returns(df, f'Close_{ticker}')
            avg_strategy.append(strategy_series)
            
            bh_series = compute_buy_hold_returns(df, f'Close_{ticker}').loc[strategy_series.index]
            avg_buyhold.append(bh_series)
            
            avg_cm += confusion_matrix(y_true, y_pred)
            count += 1
            
        # Align all strategy and buy-hold series to the same length
        if avg_strategy and avg_buyhold:
            min_len = min(len(s) for s in avg_strategy)
            common_index = avg_strategy[0].iloc[:min_len].index
            strategy_matrix = np.stack([s.values[:min_len] for s in avg_strategy])
            buyhold_matrix = np.stack([s.values[:min_len] for s in avg_buyhold])
            
        # Add data to figures
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=common_index, y=np.mean(strategy_matrix, axis=0), name='Strategy'))
        fig1.add_trace(go.Scatter(x=common_index, y=np.mean(buyhold_matrix, axis=0), name='Buy & Hold'))
        fig1.update_layout(title='Average Strategy vs Buy & Hold', yaxis_title='Cumulative Return')

        fig2 = px.imshow(avg_cm / count if count > 0 else avg_cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
        fig2.update_layout(title='Average Confusion Matrix')
        
        fig3 = px.imshow(avg_cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
        fig3.update_layout(title='Cumulative Confusion Matrix')
        
        # Aggregate feature importance by prefix
        from collections import defaultdict

        prefix_importance = defaultdict(list)

        for stat in summary_stats:
            for name, val in stat['top_features']:
                # Remove ticker suffix if present
                parts = name.split('_')
                if len(parts) >= 2 and parts[-1] in tickers:
                    stripped = '_'.join(parts[:-1])
                else:
                    stripped = name
                prefix_importance[stripped].append(val)

        # Average by prefix
        avg_prefix_importance = {k: np.mean(v) for k, v in prefix_importance.items()}
        top_features_all = sorted(avg_prefix_importance.items(), key=lambda x: x[1], reverse=True)[:3] # gets the top 3 most important features through all tickers
        
        summary_data = [
        {'Metric': 'Mean Accuracy', 'Value': f"{np.mean([np.mean(results[t][0]) for t in tickers]):.2%}"},
        {'Metric': 'Mean Strategy Return', 'Value': f"{np.mean([s['cumulative_return'] for s in summary_stats]):.2%}"},
        {'Metric': 'Mean Buy & Hold Return', 'Value': f"{np.mean([s['buy_hold_return'] for s in summary_stats]):.2%}"}
        ]
        
        # Adds the features to the table
        for i, (name, val) in enumerate(top_features_all, 1):
            summary_data.append({'Metric': f'Feature Rank {i} (Weight)', 'Value': f"{name} ({val:.3f})"})
        
        return fig1, fig2, fig3, {'display': 'inline-block', 'width': '49%'}, summary_data
    
    else:
        _, y_true, y_pred, _ = results[selected_ticker]
        df = data[selected_ticker].copy().iloc[-len(y_true):]
        df['Prediction'] = y_pred
        # Compute return curves using shared functions
        strategy_series = compute_strategy_returns(df, f'Close_{selected_ticker}')
        bh_series = compute_buy_hold_returns(df, f'Close_{selected_ticker}').loc[strategy_series.index]

        # Add data to figures
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=strategy_series.index, y=strategy_series.values, name='Strategy'))
        fig1.add_trace(go.Scatter(x=bh_series.index, y=bh_series.values, name='Buy & Hold'))
        fig1.update_layout(title=f'{selected_ticker} Strategy vs Buy & Hold', yaxis_title='Cumulative Return')

        cm = confusion_matrix(y_true, y_pred)
        fig2 = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
        fig2.update_layout(title=f'{selected_ticker} Confusion Matrix')
        
        summary = next(s for s in summary_stats if s['ticker'] == selected_ticker)
        top_features = summary['top_features'][:3]
        summary_data = [
            {'Metric': 'Accuracy', 'Value': f"{summary['mean_accuracy']:.2%}"},
            {'Metric': 'Strategy Return', 'Value': f"{summary['cumulative_return']:.2%}"},
            {'Metric': 'Buy & Hold Return', 'Value': f"{summary['buy_hold_return']:.2%}"}
        ]
        
        for i, (name, val) in enumerate(top_features, 1):
            summary_data.append({'Metric': f'Feature Rank {i} (Weight)', 'Value': f"{name} ({val:.3f})"})
        
        # Returns an invisible cumulative confusion matrix
        return fig1, fig2, go.Figure(), {'display': 'none'}, summary_data


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
    
    
    
    
    
    
    
    