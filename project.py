#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:40:53 2025

@author: jacob
"""

import pandas as pd
import yfinance as yf
import numpy as np
import os

# %%

tickers = ['AAPL', 'NVDA', 'WMT', 'KO', 'XOM', 'CVX', 'JPM', 'BRK-B', 'JNJ', 'LLY']
data = {}

for ticker in tickers:    
    df = yf.download(ticker, period='1y')
    df.dropna(inplace=True)
    data[ticker] = df

# %%

combined_df = pd.concat([df.assign(Ticker=ticker) for ticker, df in data.items()])
combined_df.to_csv('all_stocks.csv')