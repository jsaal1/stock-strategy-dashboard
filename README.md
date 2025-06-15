# Stock Strategy Dashboard

An interactive dashboard comparing a machine learning-based stock trading strategy to a simple Buy & Hold strategy.

## Overview

This project:
- Uses technical indicators (RSI, MACD, Momentum, etc.)
- Trains a Random Forest classifier to predict stock price movement 21 days ahead
- Simulates a strategy that buys when an increase is predicted and sells after 21 days
- Benchmarks it against Buy & Hold

The dashboard visualizes:
- Cumulative returns of both strategies
- Confusion matrix (prediction accuracy)
- Summary statistics for each ticker

## Strategy Logic

- Buy stock if the model predicts it will increase in 21 days
- Hold for 21 trading days, then sell
- Repeat when a new positive signal appears

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-strategy-dashboard.git
   cd stock-strategy-dashboard
   
2. Install dependencies:
  pip install pandas numpy yfinance dash plotly scikit-learn seaborn matplotlib

3. Run the dashboard:
   (Optional) Comment out the yfinance download and saving of ticker data.
   python project.py
    
4. Open your browser and go to:
  localhost:8050

Features Used for Prediction
	•	RSI (Relative Strength Index)
	•	MACD and signal line
	•	Momentum
	•	Volatility
	•	Moving Averages (SMA/EMA)
	•	Bollinger Bands
	•	Sharpe Ratio

Model Details
	•	Classifier: Random Forest
	•	Validation: Walk-forward split
	•	Target: Whether stock goes up over 21 days
