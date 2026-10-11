# Stock Strategy Dashboard

A Random Forest that predicts whether a stock will be higher in 21 trading days, a simple
long/cash strategy built on its signals, and an interactive Dash dashboard comparing it with
buy-and-hold. Started as the final project of the LINC-STEM Advanced Python Workshops (2025);
revised in October 2026 after finding that the original backtest leaked future information.

**Short version:** the first version appeared to beat buy-and-hold on 9 of 10 stocks. That edge came
from look-ahead leakage between training and test windows. With the leakage removed, the model is
*less* accurate than always predicting "up", and the strategy beats buy-and-hold on only 3 of 10 stocks.

## Setup

- **Data:** daily prices for 10 large-cap US stocks (AAPL, NVDA, WMT, KO, XOM, CVX, JPM, BRK-B, JNJ, LLY),
  June 2020 to June 2025, from Yahoo Finance. Saved in `stock_data_csv/` so results are reproducible.
- **Target:** 1 if the close 21 trading days ahead is above today's close, else 0.
- **Features:** 10 scale-free indicators: RSI(14), MACD and MACD histogram (divided by price),
  21-day return volatility, distance from the 20- and 50-day SMA, EMA20/EMA50 ratio, Bollinger %B,
  10-day return, and 21-day rolling Sharpe ratio.
- **Model:** Random Forest (100 trees, `min_samples_leaf=5`), retrained every 21 days on a rolling
  252-day window (walk-forward validation), with a 21-day purge (explained below).
- **Strategy:** on a day the model predicts "up", buy at the close and hold for 21 days; otherwise stay in
  cash. 5 bps transaction cost per side.

## The leakage, and how it was fixed

Each label looks 21 days into the future. In the original walk-forward split the training window ended
on the day before the test window started, so the last 21 training labels were computed from prices
*inside* the test window. The model was partly trained on the answers it was then tested on.

The fix is a **purge**: drop the last 21 rows of every training window, so no training label depends on a
price the model is later evaluated on.

Three smaller problems were fixed at the same time:

1. **Price-level features.** The original features included Open/High/Low/Close, Volume, and SMAs, EMAs and
   Bollinger bands in dollars. These trend over time, so a tree model trained on last year's price range
   cannot extrapolate to this year's. All features are now ratios, returns or bounded oscillators.
2. **Unknown labels counted as "down".** The last 21 rows have no future price; comparing with NaN silently
   labelled them 0. They are now dropped.
3. **Misaligned predictions.** Predictions were matched to dates by taking the last *n* rows of the data,
   which shifts them by up to 20 days when the data length is not a multiple of 21. Predictions now carry
   their own dates.

## Results

Mean over the 10 stocks, before transaction costs (`python leakage_check.py`):

| Setup | Accuracy | Always "up" | Strategy return | Buy & hold | Beats B&H |
|---|---|---|---|---|---|
| Original: raw features, no purge | 62.1% | 58.9% | +200% | +119% | 9/10 |
| Raw features, 21-day purge | 56.1% | 58.9% | +101% | +119% | 4/10 |
| Scale-free features, no purge | 55.8% | 58.9% | +132% | +119% | 7/10 |
| **Scale-free features, 21-day purge (current)** | **53.8%** | **58.9%** | **+100%** | **+119%** | **3/10** |

With 5 bps costs and risk metrics (`python project.py`), the current version averages a Sharpe ratio of
0.75 against 0.79 for buy-and-hold, and a maximum drawdown of -27% against -30%. The slightly smaller
drawdown comes from spending about 18% of the time in cash, not from predictive skill.

**Conclusion:** after removing the leakage, these technical indicators show no exploitable signal for 21-day
direction in this sample. Buy-and-hold over a five-year bull market is hard to beat by stepping out of it.

## Remaining limitations

- **Hindsight stock selection.** The tickers include 2020 to 2025's biggest winners (NVDA, LLY), which flatters
  any strategy that is mostly long.
- **Overlapping labels.** Consecutive 21-day labels share 20 days of price path, so the effective number of
  independent observations is far smaller than the row count, and accuracy differences of a few percent
  are within noise.
- **One regime.** Five years of mostly rising markets; no out-of-sample period with a different regime.

## Run it

```bash
pip install pandas numpy dash plotly scikit-learn seaborn matplotlib yfinance
python leakage_check.py   # prints the comparison table above
python project.py         # prints per-stock results and starts the dashboard
```

Then open `http://127.0.0.1:8050`. Set `DOWNLOAD = True` in `project.py` to fetch fresh data (results will change).
