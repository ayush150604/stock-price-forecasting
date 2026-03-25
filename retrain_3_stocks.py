"""
retrain_3_stocks.py  — retrain META and NVDA with log-returns mode

Run from your project folder:
    python retrain_3_stocks.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from models.lstm_model import run_lstm_experiment

# META and NVDA now use log-returns mode (set in STOCK_CONFIGS)
# This is scale-invariant so the price regime-shift is no longer a problem
for ticker in ['META', 'NVDA']:
    train_df = pd.read_csv(f'data/processed/{ticker}_train.csv', index_col=0, parse_dates=True)
    test_df  = pd.read_csv(f'data/processed/{ticker}_test.csv',  index_col=0, parse_dates=True)
    run_lstm_experiment(ticker, train_df, test_df)

print("\nDone! Check results/model_results.json for updated metrics.")
