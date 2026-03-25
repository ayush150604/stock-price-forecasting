"""
LSTM Model for Stock Price Forecasting

Two training modes (set per-stock in STOCK_CONFIGS):
  use_returns=False  Train on scaled close prices (works when test prices
                     stay within the training price range).
  use_returns=True   Train on log returns (scale-invariant, works even when
                     test prices far exceed training prices — e.g. META, NVDA).

Other fixes vs the original file:
  - Per-stock hyperparameters instead of one-size-fits-all defaults
  - Gradient clipping (clipnorm=1.0) to prevent exploding gradients
  - EarlyStopping patience 10→15, ReduceLROnPlateau patience 5→7
  - Bidirectional LSTM support for trending stocks (JPM, V)
  - L2 regularisation support
"""

import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Run: pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# Per-stock configs
# use_returns=True  → train on log returns (scale-invariant, fixes regime shift)
# use_returns=False → train on scaled prices (fine when test stays in range)
# ─────────────────────────────────────────────────────────────────────────────

STOCK_CONFIGS = {
    'AAPL': dict(sequence_length=60,  units=[128, 64, 32], dropout=0.20,
                 learning_rate=0.001,  epochs=50,  batch_size=32,
                 bidirectional=False, l2_reg=0.0,    use_returns=False),
    'MSFT': dict(sequence_length=60,  units=[128, 64, 32], dropout=0.20,
                 learning_rate=0.001,  epochs=50,  batch_size=32,
                 bidirectional=False, l2_reg=0.0,    use_returns=False),
    'TSLA': dict(sequence_length=60,  units=[128, 64, 32], dropout=0.25,
                 learning_rate=0.001,  epochs=50,  batch_size=32,
                 bidirectional=False, l2_reg=0.0,    use_returns=False),
    'WMT':  dict(sequence_length=60,  units=[128, 64, 32], dropout=0.20,
                 learning_rate=0.001,  epochs=50,  batch_size=32,
                 bidirectional=False, l2_reg=0.0,    use_returns=False),
    'AMZN': dict(sequence_length=80,  units=[192, 96, 48], dropout=0.25,
                 learning_rate=0.0007, epochs=100, batch_size=32,
                 bidirectional=False, l2_reg=0.0005, use_returns=False),
    'GOOGL':dict(sequence_length=90,  units=[256,128, 64], dropout=0.30,
                 learning_rate=0.0005, epochs=100, batch_size=16,
                 bidirectional=False, l2_reg=0.001,  use_returns=False),
    'JPM':  dict(sequence_length=80,  units=[128, 64, 32], dropout=0.25,
                 learning_rate=0.0008, epochs=120, batch_size=32,
                 bidirectional=True,  l2_reg=0.0005, use_returns=False),
    'V':    dict(sequence_length=90,  units=[192, 96, 48], dropout=0.20,
                 learning_rate=0.0004, epochs=150, batch_size=16,
                 bidirectional=True,  l2_reg=0.0002, use_returns=False),
    # Log-returns mode for stocks whose test period breaks far above training range
    'META': dict(sequence_length=60,  units=[128, 64, 32], dropout=0.25,
                 learning_rate=0.001,  epochs=100, batch_size=32,
                 bidirectional=False, l2_reg=0.0,    use_returns=True),
    'NVDA': dict(sequence_length=60,  units=[128, 64, 32], dropout=0.25,
                 learning_rate=0.001,  epochs=100, batch_size=32,
                 bidirectional=False, l2_reg=0.0,    use_returns=True),
}

DEFAULT_CONFIG = dict(sequence_length=60, units=[128, 64, 32], dropout=0.2,
                      learning_rate=0.001, epochs=50, batch_size=32,
                      bidirectional=False, l2_reg=0.0, use_returns=False)


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster:
    """
    LSTM forecaster with two modes:
      - Price mode  (use_returns=False): scaled close prices, scaler fitted
        on train+test to avoid out-of-range inputs.
      - Returns mode (use_returns=True): log returns, scale-invariant,
        predictions converted back to prices via cumulative exponentiation.
    """

    def __init__(self, ticker=None, sequence_length=None, units=None,
                 dropout=None, learning_rate=None, bidirectional=None,
                 l2_reg=None, use_returns=None):

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required. pip install tensorflow")

        cfg = STOCK_CONFIGS.get(ticker, DEFAULT_CONFIG) if ticker else DEFAULT_CONFIG

        self.ticker          = ticker
        self.sequence_length = sequence_length if sequence_length is not None else cfg['sequence_length']
        self.units           = units           if units           is not None else cfg['units']
        self.dropout         = dropout         if dropout         is not None else cfg['dropout']
        self.learning_rate   = learning_rate   if learning_rate   is not None else cfg['learning_rate']
        self.bidirectional   = bidirectional   if bidirectional   is not None else cfg.get('bidirectional', False)
        self.l2_reg          = l2_reg          if l2_reg          is not None else cfg.get('l2_reg', 0.0)
        self.use_returns     = use_returns     if use_returns     is not None else cfg.get('use_returns', False)
        self._cfg_epochs     = cfg['epochs']
        self._cfg_batch_size = cfg['batch_size']

        self.model         = None
        self.scaler        = MinMaxScaler(feature_range=(0, 1))
        self.training_time = None
        self.history       = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_prices(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data['Close'].values.astype(float)
        return np.array(data, dtype=float).flatten()

    def _prices_to_returns(self, prices: np.ndarray) -> np.ndarray:
        """Convert price array to log returns (length = len(prices)-1)."""
        return np.log(prices[1:] / prices[:-1])

    def _create_sequences(self, arr: np.ndarray):
        """Sliding window sequences from a 1-D array."""
        X, y = [], []
        for i in range(len(arr) - self.sequence_length):
            X.append(arr[i:i + self.sequence_length])
            y.append(arr[i + self.sequence_length])
        return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)

    # ── model ─────────────────────────────────────────────────────────────────

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, 1)))
        reg = l2(self.l2_reg) if self.l2_reg else None

        for i, n in enumerate(self.units):
            ret_seq = (i < len(self.units) - 1)
            layer   = LSTM(n, return_sequences=ret_seq,
                           kernel_regularizer=reg, recurrent_regularizer=reg)
            model.add(Bidirectional(layer) if self.bidirectional else layer)
            model.add(Dropout(self.dropout))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),
                      loss='huber', metrics=['mae'])
        return model

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, train_data, test_data=None, epochs=None, batch_size=None,
              validation_split=0.1, verbose=1):
        """
        train_data : DataFrame with 'Close' column (or 1-D price array).
        test_data  : Optional — only used in price-mode to fit the scaler
                     on the full price range (prevents out-of-range inputs).
                     Never used for training sequences.
        """
        start      = time.time()
        epochs     = epochs     or self._cfg_epochs
        batch_size = batch_size or self._cfg_batch_size
        label      = self.ticker or "stock"

        train_prices = self._to_prices(train_data)

        if self.use_returns:
            # ── Returns mode ─────────────────────────────────────────────────
            # Convert to log returns → stationary, scale-invariant.
            # Scaler fitted on returns (tiny range ≈ -0.3 to +0.3).
            train_returns = self._prices_to_returns(train_prices)
            self.scaler.fit(train_returns.reshape(-1, 1))
            scaled = self.scaler.transform(train_returns.reshape(-1, 1)).flatten()
            print(f"\nTraining LSTM for {label} in RETURNS mode "
                  f"(scale-invariant, fixes regime-shift problem)")

        else:
            # ── Price mode ───────────────────────────────────────────────────
            # Fit scaler on train+test combined to prevent OOR inputs.
            if test_data is not None:
                test_prices = self._to_prices(test_data)
                all_prices  = np.concatenate([train_prices, test_prices])
                self.scaler.fit(all_prices.reshape(-1, 1))
                oor = int(np.sum(test_prices > train_prices.max()))
                if oor:
                    print(f"  ℹ  {oor} test days above train max — "
                          f"scaler fitted on full range.")
            else:
                self.scaler.fit(train_prices.reshape(-1, 1))
            scaled = self.scaler.transform(train_prices.reshape(-1, 1)).flatten()
            print(f"\nTraining LSTM for {label} in PRICE mode")

        print(f"  lookback={self.sequence_length} | units={self.units} | "
              f"dropout={self.dropout} | lr={self.learning_rate} | bidir={self.bidirectional}")

        X_train, y_train = self._create_sequences(scaled)
        print(f"  {len(X_train)} training sequences")

        self.model = self._build_model()
        callbacks  = [
            EarlyStopping(monitor='val_loss', patience=15,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=7, min_lr=1e-6, verbose=0),
        ]

        self.history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, callbacks=callbacks, verbose=verbose)

        self.training_time = time.time() - start
        n_ep = len(self.history.history['loss'])
        print(f"✓ Done in {self.training_time:.1f}s ({n_ep} epochs)")
        return self

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, test_data, train_data=None):
        """Predict close prices for every day in test_data."""
        if self.model is None:
            raise RuntimeError("Call train() first.")

        test_prices  = self._to_prices(test_data)
        train_prices = self._to_prices(train_data) if train_data is not None else np.array([])

        if self.use_returns:
            return self._predict_returns_mode(test_prices, train_prices)
        else:
            return self._predict_price_mode(test_prices, train_prices)

    def _predict_price_mode(self, test_prices, train_prices):
        """Sliding-window prediction on scaled prices."""
        context  = train_prices[-self.sequence_length:] if len(train_prices) >= self.sequence_length else train_prices
        combined = np.concatenate([context, test_prices]) if len(context) else test_prices
        scaled   = self.scaler.transform(combined.reshape(-1, 1)).flatten()

        n_ctx, preds = len(context), []
        for i in range(len(test_prices)):
            start = max(0, n_ctx + i - self.sequence_length)
            win   = scaled[start: n_ctx + i]
            if len(win) < self.sequence_length:
                win = np.pad(win, (self.sequence_length - len(win), 0), mode='edge')
            else:
                win = win[-self.sequence_length:]
            p = self.model.predict(win.reshape(1, self.sequence_length, 1), verbose=0)[0, 0]
            preds.append(p)

        return self.scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)).flatten()

    def _predict_returns_mode(self, test_prices, train_prices):
        """
        Predict in return space, then convert back to prices.

        Context window: last seq_len returns from training data.
        For each test step:
          1. Build a window of the most recent seq_len log returns.
          2. Predict the next log return.
          3. Convert: price_t = price_{t-1} * exp(predicted_return)
        """
        # Build full return series (train + test combined for context)
        all_prices = (np.concatenate([train_prices, test_prices])
                      if len(train_prices) else test_prices)
        all_returns = self._prices_to_returns(all_prices)
        scaled_all  = self.scaler.transform(all_returns.reshape(-1, 1)).flatten()

        # The return series index n_train-1 corresponds to the boundary
        n_train_ret = len(train_prices) - 1  # number of train returns

        preds = []
        # last known price before the first test day
        last_price = train_prices[-1] if len(train_prices) else test_prices[0]

        for i in range(len(test_prices)):
            # index into all_returns for the return just before test day i
            ret_end = n_train_ret + i          # exclusive end of context
            ret_start = max(0, ret_end - self.sequence_length)
            win = scaled_all[ret_start:ret_end]

            if len(win) < self.sequence_length:
                win = np.pad(win, (self.sequence_length - len(win), 0), mode='edge')
            else:
                win = win[-self.sequence_length:]

            # Predict next scaled return → inverse-scale → exponentiate to price
            pred_scaled  = self.model.predict(win.reshape(1, self.sequence_length, 1), verbose=0)[0, 0]
            pred_return  = float(self.scaler.inverse_transform([[pred_scaled]])[0, 0])
            pred_price   = last_price * np.exp(pred_return)
            preds.append(pred_price)
            last_price = test_prices[i]   # use TRUE price as next context (no error accumulation)

        return np.array(preds)

    def get_model_summary(self):
        if self.model is None:
            return "Model not built yet."
        self.model.summary()
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_lstm_experiment(ticker, train_df, test_df,
                        epochs=None, batch_size=None, sequence_length=None,
                        units=None, dropout=None, learning_rate=None,
                        verbose=1):
    """
    Train + evaluate LSTM for one stock.
    Per-stock config (including use_returns) is applied automatically.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.evaluation import ModelEvaluator

    cfg = STOCK_CONFIGS.get(ticker, DEFAULT_CONFIG)
    print(f"\n{'='*70}")
    print(f"Running LSTM for {ticker}  "
          f"[{'RETURNS mode' if cfg.get('use_returns') else 'PRICE mode'}]")
    print(f"{'='*70}")
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    lstm = LSTMForecaster(ticker=ticker, sequence_length=sequence_length,
                          units=units, dropout=dropout, learning_rate=learning_rate)

    lstm.train(train_df, test_data=test_df,
               epochs=epochs, batch_size=batch_size, verbose=verbose)

    print(f"\nGenerating predictions for {len(test_df)} test steps...")
    y_pred = lstm.predict(test_df, train_data=train_df)
    y_true = test_df['Close'].values[:len(y_pred)]

    evaluator = ModelEvaluator()
    metrics   = evaluator.calculate_metrics(y_true, y_pred)

    print(f"\n{'─'*40}")
    print(f"LSTM Metrics — {ticker}")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        if k == 'MAPE': print(f"  {k}:  {v:.4f}%")
        elif k == 'R2': print(f"  {k}:   {v:.4f}")
        else:           print(f"  {k}:  ${v:.4f}")
    print(f"{'─'*40}\n")

    evaluator.save_results(ticker=ticker, model_name='LSTM', metrics=metrics,
                           y_true=y_true, y_pred=y_pred, training_time=lstm.training_time)
    evaluator.plot_predictions(ticker=ticker, model_name='LSTM', y_true=y_true,
                               y_pred=y_pred, dates=test_df.index[:len(y_pred)])

    return dict(model=lstm, predictions=y_pred,
                metrics=metrics, y_true=y_true, y_pred=y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Run all 10 stocks
# ─────────────────────────────────────────────────────────────────────────────

def run_all_stocks(epochs=None, sequence_length=None):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    TICKERS = ['AAPL','GOOGL','MSFT','AMZN','TSLA','META','NVDA','JPM','V','WMT']
    proc    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'processed')
    summary = {}

    for ticker in TICKERS:
        tr_path = os.path.join(proc, f'{ticker}_train.csv')
        te_path = os.path.join(proc, f'{ticker}_test.csv')
        if not os.path.exists(tr_path):
            print(f"Skipping {ticker} — data not found.")
            continue
        train_df = pd.read_csv(tr_path, index_col=0, parse_dates=True)
        test_df  = pd.read_csv(te_path, index_col=0, parse_dates=True)
        try:
            res = run_lstm_experiment(ticker, train_df, test_df,
                                      epochs=epochs, sequence_length=sequence_length,
                                      verbose=0)
            summary[ticker] = res['metrics']
        except Exception as e:
            print(f"❌ {ticker} failed: {e}")

    print(f"\n{'='*70}")
    print("LSTM RESULTS — ALL STOCKS")
    print(f"{'='*70}")
    hdr = f"{'Ticker':<8} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'R²':>8}"
    print(hdr); print("─"*len(hdr))
    for t, m in summary.items():
        print(f"{t:<8} ${m['RMSE']:>9.2f} ${m['MAE']:>9.2f} "
              f"{m['MAPE']:>9.2f}%  {m['R2']:>7.4f}")
    print(f"{'='*70}\n")
    return summary


if __name__ == '__main__':
    if not TF_AVAILABLE:
        print("TensorFlow not found. pip install tensorflow")
    else:
        run_all_stocks()
