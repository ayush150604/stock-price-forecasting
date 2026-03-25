"""
retrain_weak_stocks.py
======================
Targeted retraining of LSTM models for stocks that performed poorly
in the initial run. Each stock gets hyperparameters tuned to its
specific failure mode.

Weak stocks and their diagnoses
--------------------------------
GOOGL  R²=0.28  MAPE=12.3%  → High volatility (31%), large price swings.
                               Needs longer lookback + bigger network.
JPM    R²=0.38  MAPE=7.2%   → Financial-sector regime shifts.
                               Needs more epochs + gradient clipping.
META   R²=0.60  MAPE=5.6%   → Extreme volatility (45%), >$500 price range.
                               Needs larger capacity + heavier dropout.
V      R²=-0.47 MAPE=3.4%   → Model learned mean, not trend (negative R²).
                               Needs much longer lookback + lower LR.
AMZN   R²=0.65  MAPE=3.8%   → Moderate volatility, underfit.
                               Needs more units + extra epochs.
NVDA   R²=0.80  MAPE=7.2%   → Extreme volatility (52%), GPU-boom regime.
                               Needs biggest network + longest lookback.

Usage
-----
    python retrain_weak_stocks.py               # retrain all 6 weak stocks
    python retrain_weak_stocks.py --ticker GOOGL # retrain one stock only
    python retrain_weak_stocks.py --dry-run      # show config, don't train
"""

import os, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── TensorFlow ────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} loaded")
except ImportError:
    TF_AVAILABLE = False
    print("✗ TensorFlow not found. Install with: pip install tensorflow")
    sys.exit(1)

from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')
RESULTS_FILE  = os.path.join(RESULTS_DIR, 'model_results.json')

sys.path.insert(0, BASE_DIR)
from utils.evaluation import ModelEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Per-stock hyperparameter configs  (tuned to each failure mode)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {

    # GOOGL – large price swing, high vol → long lookback, bigger net
    'GOOGL': dict(
        sequence_length = 90,        # was 60; more context for trending moves
        units           = [256, 128, 64],   # was [128,64,32]
        dropout         = 0.3,
        learning_rate   = 0.0005,    # slower = more stable
        epochs          = 100,
        batch_size      = 16,        # smaller batches → smoother gradients
        bidirectional   = False,
        l2_reg          = 0.001,
        notes           = "Long lookback + bigger net for high-vol trending stock"
    ),

    # JPM – financial regime shifts → gradient clipping, patience
    'JPM': dict(
        sequence_length = 80,
        units           = [128, 64, 32],
        dropout         = 0.25,
        learning_rate   = 0.0008,
        epochs          = 120,
        batch_size      = 32,
        bidirectional   = True,      # bidirectional helps with regime detection
        l2_reg          = 0.0005,
        notes           = "Bidirectional LSTM + more epochs for regime-shift stock"
    ),

    # META – extreme vol ($305 range), overfit risk
    'META': dict(
        sequence_length = 90,
        units           = [256, 128, 64],
        dropout         = 0.4,       # heavier dropout for extreme vol
        learning_rate   = 0.0005,
        epochs          = 100,
        batch_size      = 16,
        bidirectional   = False,
        l2_reg          = 0.002,     # stronger regularisation
        notes           = "Heavy dropout + L2 for extreme-volatility stock"
    ),

    # V – negative R²: model learned mean, ignored trend → very long lookback
    'V': dict(
        sequence_length = 120,       # longest lookback to capture multi-month trend
        units           = [128, 64, 32],
        dropout         = 0.2,
        learning_rate   = 0.0003,    # very slow LR to escape mean-regression trap
        epochs          = 150,       # many epochs with patience
        batch_size      = 16,
        bidirectional   = True,
        l2_reg          = 0.0001,
        notes           = "Very long lookback + slow LR to fix mean-regression trap"
    ),

    # AMZN – moderate vol, underfit → more capacity + epochs
    'AMZN': dict(
        sequence_length = 80,
        units           = [192, 96, 48],
        dropout         = 0.25,
        learning_rate   = 0.0007,
        epochs          = 100,
        batch_size      = 32,
        bidirectional   = False,
        l2_reg          = 0.0005,
        notes           = "Larger network to fix underfitting"
    ),

    # NVDA – highest vol (52%), GPU boom creates non-stationary regime
    'NVDA': dict(
        sequence_length = 100,       # capture longer GPU-boom cycle
        units           = [256, 128, 64],
        dropout         = 0.35,
        learning_rate   = 0.0005,
        epochs          = 120,
        batch_size      = 16,
        bidirectional   = True,      # both directions help with spike detection
        l2_reg          = 0.001,
        notes           = "Bidirectional + long lookback for highest-vol stock"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> Sequential:
    """Build LSTM model from config dict."""
    seq_len = cfg['sequence_length']
    units   = cfg['units']
    dropout = cfg['dropout']
    lr      = cfg['learning_rate']
    l2_reg  = cfg.get('l2_reg', 0.0)
    bidir   = cfg.get('bidirectional', False)

    model = Sequential()
    model.add(Input(shape=(seq_len, 1)))

    for i, n_units in enumerate(units):
        return_seq = (i < len(units) - 1)
        reg = l2(l2_reg) if l2_reg else None
        kwargs = dict(return_sequences=return_seq,
                      kernel_regularizer=reg,
                      recurrent_regularizer=reg)

        layer = LSTM(n_units, **kwargs)
        if bidir:
            model.add(Bidirectional(layer))
        else:
            model.add(layer)
        model.add(Dropout(dropout))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),  # gradient clipping
        loss='huber',
        metrics=['mae']
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Sequence helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(scaled: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len, 0])
    return np.array(X), np.array(y)


def make_predictions(model, scaler, train_prices, test_prices, seq_len):
    """Generate test predictions using true-history sliding window."""
    context      = train_prices[-seq_len:]
    combined     = np.vstack([context, test_prices])
    scaled       = scaler.transform(combined)
    n_context    = len(context)
    predictions  = []

    for i in range(len(test_prices)):
        start = max(0, n_context + i - seq_len)
        end   = n_context + i
        window = scaled[start:end + 1]
        if len(window) < seq_len:
            pad    = np.full((seq_len - len(window), 1), scaled[0, 0])
            window = np.vstack([pad, window])
        else:
            window = window[-seq_len:]
        pred = model.predict(window.reshape(1, seq_len, 1), verbose=0)[0, 0]
        predictions.append(pred)

    preds = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Main retraining function
# ─────────────────────────────────────────────────────────────────────────────

def retrain_stock(ticker: str, dry_run: bool = False) -> dict | None:
    """Retrain LSTM for one stock with improved hyperparameters."""

    if ticker not in CONFIGS:
        print(f"⚠  No improved config for {ticker}. Skipping.")
        return None

    cfg = CONFIGS[ticker]
    print(f"\n{'='*65}")
    print(f"  Retraining LSTM for {ticker}")
    print(f"{'='*65}")
    print(f"  Notes     : {cfg['notes']}")
    print(f"  Lookback  : {cfg['sequence_length']} days  (was 60)")
    print(f"  Units     : {cfg['units']}")
    print(f"  Dropout   : {cfg['dropout']}")
    print(f"  LR        : {cfg['learning_rate']}")
    print(f"  Epochs    : {cfg['epochs']} (with early stopping, patience=15)")
    print(f"  Batch     : {cfg['batch_size']}")
    print(f"  Bidir     : {cfg.get('bidirectional', False)}")
    print(f"  L2 reg    : {cfg.get('l2_reg', 0.0)}")

    if dry_run:
        print("  [dry-run] Skipping actual training.")
        return None

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(PROCESSED_DIR, f'{ticker}_train.csv')
    test_path  = os.path.join(PROCESSED_DIR, f'{ticker}_test.csv')
    train_df   = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_df    = pd.read_csv(test_path,  index_col=0, parse_dates=True)

    train_prices = train_df['Close'].values.reshape(-1, 1)
    test_prices  = test_df['Close'].values.reshape(-1, 1)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train_prices)

    seq_len      = cfg['sequence_length']
    X_train, y_train = make_sequences(scaled, seq_len)
    X_train      = X_train.reshape(X_train.shape[0], seq_len, 1)

    print(f"\n  Training on {len(X_train)} sequences...")

    # ── Build & train ─────────────────────────────────────────────────────────
    model = build_model(cfg)

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs          = cfg['epochs'],
        batch_size      = cfg['batch_size'],
        validation_split= 0.1,
        callbacks       = callbacks,
        verbose         = 1,
    )
    training_time  = time.time() - t0
    actual_epochs  = len(history.history['loss'])
    final_val_loss = history.history['val_loss'][-1]

    print(f"\n  ✓ Trained {actual_epochs} epochs in {training_time:.1f}s")
    print(f"  Final val_loss: {final_val_loss:.6f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    print(f"  Generating predictions on {len(test_df)} test days...")
    y_pred = make_predictions(model, scaler, train_prices, test_prices, seq_len)
    y_true = test_df['Close'].values[:len(y_pred)]

    # ── Metrics ───────────────────────────────────────────────────────────────
    evaluator = ModelEvaluator()
    metrics   = evaluator.calculate_metrics(y_true, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  Results for {ticker} (IMPROVED LSTM)")
    print(f"  {'─'*40}")

    # Load old metrics for comparison
    old_metrics = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
        old_metrics = existing.get(f'{ticker}_LSTM', {}).get('metrics', {})

    for k, v in metrics.items():
        old_v = old_metrics.get(k, None)
        if old_v is not None:
            delta = v - old_v
            arrow = "▲" if delta > 0 else "▼"
            if k == 'R2':
                # higher R2 is better
                arrow = "▲ better" if delta > 0 else "▼ worse"
                print(f"  {k:<6}: {v:>8.4f}  (was {old_v:.4f}, {arrow} {abs(delta):.4f})")
            elif k == 'MAPE':
                arrow = "▼ better" if delta < 0 else "▲ worse"
                print(f"  {k:<6}: {v:>8.2f}%  (was {old_v:.2f}%, {arrow} {abs(delta):.2f}%)")
            else:
                arrow = "▼ better" if delta < 0 else "▲ worse"
                print(f"  {k:<6}: ${v:>8.4f}  (was ${old_v:.4f}, {arrow} ${abs(delta):.4f})")
        else:
            if k == 'MAPE':
                print(f"  {k:<6}: {v:>8.2f}%")
            elif k == 'R2':
                print(f"  {k:<6}: {v:>8.4f}")
            else:
                print(f"  {k:<6}: ${v:>8.4f}")

    # ── Save results & plot ───────────────────────────────────────────────────
    evaluator.save_results(
        ticker        = ticker,
        model_name    = 'LSTM',
        metrics       = metrics,
        y_true        = y_true,
        y_pred        = y_pred,
        training_time = training_time,
    )
    evaluator.plot_predictions(
        ticker     = ticker,
        model_name = 'LSTM',
        y_true     = y_true,
        y_pred     = y_pred,
        dates      = test_df.index[:len(y_pred)],
    )

    print(f"  ✓ Results saved → results/model_results.json")
    print(f"  ✓ Plot saved    → results/{ticker}_LSTM_predictions.png")

    return {'ticker': ticker, 'metrics': metrics, 'improved_from': old_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    print(f"\n{'='*65}")
    print("  RETRAINING COMPLETE — SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Ticker':<8} {'Old R²':>8} {'New R²':>8} {'Old MAPE':>10} {'New MAPE':>10} {'Status':>10}")
    print(f"  {'─'*60}")

    for r in all_results:
        if r is None:
            continue
        ticker   = r['ticker']
        old_r2   = r['improved_from'].get('R2',   float('nan'))
        new_r2   = r['metrics']['R2']
        old_mape = r['improved_from'].get('MAPE', float('nan'))
        new_mape = r['metrics']['MAPE']

        r2_ok   = new_r2   > old_r2
        mape_ok = new_mape < old_mape
        status  = "✅ Better" if (r2_ok and mape_ok) else ("⚠ Mixed" if (r2_ok or mape_ok) else "❌ Worse")

        print(f"  {ticker:<8} {old_r2:>8.3f} {new_r2:>8.3f} "
              f"{old_mape:>9.2f}% {new_mape:>9.2f}%  {status}")

    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain weak LSTM stocks")
    parser.add_argument('--ticker',  type=str, default=None,
                        help='Single ticker to retrain (default: all 6 weak stocks)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without training')
    args = parser.parse_args()

    # Ordered by severity (worst first) so you can Ctrl-C early if needed
    weak_tickers = ['GOOGL', 'V', 'JPM', 'META', 'NVDA', 'AMZN']

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = weak_tickers

    print(f"\n{'='*65}")
    print("  LSTM RETRAINING — WEAK STOCKS")
    print(f"  Stocks to retrain: {', '.join(tickers)}")
    if args.dry_run:
        print("  MODE: DRY RUN (configs only, no training)")
    print(f"{'='*65}")

    # Estimated time warning
    if not args.dry_run:
        est_minutes = len(tickers) * 8   # ~8 min per stock on CPU
        print(f"\n  ⏱  Estimated time: {est_minutes}–{est_minutes*2} min on CPU")
        print(f"     (EarlyStopping will cut this if val_loss plateaus)\n")

    all_results = []
    total_start = time.time()

    for ticker in tickers:
        result = retrain_stock(ticker, dry_run=args.dry_run)
        all_results.append(result)

    if not args.dry_run:
        total_time = time.time() - total_start
        print(f"\n  Total retraining time: {total_time/60:.1f} min")
        print_summary([r for r in all_results if r is not None])


if __name__ == '__main__':
    main()