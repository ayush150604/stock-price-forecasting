"""
Improved ARIMA with auto parameter tuning and rolling predictions
"""
import sys
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.data_loader import StockDataLoader
from utils.preprocessing import StockDataPreprocessor

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'MAPE': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        'R2': float(r2_score(y_true, y_pred)),
        'MSE': float(mean_squared_error(y_true, y_pred))
    }
    
    return metrics

def find_best_arima_params(train_series, max_p=5, max_d=2, max_q=5):
    """Find best ARIMA parameters using grid search"""
    print("Finding optimal ARIMA parameters...")
    
    best_aic = np.inf
    best_order = None
    
    # Test different combinations
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        print(f"  New best: {best_order} (AIC: {best_aic:.2f})")
                except:
                    continue
    
    print(f"✓ Best parameters: {best_order} (AIC: {best_aic:.2f})")
    return best_order

def rolling_forecast_arima(train_series, test_series, order):
    """
    Use rolling forecast - retrain model at each step
    This gives much better results than one-shot prediction
    """
    print("Using rolling forecast method...")
    
    predictions = []
    history = list(train_series)
    
    # Predict one step at a time
    for i in range(len(test_series)):
        try:
            # Fit model on history
            model = ARIMA(history, order=order)
            fitted = model.fit()
            
            # Predict next step
            forecast = fitted.forecast(steps=1)
            predictions.append(forecast[0])
            
            # Add actual value to history for next prediction
            history.append(test_series.iloc[i])
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(test_series)} predictions")
        except:
            # If model fails, use last prediction
            predictions.append(predictions[-1] if predictions else history[-1])
    
    return np.array(predictions)

def run_improved_arima(ticker='AAPL', auto_tune=True, use_rolling=True):
    """Run improved ARIMA on a single stock"""
    
    print("\n" + "="*70)
    print(f"Running IMPROVED ARIMA for {ticker}")
    print("="*70 + "\n")
    
    # Create directories
    results_dir = os.path.join(current_dir, 'results')
    data_raw_dir = os.path.join(current_dir, 'data', 'raw')
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}\n")
    
    # Step 1: Load Data
    print("Step 1: Loading data...")
    loader = StockDataLoader()
    
    data_file = os.path.join(data_raw_dir, f"{ticker}.csv")
    if os.path.exists(data_file):
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        data = loader.download_stock_data(ticker, period='5y')
    
    if data is None or len(data) == 0:
        print(f"❌ Failed to load data for {ticker}")
        return False
    
    print(f"✓ Data loaded: {len(data)} records")
    
    # Step 2: Preprocess
    print("\nStep 2: Preprocessing data...")
    preprocessor = StockDataPreprocessor()
    
    # Force re-process by deleting old processed files
    train_file = os.path.join('data', 'processed', f'{ticker}_train.csv')
    test_file = os.path.join('data', 'processed', f'{ticker}_test.csv')
    
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)
    
    processed = preprocessor.preprocess_stock(data, ticker, add_indicators=False)
    
    train_df = processed['train']
    test_df = processed['test']
    
    # Debug: Check what columns we have
    print(f"Train columns: {train_df.columns.tolist()}")
    print(f"Test columns: {test_df.columns.tolist()}")
    
    # Make sure Close column is numeric
    if 'Close' not in train_df.columns:
        print("❌ Error: 'Close' column not found!")
        return False
    
    # Filter only numeric rows
    train_series = pd.to_numeric(train_df['Close'], errors='coerce').dropna()
    test_series = pd.to_numeric(test_df['Close'], errors='coerce').dropna()
    
    print(f"✓ Train set: {len(train_series)} records")
    print(f"✓ Test set: {len(test_series)} records")
    
    # Step 3: Find best parameters
    print("\nStep 3: Finding best ARIMA parameters...")
    start_time = time.time()
    
    if auto_tune:
        order = find_best_arima_params(train_series, max_p=3, max_d=2, max_q=3)
    else:
        order = (1, 1, 1)  # Better default than (5,1,0)
        print(f"Using default order: {order}")
    
    # Step 4: Make Predictions
    print("\nStep 4: Making predictions...")
    
    if use_rolling:
        predictions = rolling_forecast_arima(train_series, test_series, order)
    else:
        # Standard one-shot prediction
        model = ARIMA(train_series, order=order)
        fitted = model.fit()
        predictions = fitted.forecast(steps=len(test_series))
    
    training_time = time.time() - start_time
    print(f"✓ Completed in {training_time:.2f} seconds")
    
    y_true = test_series.values
    y_pred = predictions
    
    # Step 5: Calculate Metrics
    print("\nStep 5: Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\nPerformance Metrics:")
    print(f"  RMSE: ${metrics['RMSE']:.2f}")
    print(f"  MAE: ${metrics['MAE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # Step 6: Save Results
    print("\nStep 6: Saving results...")
    
    results_file = os.path.join(results_dir, 'model_results.json')
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    result_key = f"{ticker}_ARIMA"
    all_results[result_key] = {
        'ticker': ticker,
        'model': 'ARIMA',
        'metrics': metrics,
        'predictions': {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        },
        'training_time': training_time
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"✓ Results saved to: {results_file}")
    
    # Step 7: Create Plot
    print("\nStep 7: Creating prediction chart...")
    
    plt.figure(figsize=(14, 6))
    plt.plot(test_df.index[:len(y_true)], y_true, label='Actual', color='blue', linewidth=2)
    plt.plot(test_df.index[:len(y_pred)], y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title(f'{ticker} - Improved ARIMA Predictions (Rolling Forecast)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(results_dir, f"{ticker}_ARIMA_predictions.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Chart saved to: {plot_file}")
    
    print("\n" + "="*70)
    print(f"✅ SUCCESS! {ticker} completed with improved ARIMA!")
    print("="*70)
    
    return True

def main():
    """Run improved ARIMA"""
    
    print("\n" + "="*70)
    print("IMPROVED ARIMA WITH ROLLING FORECAST")
    print("="*70)
    print("\nThis version uses:")
    print("  ✓ Auto parameter tuning")
    print("  ✓ Rolling forecast (re-train at each step)")
    print("  ✓ Better default parameters")
    print()
    
    success = run_improved_arima(
        ticker='AAPL',
        auto_tune=True,      # Find best parameters
        use_rolling=True     # Use rolling forecast
    )
    
    if success:
        print("\n✅ All done! The predictions should be MUCH better now!")
        print("   Run: streamlit run app.py")

if __name__ == "__main__":
    main()