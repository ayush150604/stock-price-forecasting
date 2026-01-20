"""
Run ARIMA on a single stock - GUARANTEED to save results
"""
import sys
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import json
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import after setting path
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

def run_arima_single_stock(ticker='AAPL'):
    """Run ARIMA on a single stock and save results"""
    
    print("\n" + "="*70)
    print(f"Running ARIMA for {ticker}")
    print("="*70 + "\n")
    
    # Create directories with absolute paths
    results_dir = os.path.join(current_dir, 'results')
    data_raw_dir = os.path.join(current_dir, 'data', 'raw')
    data_processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}\n")
    
    # Step 1: Load/Download Data
    print("Step 1: Loading data...")
    loader = StockDataLoader()
    
    data_file = os.path.join(data_raw_dir, f"{ticker}.csv")
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        print(f"Downloading data for {ticker}...")
        data = loader.download_stock_data(ticker, period='5y')
    
    if data is None or len(data) == 0:
        print(f"❌ Failed to load data for {ticker}")
        return False
    
    print(f"✓ Data loaded: {len(data)} records")
    
    # Step 2: Preprocess
    print("\nStep 2: Preprocessing data...")
    preprocessor = StockDataPreprocessor()
    processed = preprocessor.preprocess_stock(data, ticker, add_indicators=False)
    
    train_df = processed['train']
    test_df = processed['test']
    
    print(f"✓ Train set: {len(train_df)} records")
    print(f"✓ Test set: {len(test_df)} records")
    
    # Step 3: Train ARIMA
    print("\nStep 3: Training ARIMA model...")
    
    # Make sure we only get numeric data
    train_series = train_df['Close'].astype(float)
    test_series = test_df['Close'].astype(float)
    
    # Remove any NaN values
    train_series = train_series.dropna()
    test_series = test_series.dropna()
    
    print(f"✓ Train series: {len(train_series)} values")
    print(f"✓ Test series: {len(test_series)} values")
    
    start_time = time.time()
    
    # Fit ARIMA model
    order = (5, 1, 0)
    print(f"ARIMA order: {order}")
    
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    
    training_time = time.time() - start_time
    print(f"✓ Model trained in {training_time:.2f} seconds")
    
    # Step 4: Make Predictions
    print("\nStep 4: Making predictions...")
    predictions = fitted_model.forecast(steps=len(test_series))
    
    y_true = test_series.values
    y_pred = predictions.values
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    # Step 5: Calculate Metrics
    print("\nStep 5: Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\nPerformance Metrics:")
    print(f"  RMSE: ${metrics['RMSE']:.2f}")
    print(f"  MAE: ${metrics['MAE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # Step 6: Save Results to JSON
    print("\nStep 6: Saving results...")
    
    results_file = os.path.join(results_dir, 'model_results.json')
    
    # Load existing results if any
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    # Add this result
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
    
    # Save to JSON
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"✓ Results saved to: {results_file}")
    
    # Step 7: Create Prediction Plot
    print("\nStep 7: Creating prediction chart...")
    
    plt.figure(figsize=(14, 6))
    plt.plot(test_df.index, y_true, label='Actual', color='blue', linewidth=2)
    plt.plot(test_df.index, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title(f'{ticker} - ARIMA Predictions', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(results_dir, f"{ticker}_ARIMA_predictions.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Chart saved to: {plot_file}")
    
    # Step 8: Verify files exist
    print("\nStep 8: Verifying saved files...")
    
    if os.path.exists(results_file):
        file_size = os.path.getsize(results_file)
        print(f"✓ {results_file} ({file_size} bytes)")
    
    if os.path.exists(plot_file):
        file_size = os.path.getsize(plot_file)
        print(f"✓ {plot_file} ({file_size} bytes)")
    
    print("\n" + "="*70)
    print(f"✅ SUCCESS! {ticker} completed!")
    print("="*70)
    
    return True

def main():
    """Run ARIMA on AAPL"""
    
    print("\n" + "="*70)
    print("STOCK PRICE FORECASTING - SINGLE STOCK TEST")
    print("="*70)
    
    success = run_arima_single_stock('AAPL')
    
    if success:
        print("\n✅ All done! Now you can:")
        print("   1. Check results folder: dir results")
        print("   2. Run dashboard: streamlit run app.py")
        print("\nTo process all 10 stocks, run this script 10 times with different tickers")
        print("or run: python main_simple.py")
    else:
        print("\n❌ Failed! Check the errors above.")

if __name__ == "__main__":
    main()