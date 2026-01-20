"""
Quick test script - runs ARIMA on just AAPL for faster testing
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.data_loader import StockDataLoader
from utils.preprocessing import StockDataPreprocessor
from models.arima_model import run_arima_experiment

def quick_test():
    print("\n" + "="*70)
    print("QUICK TEST - Running ARIMA on AAPL only")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize
    loader = StockDataLoader()
    preprocessor = StockDataPreprocessor()
    
    ticker = 'AAPL'
    
    # Step 1: Download data
    print(f"Step 1: Downloading {ticker} data...")
    data = loader.download_stock_data(ticker, period='5y')
    
    if data is None:
        print("Failed to download data!")
        return
    
    # Step 2: Preprocess
    print(f"\nStep 2: Preprocessing {ticker}...")
    processed = preprocessor.preprocess_stock(data, ticker, add_indicators=False)
    
    # Step 3: Run ARIMA
    print(f"\nStep 3: Running ARIMA on {ticker}...")
    results = run_arima_experiment(
        ticker=ticker,
        train_df=processed['train'],
        test_df=processed['test'],
        order=(5, 1, 0),
        auto_order=False
    )
    
    print("\n" + "="*70)
    print("✅ QUICK TEST COMPLETED!")
    print("="*70)
    print("\nNow run: streamlit run app.py")
    print("Then select AAPL from the dropdown")
    print("\nTo process all 10 stocks, run: python main_simple.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    quick_test()