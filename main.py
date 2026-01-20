"""
Stock Price Forecasting - Main Pipeline
Run this script to execute the complete forecasting pipeline
"""

import os
import sys
from utils.data_loader import StockDataLoader
from utils.preprocessing import StockDataPreprocessor
from utils.evaluation import ModelEvaluator
from models.arima_model import run_arima_experiment

def create_project_structure():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'utils',
        'results',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Project structure created")

def step1_download_data():
    """Step 1: Download stock data"""
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING STOCK DATA")
    print("="*70 + "\n")
    
    loader = StockDataLoader()
    all_data = loader.download_all_stocks(period='5y')
    
    print(f"\n✓ Successfully downloaded data for {len(all_data)} companies")
    return loader

def step2_preprocess_data(loader):
    """Step 2: Preprocess all stock data"""
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING DATA")
    print("="*70 + "\n")
    
    preprocessor = StockDataPreprocessor()
    processed_stocks = {}
    
    for ticker in loader.companies.keys():
        try:
            data = loader.load_stock_data(ticker)
            if data is not None:
                processed = preprocessor.preprocess_stock(
                    data, 
                    ticker, 
                    train_size=0.8,
                    add_indicators=False  # Keep it simple for ARIMA
                )
                processed_stocks[ticker] = processed
        except Exception as e:
            print(f"Error preprocessing {ticker}: {str(e)}")
    
    print(f"\n✓ Successfully preprocessed {len(processed_stocks)} companies")
    return processed_stocks

def step3_run_arima_all_stocks(processed_stocks):
    """Step 3: Run ARIMA on all stocks"""
    print("\n" + "="*70)
    print("STEP 3: RUNNING ARIMA MODEL ON ALL STOCKS")
    print("="*70 + "\n")
    
    results_all = {}
    
    for ticker, data in processed_stocks.items():
        try:
            print(f"\nProcessing {ticker}...")
            results = run_arima_experiment(
                ticker=ticker,
                train_df=data['train'],
                test_df=data['test'],
                order=(5, 1, 0),  # Default order
                auto_order=False   # Set to True for auto parameter tuning (slower)
            )
            results_all[ticker] = results
            print(f"✓ {ticker} completed")
        except Exception as e:
            print(f"✗ Error with {ticker}: {str(e)}")
    
    print(f"\n✓ ARIMA completed for {len(results_all)} companies")
    return results_all

def step4_generate_reports(loader):
    """Step 4: Generate evaluation reports"""
    print("\n" + "="*70)
    print("STEP 4: GENERATING EVALUATION REPORTS")
    print("="*70 + "\n")
    
    evaluator = ModelEvaluator()
    evaluator.load_results()
    
    # Generate report for each company
    for ticker in loader.companies.keys():
        try:
            print(f"\n{ticker} Report:")
            print("-" * 70)
            evaluator.generate_report(ticker)
        except Exception as e:
            print(f"Error generating report for {ticker}: {str(e)}")

def run_complete_pipeline():
    """Run the complete forecasting pipeline"""
    print("\n" + "="*70)
    print("STOCK PRICE FORECASTING PIPELINE")
    print("="*70 + "\n")
    
    # Create project structure
    create_project_structure()
    
    # Step 1: Download data
    loader = step1_download_data()
    
    # Step 2: Preprocess data
    processed_stocks = step2_preprocess_data(loader)
    
    # Step 3: Run ARIMA on all stocks
    results = step3_run_arima_all_stocks(processed_stocks)
    
    # Step 4: Generate reports
    step4_generate_reports(loader)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    print("Next steps:")
    print("1. Check the 'results/' folder for predictions and metrics")
    print("2. Run 'streamlit run app.py' to view the dashboard")
    print("3. Add more models (LSTM, Prophet, etc.) in the 'models/' folder")
    print("="*70 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Forecasting Pipeline')
    parser.add_argument('--step', type=str, choices=['all', '1', '2', '3', '4'], 
                       default='all', help='Which step to run')
    
    args = parser.parse_args()
    
    if args.step == 'all':
        run_complete_pipeline()
    elif args.step == '1':
        create_project_structure()
        step1_download_data()
    elif args.step == '2':
        loader = StockDataLoader()
        step2_preprocess_data(loader)
    elif args.step == '3':
        loader = StockDataLoader()
        preprocessor = StockDataPreprocessor()
        processed_stocks = {}
        for ticker in loader.companies.keys():
            data = loader.load_stock_data(ticker)
            if data is not None:
                processed_stocks[ticker] = preprocessor.preprocess_stock(data, ticker)
        step3_run_arima_all_stocks(processed_stocks)
    elif args.step == '4':
        loader = StockDataLoader()
        step4_generate_reports(loader)