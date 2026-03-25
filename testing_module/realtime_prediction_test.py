"""
Real-time Stock Price Prediction Test
Predicts March 20, 2026 prices using March 18-19, 2026 data
Creates separate testing folder - does NOT disturb existing data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

# Create separate testing directory
TEST_DIR = os.path.join(parent_dir, 'testing_module')
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, 'results'), exist_ok=True)

class RealtimePredictionTest:
    """Test model on real-time data"""
    
    def __init__(self):
        self.test_dir = TEST_DIR
        self.results = {}
    
    def download_latest_data(self, ticker, days_back=1000):
        """
        Download recent stock data including latest trading days
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days of historical data to download
            
        Returns:
            DataFrame with stock data
        """
        print(f"\n{'='*70}")
        print(f"DOWNLOADING LATEST DATA FOR {ticker}")
        print(f"{'='*70}\n")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            print(f"Downloading data from {start_date.date()} to {end_date.date()}...")
            
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"❌ No data retrieved for {ticker}")
                return None
            
            # Save to CSV
            csv_file = os.path.join(self.test_dir, 'data', f'{ticker}_latest.csv')
            df.to_csv(csv_file)
            
            print(f"✓ Downloaded {len(df)} records")
            print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"✓ Saved to: {csv_file}\n")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading data: {str(e)}")
            return None
    
    def get_specific_dates(self, df, target_date_str='2026-03-20'):
        """
        Extract data for specific test dates
        
        Args:
            df: Full DataFrame
            target_date_str: Date to predict (format: 'YYYY-MM-DD')
            
        Returns:
            training_data, actual_value, test_dates
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING TEST DATA")
        print(f"{'='*70}\n")
        
        try:
            # Convert target date and make timezone-aware if needed
            target_date = pd.to_datetime(target_date_str)
            
            # Make df index timezone-naive for comparison
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Get the 2 days before target date for testing
            test_start = target_date - timedelta(days=4)  # Buffer for weekends
            
            # Filter data
            historical_data = df[df.index < test_start]
            test_window = df[(df.index >= test_start) & (df.index < target_date)]
            
            print(f"Target prediction date: {target_date.date()}")
            print(f"\nHistorical data (for training):")
            print(f"  Records: {len(historical_data)}")
            if len(historical_data) > 0:
                print(f"  Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
            
            print(f"\nRecent data (last 10 trading days before target):")
            if len(test_window) > 0:
                print(test_window[['Close']].tail(10))
            else:
                print("  No data in test window - using latest available data")
                print(df[['Close']].tail(5))
            
            # Check if we have data for target date
            target_data = df[df.index.date == target_date.date()]
            
            if not target_data.empty:
                actual_value = target_data['Close'].iloc[0]
                print(f"\n✓ Actual value for {target_date.date()}: ${actual_value:.2f}")
            else:
                actual_value = None
                print(f"\n⚠ No actual data for {target_date.date()} yet (market not closed)")
            
            return historical_data, test_window, actual_value, target_date
            
        except Exception as e:
            print(f"❌ Error extracting dates: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def train_and_predict(self, historical_data, test_window, ticker):
        """
        Train ARIMA on historical data and predict next day
        
        Args:
            historical_data: Training data
            test_window: Recent 2 days data
            ticker: Stock ticker
            
        Returns:
            prediction, confidence_interval
        """
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL AND MAKING PREDICTION")
        print(f"{'='*70}\n")
        
        try:
            # Prepare training data
            train_series = historical_data['Close'].values
            
            print(f"Training data points: {len(train_series)}")
            print(f"Training on data up to: {historical_data.index[-1].date()}\n")
            
            # Find best ARIMA parameters
            print("Finding optimal ARIMA parameters...")
            best_aic = np.inf
            best_order = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(train_series, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            print(f"✓ Best parameters: {best_order} (AIC: {best_aic:.2f})\n")
            
            # Train final model
            print("Training final model...")
            model = ARIMA(train_series, order=best_order)
            fitted_model = model.fit()
            
            # Make prediction for next day
            print("Generating prediction...\n")
            forecast = fitted_model.forecast(steps=1)
            prediction = forecast[0]
            
            # Get confidence interval
            forecast_obj = fitted_model.get_forecast(steps=1)
            conf_int = forecast_obj.conf_int()
            
            print(f"{'='*70}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*70}\n")
            print(f"📊 Predicted Close Price: ${prediction:.2f}")
            print(f"📉 95% Confidence Interval: ${conf_int[0][0]:.2f} - ${conf_int[0][1]:.2f}")
            
            # Additional context
            last_close = historical_data['Close'].iloc[-1]
            change = prediction - last_close
            change_pct = (change / last_close) * 100
            
            print(f"\n📈 Context:")
            print(f"  Last known close: ${last_close:.2f}")
            print(f"  Predicted change: ${change:+.2f} ({change_pct:+.2f}%)")
            
            return prediction, conf_int, best_order
            
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")
            return None, None, None
    
    def evaluate_prediction(self, prediction, actual_value):
        """
        Evaluate prediction if actual value is available
        
        Args:
            prediction: Predicted value
            actual_value: Actual value (if available)
            
        Returns:
            metrics dictionary
        """
        if actual_value is None:
            print(f"\n⚠ Cannot evaluate - actual value not available yet")
            return None
        
        print(f"\n{'='*70}")
        print(f"EVALUATION")
        print(f"{'='*70}\n")
        
        error = actual_value - prediction
        abs_error = abs(error)
        pct_error = (abs_error / actual_value) * 100
        
        metrics = {
            'predicted': float(prediction),
            'actual': float(actual_value),
            'error': float(error),
            'absolute_error': float(abs_error),
            'percentage_error': float(pct_error)
        }
        
        print(f"Predicted: ${prediction:.2f}")
        print(f"Actual:    ${actual_value:.2f}")
        print(f"Error:     ${error:+.2f}")
        print(f"Abs Error: ${abs_error:.2f}")
        print(f"MAPE:      {pct_error:.2f}%")
        
        if pct_error < 1:
            print(f"\n✅ Excellent prediction! (Error < 1%)")
        elif pct_error < 3:
            print(f"\n✅ Good prediction! (Error < 3%)")
        elif pct_error < 5:
            print(f"\n⚠ Acceptable prediction (Error < 5%)")
        else:
            print(f"\n❌ High error (Error > 5%)")
        
        return metrics
    
    def save_results(self, ticker, prediction, actual_value, conf_int, metrics, order):
        """Save test results to JSON"""
        
        results_file = os.path.join(self.test_dir, 'results', f'{ticker}_prediction_test.json')
        
        result_data = {
            'ticker': ticker,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_date': '2026-03-20',
            'arima_order': order,
            'prediction': float(prediction),
            'confidence_interval': {
                'lower': float(conf_int[0][0]),
                'upper': float(conf_int[0][1])
            },
            'actual_value': float(actual_value) if actual_value else None,
            'metrics': metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    def run_test(self, ticker, target_date='2026-03-20'):
        """
        Run complete prediction test
        
        Args:
            ticker: Stock ticker symbol
            target_date: Date to predict
        """
        print(f"\n{'#'*70}")
        print(f"# REAL-TIME PREDICTION TEST")
        print(f"# Ticker: {ticker}")
        print(f"# Target Date: {target_date}")
        print(f"# Test Directory: {self.test_dir}")
        print(f"{'#'*70}\n")
        
        # Step 1: Download latest data
        df = self.download_latest_data(ticker)
        if df is None:
            return False
        
        # Step 2: Extract specific dates
        historical_data, test_window, actual_value, target_dt = self.get_specific_dates(df, target_date)
        if historical_data is None:
            return False
        
        # Step 3: Train and predict
        prediction, conf_int, order = self.train_and_predict(historical_data, test_window, ticker)
        if prediction is None:
            return False
        
        # Step 4: Evaluate if actual data available
        metrics = self.evaluate_prediction(prediction, actual_value)
        
        # Step 5: Save results
        self.save_results(ticker, prediction, actual_value, conf_int, metrics, order)
        
        print(f"\n{'='*70}")
        print(f"✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}\n")
        
        return True


def main():
    """Run prediction test for 2 companies"""
    
    print(f"\n{'#'*70}")
    print(f"# STOCK PRICE PREDICTION TEST - MARCH 20, 2026")
    print(f"# Using data from March 18-19, 2026")
    print(f"{'#'*70}\n")
    
    # Test companies
    companies = {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.'
    }
    
    tester = RealtimePredictionTest()
    
    print(f"Testing with 2 companies: {', '.join([f'{t} ({n})' for t, n in companies.items()])}\n")
    print(f"⚠ Note: If March 20, 2026 market hasn't closed yet,")
    print(f"   actual values won't be available for comparison.\n")
    
    input("Press Enter to start the test...")
    
    results_summary = []
    
    for ticker, name in companies.items():
        print(f"\n\n{'='*70}")
        print(f"TESTING {ticker} - {name}")
        print(f"{'='*70}\n")
        
        success = tester.run_test(ticker, target_date='2026-03-20')
        
        if success:
            results_summary.append(f"✅ {ticker}: Success")
        else:
            results_summary.append(f"❌ {ticker}: Failed")
    
    # Final summary
    print(f"\n\n{'#'*70}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*70}\n")
    
    for result in results_summary:
        print(result)
    
    print(f"\n📁 All results saved in: {TEST_DIR}")
    print(f"   - Data: {TEST_DIR}/data/")
    print(f"   - Results: {TEST_DIR}/results/\n")
    
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()