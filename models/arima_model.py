import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    """
    ARIMA model for stock price forecasting
    """
    
    def __init__(self, order=(5, 1, 0)):
        """
        Initialize ARIMA model
        
        Args:
            order: Tuple of (p, d, q) parameters
                   p: AR order (autoregressive)
                   d: Differencing order
                   q: MA order (moving average)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.training_time = None
    
    def check_stationarity(self, series, plot=False):
        """
        Check if time series is stationary using ADF test
        
        Args:
            series: Time series data
            plot: Whether to plot the series
            
        Returns:
            Boolean indicating stationarity
        """
        result = adfuller(series.dropna())
        
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")
        
        is_stationary = result[1] < 0.05
        print(f"\nSeries is {'stationary' if is_stationary else 'non-stationary'}")
        
        if plot:
            plt.figure(figsize=(12, 4))
            plt.plot(series)
            plt.title('Time Series')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.show()
        
        return is_stationary
    
    def find_optimal_order(self, train_data, max_p=5, max_q=5):
        """
        Find optimal ARIMA order using AIC
        
        Args:
            train_data: Training time series
            max_p: Maximum p value to test
            max_q: Maximum q value to test
            
        Returns:
            Optimal (p, d, q) order
        """
        print("Finding optimal ARIMA parameters...")
        
        best_aic = np.inf
        best_order = None
        
        # Test different combinations
        for p in range(max_p + 1):
            for d in range(2):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"Optimal order: {best_order} with AIC: {best_aic:.2f}")
        return best_order
    
    def train(self, train_data, auto_order=False):
        """
        Train ARIMA model
        
        Args:
            train_data: Training time series (pandas Series or DataFrame)
            auto_order: Whether to automatically find optimal order
            
        Returns:
            Fitted model
        """
        start_time = time.time()
        
        # Extract series if DataFrame
        if isinstance(train_data, pd.DataFrame):
            train_series = train_data['Close']
        else:
            train_series = train_data
        
        # Find optimal order if requested
        if auto_order:
            self.order = self.find_optimal_order(train_series)
        
        print(f"\nTraining ARIMA{self.order}...")
        
        # Fit model
        self.model = ARIMA(train_series, order=self.order)
        self.fitted_model = self.model.fit()
        
        self.training_time = time.time() - start_time
        
        print(f"✓ Model trained in {self.training_time:.2f} seconds")
        print(f"\nModel Summary:")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        
        return self.fitted_model
    
    def predict(self, steps):
        """
        Make predictions
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Predictions array
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.fitted_model.forecast(steps=steps)
        return predictions
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Args:
            test_data: Test time series
            
        Returns:
            Predictions on test set
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract series if DataFrame
        if isinstance(test_data, pd.DataFrame):
            test_series = test_data['Close']
        else:
            test_series = test_data
        
        # Make predictions
        predictions = self.predict(steps=len(test_series))
        
        return predictions
    
    def plot_diagnostics(self):
        """
        Plot model diagnostics
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.fitted_model.plot_diagnostics(figsize=(15, 8))
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self):
        """
        Get model summary
        
        Returns:
            Summary string
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.fitted_model.summary()

def run_arima_experiment(ticker, train_df, test_df, order=(5, 1, 0), auto_order=False):
    """
    Run complete ARIMA experiment for a stock
    
    Args:
        ticker: Stock ticker
        train_df: Training DataFrame
        test_df: Testing DataFrame
        order: ARIMA order (p, d, q)
        auto_order: Whether to find optimal order
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Running ARIMA for {ticker}")
    print(f"{'='*70}\n")
    
    # Initialize model
    arima = ARIMAForecaster(order=order)
    
    # Check stationarity
    print("Checking stationarity...")
    arima.check_stationarity(train_df['Close'])
    print()
    
    # Train model
    arima.train(train_df, auto_order=auto_order)
    
    # Make predictions
    print(f"\nMaking predictions on test set ({len(test_df)} steps)...")
    predictions = arima.evaluate(test_df)
    
    # Get actual values
    y_true = test_df['Close'].values
    y_pred = predictions.values
    
    # Calculate metrics
    from utils.evaluation import ModelEvaluator
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    print(f"\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    evaluator.save_results(
        ticker=ticker,
        model_name='ARIMA',
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        training_time=arima.training_time
    )
    
    # Plot predictions
    evaluator.plot_predictions(
        ticker=ticker,
        model_name='ARIMA',
        y_true=y_true,
        y_pred=y_pred,
        dates=test_df.index
    )
    
    return {
        'model': arima,
        'predictions': predictions,
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred
    }

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    
    from utils.data_loader import StockDataLoader
    from utils.preprocessing import StockDataPreprocessor
    
    # Load and preprocess data
    loader = StockDataLoader()
    preprocessor = StockDataPreprocessor()
    
    ticker = 'AAPL'
    data = loader.load_stock_data(ticker)
    
    if data is not None:
        # Preprocess
        processed = preprocessor.preprocess_stock(data, ticker, add_indicators=False)
        
        # Run ARIMA
        results = run_arima_experiment(
            ticker=ticker,
            train_df=processed['train'],
            test_df=processed['test'],
            auto_order=True
        )