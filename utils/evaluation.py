import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class ModelEvaluator:
    """
    Evaluate and compare time series forecasting models
    """
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate various evaluation metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred)
        }
        
        return metrics
    
    def save_results(self, ticker, model_name, metrics, y_true, y_pred, training_time=None):
        """
        Save model results
        
        Args:
            ticker: Stock ticker
            model_name: Name of the model
            metrics: Dictionary of metrics
            y_true: Actual values
            y_pred: Predicted values
            training_time: Time taken to train (optional)
        """
        key = f"{ticker}_{model_name}"
        
        # Load existing results first
        results_file = os.path.join(self.results_dir, 'model_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        
        self.results[key] = {
            'ticker': ticker,
            'model': model_name,
            'metrics': metrics,
            'predictions': {
                'y_true': y_true.tolist() if isinstance(y_true, np.ndarray) else list(y_true),
                'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred)
            },
            'training_time': training_time
        }
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"✓ Results saved to: {results_file}")
        print(f"✓ Key: {key}")
    
    def load_results(self):
        """
        Load previously saved results
        
        Returns:
            Dictionary of results
        """
        results_file = os.path.join(self.results_dir, 'model_results.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            return self.results
        else:
            print("No results file found")
            return {}
    
    def plot_predictions(self, ticker, model_name, y_true, y_pred, dates=None, save=True):
        """
        Plot actual vs predicted values
        
        Args:
            ticker: Stock ticker
            model_name: Name of the model
            y_true: Actual values
            y_pred: Predicted values
            dates: Date index (optional)
            save: Whether to save the plot
        """
        plt.figure(figsize=(14, 6))
        
        if dates is not None:
            plt.plot(dates, y_true, label='Actual', color='blue', linewidth=2)
            plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
        else:
            plt.plot(y_true, label='Actual', color='blue', linewidth=2)
            plt.plot(y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
        
        plt.title(f'{ticker} - {model_name} Predictions', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, f"{ticker}_{model_name}_predictions.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {filename}")
        
        plt.close()
    
    def compare_models(self, ticker):
        """
        Compare all models for a specific ticker
        
        Args:
            ticker: Stock ticker
            
        Returns:
            DataFrame with comparison
        """
        if not self.results:
            self.load_results()
        
        ticker_results = {k: v for k, v in self.results.items() if v['ticker'] == ticker}
        
        if not ticker_results:
            print(f"No results found for {ticker}")
            return None
        
        comparison = []
        for key, result in ticker_results.items():
            row = {
                'Model': result['model'],
                **result['metrics']
            }
            if result.get('training_time'):
                row['Training Time (s)'] = result['training_time']
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('RMSE')
        
        return df
    
    def plot_model_comparison(self, ticker, metric='RMSE', save=True):
        """
        Plot comparison of different models
        
        Args:
            ticker: Stock ticker
            metric: Metric to compare ('RMSE', 'MAE', 'MAPE', 'R2')
            save: Whether to save the plot
        """
        comparison_df = self.compare_models(ticker)
        
        if comparison_df is None:
            return
        
        plt.figure(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(comparison_df))
        bars = plt.bar(comparison_df['Model'], comparison_df[metric], color=colors)
        
        plt.title(f'{ticker} - Model Comparison ({metric})', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, f"{ticker}_model_comparison_{metric}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {filename}")
        
        plt.close()
    
    def generate_report(self, ticker):
        """
        Generate a comprehensive evaluation report
        
        Args:
            ticker: Stock ticker
        """
        comparison_df = self.compare_models(ticker)
        
        if comparison_df is None:
            return
        
        print(f"\n{'='*70}")
        print(f"EVALUATION REPORT FOR {ticker}")
        print(f"{'='*70}\n")
        
        print(comparison_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print(f"BEST PERFORMING MODELS")
        print(f"{'='*70}\n")
        
        for metric in ['RMSE', 'MAE', 'MAPE']:
            best_model = comparison_df.loc[comparison_df[metric].idxmin()]
            print(f"Best {metric}: {best_model['Model']} ({best_model[metric]:.4f})")
        
        best_r2 = comparison_df.loc[comparison_df['R2'].idxmax()]
        print(f"Best R2: {best_r2['Model']} ({best_r2['R2']:.4f})")
        
        print(f"\n{'='*70}\n")

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Simulate some predictions
    y_true = np.random.randn(100) * 10 + 100
    y_pred = y_true + np.random.randn(100) * 2
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)