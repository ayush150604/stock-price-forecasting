# Stock Price Trend Forecasting Using Time-Series Models

A comprehensive stock price forecasting system implementing multiple machine learning models for comparative analysis.

## 📋 Project Overview

This project forecasts stock prices for 10 major companies using various time-series models and provides a comparative analysis through an interactive Streamlit dashboard.

**Companies Analyzed:**
- Apple Inc. (AAPL)
- Alphabet Inc. (GOOGL)
- Microsoft Corporation (MSFT)
- Amazon.com Inc. (AMZN)
- Tesla Inc. (TSLA)
- Meta Platforms Inc. (META)
- NVIDIA Corporation (NVDA)
- JPMorgan Chase & Co. (JPM)
- Visa Inc. (V)
- Walmart Inc. (WMT)

**Models Implemented:**
- ✅ ARIMA (Autoregressive Integrated Moving Average)
- 🔄 LSTM (Long Short-Term Memory) - Coming soon
- 🔄 GRU (Gated Recurrent Unit) - Coming soon
- 🔄 Prophet (Facebook's Time Series Model) - Coming soon
- 🔄 Linear Regression - Coming soon

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create the project directory
mkdir stock-forecasting
cd stock-forecasting

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Run everything at once
python main.py

# Or run individual steps
python main.py --step 1  # Download data
python main.py --step 2  # Preprocess data
python main.py --step 3  # Run models
python main.py --step 4  # Generate reports
```

### 3. View Results

```bash
# Launch the Streamlit dashboard (once created)
streamlit run app.py
```

## 📁 Project Structure

```
stock-price-forecasting/
├── data/
│   ├── raw/              # Downloaded stock data (CSV files)
│   └── processed/        # Preprocessed data (train/test splits)
├── models/
│   ├── arima_model.py    # ARIMA implementation
│   ├── lstm_model.py     # LSTM implementation (to be added)
│   ├── gru_model.py      # GRU implementation (to be added)
│   └── prophet_model.py  # Prophet implementation (to be added)
├── utils/
│   ├── data_loader.py    # Data downloading utilities
│   ├── preprocessing.py  # Data preprocessing utilities
│   └── evaluation.py     # Model evaluation utilities
├── results/
│   ├── model_results.json           # All model results
│   └── *_predictions.png            # Prediction visualizations
├── notebooks/
│   └── exploratory_analysis.ipynb   # Jupyter notebook for exploration
├── main.py               # Main pipeline script
├── app.py                # Streamlit dashboard (to be created)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Usage Examples

### Download Data for All Companies

```python
from utils.data_loader import StockDataLoader

loader = StockDataLoader()
all_data = loader.download_all_stocks(period='5y')
```

### Preprocess a Single Stock

```python
from utils.preprocessing import StockDataPreprocessor

preprocessor = StockDataPreprocessor()
data = loader.load_stock_data('AAPL')
processed = preprocessor.preprocess_stock(data, 'AAPL')
```

### Run ARIMA Model

```python
from models.arima_model import run_arima_experiment

results = run_arima_experiment(
    ticker='AAPL',
    train_df=processed['train'],
    test_df=processed['test'],
    order=(5, 1, 0),
    auto_order=True  # Automatically find best parameters
)
```

### Evaluate and Compare Models

```python
from utils.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.load_results()

# Compare models for a specific stock
comparison = evaluator.compare_models('AAPL')
print(comparison)

# Generate comprehensive report
evaluator.generate_report('AAPL')

# Plot comparison
evaluator.plot_model_comparison('AAPL', metric='RMSE')
```

## 📊 Evaluation Metrics

The project uses the following metrics to evaluate model performance:

- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **MAPE** (Mean Absolute Percentage Error): Lower is better
- **R²** (R-squared): Higher is better (max 1.0)

## 🎯 Next Steps

1. **Add More Models**
   - Implement LSTM model in `models/lstm_model.py`
   - Implement GRU model in `models/gru_model.py`
   - Implement Prophet model in `models/prophet_model.py`
   - Add ensemble methods

2. **Create Streamlit Dashboard**
   - Interactive stock selector
   - Real-time predictions
   - Model comparison visualizations
   - Download results feature

3. **Advanced Features**
   - Add technical indicators for feature engineering
   - Implement hyperparameter tuning
   - Add cross-validation
   - Real-time data updates

## 📝 Notes

- Data is fetched from Yahoo Finance using `yfinance` library
- Default time period: Last 5 years
- Train/Test split: 80/20
- All results are saved in JSON format for easy comparison

## 🤝 Contributing

This is a final year project. Feel free to:
- Add new models
- Improve preprocessing techniques
- Enhance visualizations
- Optimize hyperparameters

## 📄 License

All right are reserved to ayush singh

## 👤 Author

Ayush Singh's Final Year Project - Stock Price Trend Forecasting Using Time-Series Models


USERNAME → ayush150604
Name → Ayush Singh
Email → ayushsingh1562004@gmail.com
University Name → Dr. A.P.J. Abdul Kalam Technical University, a major state technical university located in Lucknow, Uttar Pradesh
LinkedIn → https://www.linkedin.com/in/ayush-singh-09418224a/

