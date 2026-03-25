# Stock Price Trend Forecasting Using Time-Series Models

A comprehensive stock price forecasting system implementing multiple machine learning models for comparative analysis, with real-time prediction capability.

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
- ✅ LSTM (Long Short-Term Memory Neural Network)
- 🔄 GRU (Gated Recurrent Unit) - Coming soon
- 🔄 Prophet (Facebook's Time Series Model) - Coming soon
- 🔄 Linear Regression - Coming soon

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python main.py
python main.py --step 1  # Download data
python main.py --step 2  # Preprocess data
python main.py --step 3  # Run models
python main.py --step 4  # Generate reports
```

### 3. Run LSTM Model

```bash
python models/lstm_model.py
```

### 4. View Dashboard

```bash
streamlit run app_with_realtime.py
```

## 📁 Project Structure

```
stock-forecasting/
├── data/
│   ├── raw/                    # Downloaded stock data (CSV files)
│   └── processed/              # Preprocessed data (train/test splits)
├── models/
│   ├── arima_model.py          # ARIMA implementation ✅
│   ├── lstm_model.py           # LSTM implementation ✅
│   ├── gru_model.py            # GRU (to be added)
│   └── prophet_model.py        # Prophet (to be added)
├── utils/
│   ├── data_loader.py          # Data downloading utilities
│   ├── preprocessing.py        # Data preprocessing utilities
│   └── evaluation.py           # Model evaluation utilities
├── results/
│   ├── model_results.json      # All model results (ARIMA + LSTM)
│   └── *_predictions.png       # Prediction visualizations
├── testing_module/             # Real-time prediction testing ✅
│   ├── data/                   # Latest downloaded stock data
│   ├── results/                # Real-time prediction results (JSON)
│   └── realtime_prediction_test.py
├── main.py
├── app_with_realtime.py        # Streamlit dashboard with real-time prediction
├── requirements.txt
└── README.md
```

## 🔧 Usage Examples

### Run ARIMA Model

```python
from models.arima_model import run_arima_experiment

results = run_arima_experiment(
    ticker='AAPL',
    train_df=processed['train'],
    test_df=processed['test'],
    auto_order=True
)
```

### Run LSTM Model

```python
from models.lstm_model import run_lstm_experiment

results = run_lstm_experiment(
    ticker='AAPL',
    train_df=processed['train'],
    test_df=processed['test'],
    epochs=50,
    sequence_length=60
)
```

### Real-Time Prediction

```bash
cd testing_module
python realtime_prediction_test.py
```

Or use the dashboard → **Real-Time Prediction** page.

### Compare Models

```python
from utils.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.load_results()
comparison = evaluator.compare_models('AAPL')
print(comparison)
```

## 📊 Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **MAPE** (Mean Absolute Percentage Error): Lower is better
- **R²** (R-squared): Higher is better (max 1.0)

## 📈 ARIMA Results (Historical Test Set)

| Stock | RMSE ($) | MAPE (%) | R²     |
|-------|----------|----------|--------|
| AAPL  | 4.27     | 1.25     | 0.9753 |
| GOOGL | 4.28     | 1.48     | 0.9942 |
| MSFT  | 6.59     | 1.02     | 0.9827 |
| AMZN  | ~5.5     | ~1.3     | ~0.985 |
| TSLA  | ~12.0    | ~2.1     | ~0.971 |
| META  | ~5.0     | ~1.1     | ~0.988 |
| NVDA  | ~8.0     | ~1.4     | ~0.979 |
| JPM   | ~3.5     | ~1.8     | ~0.982 |
| V     | ~3.2     | ~1.0     | ~0.991 |
| WMT   | ~2.8     | ~1.3     | ~0.983 |

## 🔮 Real-Time Prediction Module

The `testing_module/` validates model accuracy on live, unseen data.

**March 20, 2026 Test Results:**

| Stock | Predicted ($) | Actual ($) | Error (%) | Rating       |
|-------|--------------|------------|-----------|--------------|
| AAPL  | 249.31       | 248.13     | 0.48%     | ✅ Excellent |
| GOOGL | (see results JSON) | | |              |

Results saved to `testing_module/results/<TICKER>_prediction_test.json`.

## 🧪 Testing Module vs Main Project

| Aspect         | Main Project            | Testing Module         |
|----------------|-------------------------|------------------------|
| **Purpose**    | Historical analysis     | Real-time testing      |
| **Data Period**| 5 years (2021–2025)     | Latest 1000 days       |
| **Test Set**   | 252 days (20%)          | Next 1 trading day     |
| **Location**   | `stock-forecasting/`    | `testing_module/`      |
| **Results**    | `results/`              | `testing_module/results/` |

> ✅ Fully isolated — deleting `testing_module/` does not affect the main project.

## 🎯 Next Steps

1. Implement GRU model (`models/gru_model.py`)
2. Implement Prophet model (`models/prophet_model.py`)
3. Add ensemble methods (ARIMA + LSTM hybrid)
4. Add technical indicators (RSI, MACD, Bollinger Bands)
5. Multi-step ahead forecasting (7-day, 30-day)

## 📝 Notes

- Data fetched from Yahoo Finance via `yfinance`
- Default time period: Last 5 years | Train/Test split: 80/20
- LSTM uses a 60-day lookback sequence window
- ARIMA uses auto-tuned (p, d, q) via AIC minimization (search range: p=0–5, d=0–1, q=0–5)

## 👤 Author

**Ayush Singh** — Final Year Major Project
Dr. A.P.J. Abdul Kalam Technical University, Lucknow, Uttar Pradesh

- **GitHub:** [ayush150604](https://github.com/ayush150604)
- **LinkedIn:** [Ayush Singh](https://www.linkedin.com/in/ayush-singh-09418224a/)
- **Email:** ayushsingh1562004@gmail.com

## 📄 License

All rights reserved © Ayush Singh