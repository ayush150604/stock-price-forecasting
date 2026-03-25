# 📈 StockCast: Stock Price Trend Forecasting Using Time-Series Models

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-green.svg)

An end-to-end, production-ready stock price forecasting system and interactive dashboard. This project implements both statistical (ARIMA) and deep learning (LSTM) models to forecast stock prices for 10 major companies, utilizing advanced time-series techniques like log-return transformations to ensure scale-invariance and stationarity.

---

## 🎯 Project Explanation

Predicting stock prices is notoriously difficult due to extreme market volatility, regime shifts, and non-stationary data. **StockCast** tackles these challenges by comparing traditional statistical forecasting against modern neural networks. 

Instead of training models on raw, non-stationary stock prices (which often results in a "flat-line" random walk prediction), this system natively converts raw prices into **log-returns**. By predicting the percentage change rather than the raw price, the models can accurately handle extreme volatility and scale-invariance (e.g., preventing LSTM scaler extrapolation during all-time market highs). 

### ✨ Core Features
* **Dual-Model Forecasting:** Side-by-side evaluation of an auto-tuned ARIMA model (using 1-step rolling forecasts) vs. a deep learning LSTM network (128→64→32 architecture with 60-day lookback).
* **Real-Time "Time Machine" Prediction:** Predict the next trading day's close from *any* selected historical date to back-test accuracy live, complete with 95% confidence intervals.
* **StockBot AI Assistant:** An integrated chatbot that fetches live prices, detects market crashes via SPY indexing, and explains model architectures.
* **Stock Alert Monitor:** Set custom drop-thresholds (e.g., 3%) for live crash detection with a 60-second auto-refresh.
* **Role-Based Access Control:** Secure login system with Admin, Analyst, and Viewer profiles.

---

## 🛠️ Tech Stack

**Data Pipeline & Processing:**
* `yfinance` (Live data ingestion)
* `pandas` & `numpy` (Data manipulation & log-return scaling)
* `scikit-learn` (MinMaxScaler & Evaluation Metrics)

**Machine Learning & AI:**
* `statsmodels` (ARIMA forecasting & AIC optimization)
* `TensorFlow` / `Keras` (LSTM neural networks, Huber loss, Early Stopping)

**Frontend & Visualization:**
* `Streamlit` (Interactive web application framework)
* `Plotly` (Interactive charting, bounded confidence intervals)
* `matplotlib` & `seaborn` (Static plotting)

---

## ⚙️ How It Works

1.  **Data Ingestion:** The system downloads 5 years of historical data for 10 major large-cap stocks (AAPL, MSFT, NVDA, META, etc.) directly from Yahoo Finance.
2.  **Preprocessing & Transformation:** * Data is split 80/20 chronologically.
    * Raw close prices ($P_t$) are converted to log-returns ($r_t = \log(P_t / P_{t-1})$) to make the series stationary and immune to extreme price regime shifts.
3.  **Model Training:**
    * **ARIMA:** Auto-tunes $(p,d,q)$ parameters via AIC minimization and utilizes a rolling forecast to predict one day ahead, updating its history with each step.
    * **LSTM:** Uses a sliding window of 60 days. It trains on log-returns for highly volatile stocks to prevent the neural network from extrapolating beyond its training scaler limits.
4.  **Inference & Conversion:** Predicted returns are converted back into dollar values ($P_{t+1} = P_t \times \exp(r_{t+1})$) and rendered onto the UI alongside 95% confidence bands.

---

## 📸 Output Screenshots


### 1. Real-Time Prediction & Visualization
![Real-Time Prediction](images/realtime_prediction.png)
*Live next-day forecasting with exact 95% confidence intervals, bounded rendering, and historical trend lines.*

### 2. Individual Stock Analysis (ARIMA vs LSTM)
![Model Comparison](images/model_comparison.png)
*Side-by-side metric comparison (RMSE, MAE, MAPE, R²) and interactive prediction toggles.*

### 3. StockBot AI & Alert Monitor
![Chatbot and Alerts](images/chatbot_alerts.png)
*Live market crash detection, custom threshold alerts, and the integrated StockBot assistant.*

---

## 🚀 Installation & Usage

### 1. Clone & Setup
```bash
git clone [https://github.com/ayush150604/stock-price-forecasting.git](https://github.com/ayush150604/stock-price-forecasting.git)
cd stock-price-forecasting
pip install -r requirements.txt
Gemini said
Here is a comprehensive, production-ready README.md tailored exactly to the architecture, features, and breakthroughs of your project.

It is structured to instantly impress evaluators, highlighting the advanced time-series techniques (like log-returns) and the robust Streamlit dashboard you've built.

Markdown
# 📈 StockCast: Stock Price Trend Forecasting Using Time-Series Models

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-green.svg)

An end-to-end, production-ready stock price forecasting system and interactive dashboard. This project implements both statistical (ARIMA) and deep learning (LSTM) models to forecast stock prices for 10 major companies, utilizing advanced time-series techniques like log-return transformations to ensure scale-invariance and stationarity.

---

## 🎯 Project Explanation

Predicting stock prices is notoriously difficult due to extreme market volatility, regime shifts, and non-stationary data. **StockCast** tackles these challenges by comparing traditional statistical forecasting against modern neural networks. 

Instead of training models on raw, non-stationary stock prices (which often results in a "flat-line" random walk prediction), this system natively converts raw prices into **log-returns**. By predicting the percentage change rather than the raw price, the models can accurately handle extreme volatility and scale-invariance (e.g., preventing LSTM scaler extrapolation during all-time market highs). 

### ✨ Core Features
* **Dual-Model Forecasting:** Side-by-side evaluation of an auto-tuned ARIMA model (using 1-step rolling forecasts) vs. a deep learning LSTM network (128→64→32 architecture with 60-day lookback).
* **Real-Time "Time Machine" Prediction:** Predict the next trading day's close from *any* selected historical date to back-test accuracy live, complete with 95% confidence intervals.
* **StockBot AI Assistant:** An integrated chatbot that fetches live prices, detects market crashes via SPY indexing, and explains model architectures.
* **Stock Alert Monitor:** Set custom drop-thresholds (e.g., 3%) for live crash detection with a 60-second auto-refresh.
* **Role-Based Access Control:** Secure login system with Admin, Analyst, and Viewer profiles.

---

## 🛠️ Tech Stack

**Data Pipeline & Processing:**
* `yfinance` (Live data ingestion)
* `pandas` & `numpy` (Data manipulation & log-return scaling)
* `scikit-learn` (MinMaxScaler & Evaluation Metrics)

**Machine Learning & AI:**
* `statsmodels` (ARIMA forecasting & AIC optimization)
* `TensorFlow` / `Keras` (LSTM neural networks, Huber loss, Early Stopping)

**Frontend & Visualization:**
* `Streamlit` (Interactive web application framework)
* `Plotly` (Interactive charting, bounded confidence intervals)
* `matplotlib` & `seaborn` (Static plotting)

---

## ⚙️ How It Works

1.  **Data Ingestion:** The system downloads 5 years of historical data for 10 major large-cap stocks (AAPL, MSFT, NVDA, META, etc.) directly from Yahoo Finance.
2.  **Preprocessing & Transformation:** * Data is split 80/20 chronologically.
    * Raw close prices ($P_t$) are converted to log-returns ($r_t = \log(P_t / P_{t-1})$) to make the series stationary and immune to extreme price regime shifts.
3.  **Model Training:**
    * **ARIMA:** Auto-tunes $(p,d,q)$ parameters via AIC minimization and utilizes a rolling forecast to predict one day ahead, updating its history with each step.
    * **LSTM:** Uses a sliding window of 60 days. It trains on log-returns for highly volatile stocks to prevent the neural network from extrapolating beyond its training scaler limits.
4.  **Inference & Conversion:** Predicted returns are converted back into dollar values ($P_{t+1} = P_t \times \exp(r_{t+1})$) and rendered onto the UI alongside 95% confidence bands.

---



### 1. Real-Time Prediction & Visualization
![Real-Time Prediction](images/realtime_prediction.png)
*Live next-day forecasting with exact 95% confidence intervals, bounded rendering, and historical trend lines.*

### 2. Individual Stock Analysis (ARIMA vs LSTM)
![Model Comparison](images/model_comparison.png)
*Side-by-side metric comparison (RMSE, MAE, MAPE, R²) and interactive prediction toggles.*

### 3. StockBot AI & Alert Monitor
![Chatbot and Alerts](images/chatbot_alerts.png)
*Live market crash detection, custom threshold alerts, and the integrated StockBot assistant.*

---

## 🚀 Installation & Usage

### 1. Clone & Setup
```bash
git clone [https://github.com/ayush150604/stock-price-forecasting.git](https://github.com/ayush150604/stock-price-forecasting.git)
cd stock-price-forecasting
pip install -r requirements.txt

##2. Generate Model Results
Generate the historical ARIMA predictions and model_results.json:

python main_simple.py
Train the deep learning LSTM models (Requires TensorFlow):

python models/lstm_model.py

3. Launch the Dashboard
streamlit run app_with_realtime.py
📊 Evaluation & Results
The models are evaluated on the final 20% of the historical dataset (approx. 252 trading days).

ARIMA Performance: Highly effective on linear, near-random walk stocks, achieving MAPE between 1% - 3% and R² > 0.97 across the board. Real-time live testing on MSFT yielded an error of just 0.46%.

LSTM Performance: Captures non-linear momentum. Highly effective when targeted specifically at high-volatility regimes using log-return scaling (e.g., NVDA R² improved to 0.98).

🔮 Future Enhancements
Feature Engineering: Integrate technical indicators (RSI, MACD, Bollinger Bands) to improve LSTM accuracy.

Additional Models: Implement Prophet (Facebook) and GRU architectures.

Sentiment Analysis: Factor in financial news and Twitter sentiment for predictive weighting.

👤 Author
Ayush Singh Final Year Major Project

Dr. A.P.J. Abdul Kalam Technical University, Lucknow, Uttar Pradesh

GitHub: @ayush150604

LinkedIn: Ayush Singh

Email: ayushsingh1562004@gmail.com

