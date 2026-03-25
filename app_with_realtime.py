"""
Enhanced Stock Price Forecasting Dashboard with Real-Time Prediction,
Login System, AI Chatbot, and Stock Alert Monitor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import warnings
import time
import hashlib
warnings.filterwarnings('ignore')

# LSTM real-time prediction — optional, disabled gracefully if TensorFlow missing
try:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.lstm_model import LSTMForecaster, STOCK_CONFIGS as LSTM_STOCK_CONFIGS
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    h1 { color: #1f77b4; padding-bottom: 20px; }
    h2 { color: #2c3e50; padding-top: 20px; }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    /* Login card */
    .login-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 40px;
        border-radius: 16px;
        border: 1px solid #374151;
        max-width: 420px;
        margin: 60px auto;
    }
    /* Chat bubbles */
    .chat-user {
        background: #1f77b4;
        color: white;
        padding: 10px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 75%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .chat-bot {
        background: #f0f2f6;
        color: #1a1a2e;
        padding: 10px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        max-width: 75%;
        word-wrap: break-word;
    }
    .chat-container {
        height: 420px;
        overflow-y: auto;
        padding: 12px;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background: #fafafa;
        margin-bottom: 12px;
    }
    /* Alert cards */
    .alert-danger {
        background: #fff0f0;
        border-left: 5px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .alert-warning {
        background: #fffbf0;
        border-left: 5px solid #f39c12;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .alert-ok {
        background: #f0fff4;
        border-left: 5px solid #27ae60;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# LOGIN SYSTEM
# ════════════════════════════════════════════════════════════════════════════

# Pre-defined user accounts  (username → {password_hash, display_name, role})
def _h(p):
    return hashlib.sha256(p.encode()).hexdigest()

USERS = {
    "ayush2004": {
        "password_hash": _h("Ayush@1234"),
        "display_name":  "Ayush Singh",
        "role":          "Admin"
    },
    "analyst01": {
        "password_hash": _h("Analyst#99"),
        "display_name":  "Priya Sharma",
        "role":          "Analyst"
    },
    "viewer02": {
        "password_hash": _h("View@2026"),
        "display_name":  "Rahul Verma",
        "role":          "Viewer"
    },
}

def show_login_page():
    """Render the login page. Returns True if login succeeded."""
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center; margin-top:40px; margin-bottom:30px;">
            <h1 style="font-size:2.4rem; color:#1f77b4; margin-bottom:4px;">📈 StockCast</h1>
            <p style="color:#6b7280; font-size:1rem;">AI-Powered Stock Price Forecasting</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("#### 🔐 Sign in to your account")
            st.markdown("")

            username = st.text_input("Username", placeholder="Enter your username",
                                     key="login_user")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter your password",
                                     key="login_pass")

            st.markdown("")
            login_btn = st.button("Sign In →", type="primary",
                                  use_container_width=True)

            if login_btn:
                if username in USERS and                    USERS[username]["password_hash"] == _h(password):
                    st.session_state["logged_in"]    = True
                    st.session_state["username"]     = username
                    st.session_state["display_name"] = USERS[username]["display_name"]
                    st.session_state["role"]         = USERS[username]["role"]
                    st.session_state["chat_history"] = []
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

            st.markdown("---")
            st.markdown("""
            <div style="font-size:0.8rem; color:#9ca3af; text-align:center;">
            Demo accounts:<br>
            <b>ayush2004</b> / Ayush@1234 &nbsp;|&nbsp;
            <b>analyst01</b> / Analyst#99 &nbsp;|&nbsp;
            <b>viewer02</b> / View@2026
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ════════════════════════════════════════════════════════════════════════════

def chatbot_response(user_msg):
    msg = user_msg.lower().strip()

    if any(w in msg for w in ["hello","hi","hey","namaste"]):
        return "Hello! I am StockBot. Ask me about stock prices, market crashes, ARIMA, LSTM, or how to use this dashboard."

    if "who are you" in msg or "what are you" in msg:
        return "I am StockBot, the AI assistant for this Stock Forecasting Dashboard."

    if "help" in msg or "what can you do" in msg:
        lines = [
            "Here is what I can do:",
            "",
            "- Stock price: ask 'price of AAPL' or 'current MSFT price'",
            "- Market crash: ask 'is there a crash today'",
            "- Alerts: go to the Alerts tab to monitor stocks",
            "- Model info: ask 'what is ARIMA' or 'how does LSTM work'",
            "- Prediction: ask 'predict GOOGL' to go to the prediction page",
        ]
        return "\n".join(lines)

    for ticker in ["aapl","msft","googl","amzn","tsla","meta","nvda","jpm","v","wmt"]:
        if ticker in msg and any(w in msg for w in ["price","cost","worth","value","trading","close","open"]):
            try:
                data = yf.Ticker(ticker.upper()).history(period="2d")
                if not data.empty:
                    price  = data["Close"].iloc[-1]
                    prev   = data["Close"].iloc[-2] if len(data) > 1 else price
                    change = ((price - prev) / prev) * 100
                    arrow  = "UP" if change >= 0 else "DOWN"
                    return (f"{ticker.upper()} latest close: ${price:.2f} | "
                            f"{arrow} {abs(change):.2f}% from prev close (${prev:.2f}). "
                            f"Use Real-Time Prediction tab for a forecast.")
            except Exception:
                return f"Sorry, could not fetch live data for {ticker.upper()} right now."

    if any(w in msg for w in ["crash","collapse","market down","big drop","falling"]):
        try:
            spy = yf.Ticker("SPY").history(period="5d")
            if not spy.empty and len(spy) >= 2:
                ret = ((spy["Close"].iloc[-1] - spy["Close"].iloc[-2]) / spy["Close"].iloc[-2]) * 100
                if ret <= -3:
                    return f"CRASH ALERT: S&P 500 (SPY) is down {ret:.2f}% today. Significant market drop detected."
                elif ret <= -1:
                    return f"Market Decline: S&P 500 (SPY) is down {ret:.2f}% today. Not a crash, but notable weakness."
                else:
                    return f"No Crash Detected. S&P 500 (SPY) is at {ret:+.2f}% today. Markets appear stable."
        except Exception:
            return "Could not fetch live market data. Check your Alerts tab for monitoring."

    if "arima" in msg:
        lines = [
            "ARIMA = Autoregressive Integrated Moving Average.",
            "",
            "- AR(p): uses past prices to predict future ones",
            "- I(d): differencing to make the series stationary",
            "- MA(q): uses past forecast errors",
            "",
            "In this project, ARIMA is trained on log returns to avoid flat predictions.",
            "Parameters are auto-tuned using AIC minimization."
        ]
        return "\n".join(lines)

    if "lstm" in msg or "long short" in msg:
        lines = [
            "LSTM = Long Short-Term Memory neural network.",
            "",
            "- Uses a 60-day lookback window",
            "- Architecture: 128 -> 64 -> 32 LSTM units + Dropout(0.2)",
            "- Unlike ARIMA, LSTM captures non-linear patterns and momentum.",
            "",
            "Train it by running: python models/lstm_model.py"
        ]
        return "\n".join(lines)

    if any(w in msg for w in ["predict","forecast","tomorrow","next day"]):
        for ticker in ["aapl","msft","googl","amzn","tsla","meta","nvda","jpm","v","wmt"]:
            if ticker in msg:
                return f"Go to Real-Time Prediction tab, select {ticker.upper()}, choose a date, and click Predict Closing Price."
        return "Go to the Real-Time Prediction tab, select a stock and date, then click Predict Closing Price."

    if any(w in msg for w in ["alert","notify","monitor","watch"]):
        lines = [
            "Go to the Alerts tab in the sidebar.",
            "",
            "You can:",
            "- Select any stocks to monitor",
            "- Set a crash threshold (e.g. alert if price drops > 3%)",
            "- Click Check Now to scan instantly",
            "- Enable Auto-refresh to monitor every 60 seconds"
        ]
        return "\n".join(lines)

    if any(w in msg for w in ["bye","goodbye","thanks","thank you"]):
        return "Goodbye! Good luck with your trading analysis."

    lines = [
        "I am not sure about that. Try asking:",
        "- 'price of AAPL'",
        "- 'is there a crash today'",
        "- 'what is ARIMA'",
        "- 'how to set an alert'",
        "- 'predict TSLA'"
    ]
    return "\n".join(lines)


def check_stock_alerts(tickers: list, threshold_pct: float) -> list:
    """
    For each ticker in the list, download the last 2 days of data and
    compute the day-over-day % change.  Returns a list of alert dicts.
    """
    alerts = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if data is None or len(data) < 2:
                alerts.append({"ticker": ticker, "status": "no_data",
                                "change_pct": 0, "price": 0, "prev": 0})
                continue
            price      = float(data["Close"].iloc[-1])
            prev_price = float(data["Close"].iloc[-2])
            change_pct = ((price - prev_price) / prev_price) * 100
            if change_pct <= -threshold_pct:
                status = "crash"
            elif change_pct <= -(threshold_pct / 2):
                status = "warning"
            else:
                status = "ok"
            alerts.append({
                "ticker":     ticker,
                "status":     status,
                "change_pct": change_pct,
                "price":      price,
                "prev":       prev_price
            })
        except Exception as e:
            alerts.append({"ticker": ticker, "status": "error",
                           "change_pct": 0, "price": 0, "prev": 0,
                           "error": str(e)})
    return alerts


def show_alerts_page(companies):
    """Render the full Alerts & Monitor page."""
    st.header("🔔 Stock Alert Monitor")
    st.markdown("Set thresholds and get instant crash / loss alerts for selected stocks.")
    st.markdown("---")

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        st.subheader("⚙️ Alert Settings")

        watched = st.multiselect(
            "Stocks to monitor:",
            options=list(companies.keys()),
            default=["AAPL", "MSFT", "TSLA"],
            format_func=lambda x: f"{x} — {companies[x]}"
        )

        threshold = st.slider(
            "Alert threshold (% drop):",
            min_value=0.5, max_value=10.0, value=3.0, step=0.5,
            help="Trigger a CRASH alert if a stock drops more than this % from yesterday."
        )

        st.markdown(f"""
        <div style='font-size:0.85rem; color:#555; background:#f8f9fa;
                    border-radius:8px; padding:10px; margin-top:8px;'>
        🔴 <b>Crash</b>: drop &gt; {threshold:.1f}%<br>
        🟡 <b>Warning</b>: drop &gt; {threshold/2:.1f}%<br>
        🟢 <b>OK</b>: within normal range
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        check_btn = st.button("🔍 Check Now", type="primary",
                              use_container_width=True)

        auto_refresh = st.checkbox("🔄 Auto-refresh every 60 seconds")

        if auto_refresh:
            st.info("Auto-refresh enabled — page will reload every 60s.")
            time.sleep(60)
            st.rerun()

    with col_res:
        st.subheader("📊 Alert Results")

        if check_btn or auto_refresh:
            if not watched:
                st.warning("Select at least one stock to monitor.")
            else:
                with st.spinner("Fetching live prices..."):
                    results_alert = check_stock_alerts(watched, threshold)

                # Summary counts
                n_crash   = sum(1 for a in results_alert if a["status"] == "crash")
                n_warn    = sum(1 for a in results_alert if a["status"] == "warning")
                n_ok      = sum(1 for a in results_alert if a["status"] == "ok")

                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 Crash Alerts", n_crash)
                m2.metric("🟡 Warnings",     n_warn)
                m3.metric("🟢 Normal",        n_ok)
                st.markdown("---")

                # Per-stock cards
                for a in results_alert:
                    arrow  = "▲" if a["change_pct"] >= 0 else "▼"
                    color  = "#27ae60" if a["change_pct"] >= 0 else "#e74c3c"
                    if a["status"] == "crash":
                        css_cls = "alert-danger"
                        icon    = "🚨"
                        label   = "CRASH ALERT"
                    elif a["status"] == "warning":
                        css_cls = "alert-warning"
                        icon    = "⚠️"
                        label   = "WARNING"
                    elif a["status"] == "ok":
                        css_cls = "alert-ok"
                        icon    = "✅"
                        label   = "NORMAL"
                    else:
                        css_cls = "alert-warning"
                        icon    = "❓"
                        label   = "NO DATA"

                    st.markdown(f"""
                    <div class="{css_cls}">
                        <b>{icon} {a["ticker"]} — {companies.get(a["ticker"], a["ticker"])}</b>
                        &nbsp;&nbsp;<span style="font-size:0.8rem; color:#888;">{label}</span><br>
                        Current: <b>${a["price"]:.2f}</b>
                        &nbsp;|&nbsp; Prev close: ${a["prev"]:.2f}
                        &nbsp;|&nbsp;
                        <span style="color:{color}; font-weight:bold;">
                            {arrow} {a["change_pct"]:+.2f}%
                        </span>
                    </div>""", unsafe_allow_html=True)

                st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("👆 Click **Check Now** to scan for alerts, or enable auto-refresh.")

            # Show example alert cards
            st.markdown("**Preview — what alerts look like:**")
            st.markdown("""
            <div class="alert-danger">
                🚨 <b>TSLA — Tesla Inc.</b> &nbsp; <span style="font-size:0.8rem; color:#888;">CRASH ALERT</span><br>
                Current: $210.50 &nbsp;|&nbsp; Prev close: $225.30 &nbsp;|&nbsp;
                <span style="color:#e74c3c; font-weight:bold;">▼ -6.57%</span>
            </div>
            <div class="alert-warning">
                ⚠️ <b>NVDA — NVIDIA Corporation</b> &nbsp; <span style="font-size:0.8rem; color:#888;">WARNING</span><br>
                Current: $820.10 &nbsp;|&nbsp; Prev close: $843.90 &nbsp;|&nbsp;
                <span style="color:#e74c3c; font-weight:bold;">▼ -2.82%</span>
            </div>
            <div class="alert-ok">
                ✅ <b>AAPL — Apple Inc.</b> &nbsp; <span style="font-size:0.8rem; color:#888;">NORMAL</span><br>
                Current: $187.20 &nbsp;|&nbsp; Prev close: $186.50 &nbsp;|&nbsp;
                <span style="color:#27ae60; font-weight:bold;">▲ +0.38%</span>
            </div>
            """, unsafe_allow_html=True)


def show_chatbot_page():
    """Render the full Chatbot page."""
    st.header("🤖 StockBot — AI Assistant")
    st.markdown("Ask me anything about stocks, predictions, ARIMA, LSTM, or market alerts.")
    st.markdown("---")

    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    col_chat, col_quick = st.columns([3, 1])

    with col_quick:
        st.subheader("⚡ Quick Questions")
        quick_qs = [
            "Price of AAPL",
            "Price of TSLA",
            "Is there a crash today?",
            "What is ARIMA?",
            "How does LSTM work?",
            "How to set an alert?",
            "Predict MSFT",
        ]
        for q in quick_qs:
            if st.button(q, use_container_width=True, key=f"quick_{q}"):
                st.session_state["chat_history"].append(
                    {"role": "user", "content": q})
                reply = chatbot_response(q)
                st.session_state["chat_history"].append(
                    {"role": "bot", "content": reply})
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    with col_chat:
        # Chat display
        chat_html = '<div class="chat-container" id="chat-box">'
        if not st.session_state["chat_history"]:
            chat_html += '<div style="color:#aaa; text-align:center; margin-top:180px;">'
            chat_html += 'Start a conversation below ↓</div>'
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                chat_html += f'<div class="chat-user">{msg["content"]}</div>'
            else:
                # Convert **bold** markdown to <b> for HTML
                content = msg["content"]
                import re
                content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
                content = content.replace("\n", "<br>")
                chat_html += f'<div class="chat-bot">{content}</div>'
        chat_html += '</div>'

        # Auto-scroll to bottom
        chat_html += '<script>var cb=document.getElementById("chat-box");if(cb)cb.scrollTop=cb.scrollHeight;</script>'
        st.markdown(chat_html, unsafe_allow_html=True)

        # Input row
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_input = st.text_input(
                "Message", label_visibility="collapsed",
                placeholder="Ask StockBot anything...",
                key="chat_input"
            )
        with btn_col:
            send = st.button("Send", type="primary", use_container_width=True)

        if send and user_input.strip():
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input.strip()})
            reply = chatbot_response(user_input.strip())
            st.session_state["chat_history"].append(
                {"role": "bot", "content": reply})
            st.rerun()

# Company information
COMPANIES = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.'
}

@st.cache_data
def load_results():
    """Load model results from JSON file"""
    results_file = 'results/model_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_stock_data(ticker, data_type='train'):
    """Load stock data from CSV"""
    if data_type == 'train':
        file_path = f'data/processed/{ticker}_train.csv'
    else:
        file_path = f'data/processed/{ticker}_test.csv'
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    return None

def download_latest_data(ticker, days_back=1000):
    """Download latest stock data for real-time prediction"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        # Make timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        return df
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def predict_next_day(ticker, target_date=None):
    """
    Predict the closing price for `target_date` (or the next trading day
    after `target_date`) using ARIMA on log returns.

    Parameters
    ----------
    ticker      : stock ticker symbol
    target_date : datetime.date — the date whose close price we want to predict.
                  The model trains on all data BEFORE this date, then forecasts
                  this date's close.  If None, uses today (latest available data).

    Key idea
    --------
    Raw stock prices are non-stationary — ARIMA fitted directly on them
    will nearly always select ARIMA(0,1,0), which is a pure random walk.
    Instead we fit ARIMA on log returns (already stationary, d=0) and
    convert the predicted return back to a price.
    """
    from datetime import date as date_type

    with st.spinner(f'📊 Downloading data for {ticker}...'):
        df = download_latest_data(ticker, days_back=1500)

    if df is None or len(df) < 100:
        st.error("Insufficient data for prediction")
        return None

    try:
        # ── Slice data up to (but NOT including) the target date ─────────────
        if target_date is not None:
            # Convert date → pandas Timestamp for comparison
            cutoff = pd.Timestamp(target_date)
            df_train = df[df.index < cutoff]
            if len(df_train) < 60:
                st.error(f"Not enough historical data before {target_date}. Choose a later date.")
                return None
        else:
            df_train = df

        close_prices = df_train['Close'].values
        last_date    = df_train.index[-1]
        last_price   = float(close_prices[-1])

        # ── Step 1: compute log returns (stationary series) ─────────────────
        log_returns = np.diff(np.log(close_prices))   # length = N-1

        # ── Step 2: find best ARIMA(p,0,q) on returns ───────────────────────
        with st.spinner('🔍 Finding optimal ARIMA parameters on log returns...'):
            best_aic   = np.inf
            best_order = None

            # d=0 because log returns are already stationary
            p_range = range(0, 6)
            q_range = range(0, 6)

            progress_bar  = st.progress(0)
            total_combos  = len(p_range) * len(q_range)
            current       = 0

            for p in p_range:
                for q in q_range:
                    if p == 0 and q == 0:
                        current += 1
                        progress_bar.progress(current / total_combos)
                        continue          # ARIMA(0,0,0) = white noise, skip
                    try:
                        mdl    = ARIMA(log_returns, order=(p, 0, q))
                        fitted = mdl.fit()
                        if fitted.aic < best_aic:
                            best_aic   = fitted.aic
                            best_order = (p, 0, q)
                    except Exception:
                        pass
                    current += 1
                    progress_bar.progress(current / total_combos)

            progress_bar.empty()

        if best_order is None:
            best_order = (1, 0, 1)   # safe fallback

        # ── Step 3: train final model & forecast next return ─────────────────
        with st.spinner(f'🤖 Training ARIMA{best_order} on log returns...'):
            final_model  = ARIMA(log_returns, order=best_order)
            fitted_model = final_model.fit()

        with st.spinner('🎯 Generating prediction...'):
            forecast_obj = fitted_model.get_forecast(steps=1)

            # predicted_mean can be pandas Series OR numpy array depending on statsmodels version
            pm = forecast_obj.predicted_mean
            pred_return = float(pm.iloc[0] if hasattr(pm, 'iloc') else float(pm.flat[0]))

            # conf_int can be DataFrame OR numpy array
            ci = forecast_obj.conf_int(alpha=0.05)
            if hasattr(ci, 'iloc'):
                pred_return_lo = float(ci.iloc[0, 0])
                pred_return_hi = float(ci.iloc[0, 1])
            else:
                ci_arr = ci.flatten() if hasattr(ci, 'flatten') else ci
                pred_return_lo = float(ci_arr[0])
                pred_return_hi = float(ci_arr[1])

        # ── Step 4: convert log returns → price ──────────────────────────────
        predicted_price = last_price * np.exp(pred_return)
        conf_lower      = last_price * np.exp(pred_return_lo)
        conf_upper      = last_price * np.exp(pred_return_hi)

        # Sanity-clamp: confidence bounds must straddle the prediction
        conf_lower = min(conf_lower, predicted_price)
        conf_upper = max(conf_upper, predicted_price)

        # Determine the actual prediction date:
        # if user passed target_date use it, else it's next trading day after last_date
        if target_date is not None:
            from datetime import date as _d
            pred_date = pd.Timestamp(target_date)
        else:
            pred_date = last_date + timedelta(days=1)
            while pred_date.weekday() >= 5:
                pred_date += timedelta(days=1)

        result = {
            'ticker':           ticker,
            'last_date':        last_date,
            'last_price':       last_price,
            'predicted_price':  float(predicted_price),
            'confidence_lower': float(conf_lower),
            'confidence_upper': float(conf_upper),
            'change':           float(predicted_price - last_price),
            'change_pct':       float(((predicted_price - last_price) / last_price) * 100),
            'arima_order':      best_order,
            'aic':              float(best_aic),
            'historical_data':  df_train,   # only show training data in chart
            'pred_date':        pred_date    # exact date being predicted
        }

        return result

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None


def predict_next_day_lstm(ticker, target_date=None):
    """
    Real-time prediction using LSTM on log returns.
    Mirrors predict_next_day() but uses the trained LSTMForecaster
    in returns-mode for scale-invariant predictions.
    """
    from datetime import date as date_type

    with st.spinner(f'📊 Downloading data for {ticker}...'):
        df = download_latest_data(ticker, days_back=1500)

    if df is None or len(df) < 100:
        st.error("Insufficient data for prediction")
        return None

    try:
        if target_date is not None:
            cutoff   = pd.Timestamp(target_date)
            df_train = df[df.index < cutoff]
            if len(df_train) < 100:
                st.error(f"Not enough historical data before {target_date}.")
                return None
        else:
            df_train = df

        close_prices = df_train['Close'].values.astype(float)
        last_date    = df_train.index[-1]
        last_price   = float(close_prices[-1])

        cfg         = LSTM_STOCK_CONFIGS.get(ticker, {})
        seq_len     = cfg.get('sequence_length', 60)
        use_returns = cfg.get('use_returns', False)

        with st.spinner(f'🤖 Training LSTM for {ticker} (this takes ~20–40 seconds)...'):
            lstm = LSTMForecaster(ticker=ticker)
            # Build a temporary DataFrame so train() works correctly
            train_df_temp = pd.DataFrame({'Close': close_prices},
                                         index=df_train.index)
            lstm.train(train_df_temp, verbose=0)

        with st.spinner('🎯 Generating LSTM prediction...'):
            if use_returns:
                # Returns mode: predict next log return → convert to price
                log_returns  = np.diff(np.log(close_prices))
                scaled_all   = lstm.scaler.transform(log_returns.reshape(-1, 1)).flatten()
                window       = scaled_all[-seq_len:]
                if len(window) < seq_len:
                    window = np.pad(window, (seq_len - len(window), 0), mode='edge')
                pred_scaled  = lstm.model.predict(
                    window.reshape(1, seq_len, 1), verbose=0)[0, 0]
                pred_return  = float(lstm.scaler.inverse_transform([[pred_scaled]])[0, 0])
                predicted_price = last_price * np.exp(pred_return)
                # 95% CI: ±1.96 * rolling return std
                ret_std      = float(np.std(log_returns[-60:]))
                conf_lower   = last_price * np.exp(pred_return - 1.96 * ret_std)
                conf_upper   = last_price * np.exp(pred_return + 1.96 * ret_std)
            else:
                # Price mode: predict directly
                scaled_prices = lstm.scaler.transform(close_prices.reshape(-1, 1)).flatten()
                window        = scaled_prices[-seq_len:]
                if len(window) < seq_len:
                    window = np.pad(window, (seq_len - len(window), 0), mode='edge')
                pred_scaled   = lstm.model.predict(
                    window.reshape(1, seq_len, 1), verbose=0)[0, 0]
                predicted_price = float(
                    lstm.scaler.inverse_transform([[pred_scaled]])[0, 0])
                price_std     = float(np.std(close_prices[-60:]))
                conf_lower    = predicted_price - 1.96 * price_std * 0.02
                conf_upper    = predicted_price + 1.96 * price_std * 0.02

        conf_lower = min(conf_lower, predicted_price)
        conf_upper = max(conf_upper, predicted_price)

        if target_date is not None:
            pred_date = pd.Timestamp(target_date)
        else:
            pred_date = last_date + timedelta(days=1)
            while pred_date.weekday() >= 5:
                pred_date += timedelta(days=1)

        return {
            'ticker':           ticker,
            'last_date':        last_date,
            'last_price':       last_price,
            'predicted_price':  float(predicted_price),
            'confidence_lower': float(conf_lower),
            'confidence_upper': float(conf_upper),
            'change':           float(predicted_price - last_price),
            'change_pct':       float(((predicted_price - last_price) / last_price) * 100),
            'model_name':       'LSTM',
            'historical_data':  df_train,
            'pred_date':        pred_date,
        }

    except Exception as e:
        st.error(f"LSTM prediction error: {str(e)}")
        return None


def plot_realtime_prediction(result):
    """Create prediction visualization — history + next-day prediction with CI."""

    df          = result['historical_data']
    ticker      = result['ticker']

    # Last 20 trading days of history
    recent_data = df.tail(20).copy()

    last_date  = result['last_date']
    pred_price = result['predicted_price']
    ci_lo      = result['confidence_lower']
    ci_hi      = result['confidence_upper']
    change_pct = result['change_pct']

    # Use the exact prediction date stored in result (set by predict_next_day)
    next_date = result.get('pred_date', last_date + timedelta(days=1))
    if hasattr(next_date, 'to_pydatetime'):
        next_date = next_date.to_pydatetime()

    # ── Build unified x/y arrays including the prediction point ──────────────
    hist_x = list(recent_data.index)
    hist_y = list(recent_data['Close'].values)

    # All x values that will appear in the chart
    all_x = hist_x + [next_date]
    all_y = hist_y + [pred_price, ci_lo, ci_hi]

    # y-axis range with padding
    y_pad  = (max(all_y) - min(all_y)) * 0.15
    y_min  = min(all_y) - y_pad
    y_max  = max(all_y) + y_pad

    # ── Colours ───────────────────────────────────────────────────────────────
    label_color = '#27AE60' if change_pct >= 0 else '#E74C3C'
    arrow_sym   = '▲' if change_pct >= 0 else '▼'

    fig = go.Figure()

    # 1. Historical price line
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_y,
        mode='lines',
        name='Historical Price',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='<b>%{x|%b %d}</b><br>$%{y:.2f}<extra>Historical</extra>'
    ))

    # 2. Dashed connector: last close → prediction (only 2 points)
    fig.add_trace(go.Scatter(
        x=[last_date, next_date],
        y=[result['last_price'], pred_price],
        mode='lines+markers',
        name='Predicted Next Close',
        line=dict(color='#A23B72', width=2, dash='dash'),
        marker=dict(size=[5, 12], symbol=['circle', 'star'], color='#A23B72'),
        hovertemplate='<b>%{x|%b %d}</b><br>$%{y:.2f}<extra>Prediction</extra>'
    ))

    # 3. CI as two horizontal cap lines + a vertical connector — pure Scatter,
    #    no error_y (which forces Plotly to expand the axis)
    cap_half = timedelta(hours=4)   # width of the horizontal caps
    ci_x_cap = [next_date - cap_half, next_date + cap_half]

    # top cap
    fig.add_trace(go.Scatter(
        x=ci_x_cap, y=[ci_hi, ci_hi],
        mode='lines', line=dict(color='#F18F01', width=2),
        showlegend=False, hoverinfo='skip'
    ))
    # vertical bar
    fig.add_trace(go.Scatter(
        x=[next_date, next_date], y=[ci_lo, ci_hi],
        mode='lines', line=dict(color='#F18F01', width=2),
        name='95% Confidence Interval', hoverinfo='skip'
    ))
    # bottom cap
    fig.add_trace(go.Scatter(
        x=ci_x_cap, y=[ci_lo, ci_lo],
        mode='lines', line=dict(color='#F18F01', width=2),
        showlegend=False, hoverinfo='skip'
    ))

    # 4. Price annotation above the CI top
    fig.add_annotation(
        x=next_date,
        y=ci_hi,
        text=f'{arrow_sym} ${pred_price:.2f} ({change_pct:+.2f}%)',
        showarrow=False,
        font=dict(size=12, color=label_color, family='Arial Black'),
        xanchor='center',
        yanchor='bottom',
        yshift=6
    )

    # ── Layout — hard clamp both axes, disable autorange ─────────────────────
    # Add a small right-side buffer (4 hours) so the CI caps aren't clipped
    x_end_str   = (next_date + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')
    x_start_str = hist_x[0].strftime('%Y-%m-%d')

    fig.update_layout(
        title=dict(
            text=f'{ticker} — Real-Time Price Prediction (ARIMA on Log Returns)',
            font=dict(size=15)
        ),
        xaxis=dict(
            title='Date',
            range=[x_start_str, x_end_str],
            autorange=False,
            type='date',
            showgrid=True, gridcolor='#efefef'
        ),
        yaxis=dict(
            title='Stock Price ($)',
            range=[y_min, y_max],
            autorange=False,
            showgrid=True, gridcolor='#efefef'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=480,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )

    return fig

def plot_predictions(ticker, results_data):
    """Create interactive plotly chart for predictions"""
    test_data = load_stock_data(ticker, 'test')
    
    if test_data is None:
        st.error("Test data not found!")
        return
    
    y_true = results_data['predictions']['y_true']
    y_pred = results_data['predictions']['y_pred']
    dates = test_data.index[:len(y_true)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='lines',
        name='Actual Price',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Actual</b>: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode='lines',
        name='Predicted Price',
        line=dict(color='#A23B72', width=2, dash='dash'),
        hovertemplate='<b>Date</b>: %{x}<br><b>Predicted</b>: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{COMPANIES[ticker]} - ARIMA Price Predictions',
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_error_distribution(ticker, results_data):
    """Plot error distribution"""
    y_true = np.array(results_data['predictions']['y_true'])
    y_pred = np.array(results_data['predictions']['y_pred'])
    errors = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        name='Prediction Errors',
        marker_color='#F18F01',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Prediction Error Distribution',
        xaxis_title='Prediction Error ($)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_all_models_comparison(results):
    """Compare all models across all stocks"""
    comparison_data = []

    for key, value in results.items():
        comparison_data.append({
            'Stock': value['ticker'],
            'Company': COMPANIES.get(value['ticker'], value['ticker']),
            'Model': value['model'],
            'RMSE': value['metrics']['RMSE'],
            'MAE': value['metrics']['MAE'],
            'MAPE': value['metrics']['MAPE'],
            'R²': value['metrics']['R2']
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values(['Stock', 'Model'])

    fig = px.bar(
        df,
        x='Company',
        y='RMSE',
        color='Model',
        barmode='group',
        title='RMSE Comparison Across All Stocks — ARIMA vs LSTM',
        labels={'RMSE': 'Root Mean Squared Error ($)', 'Company': 'Company'},
        color_discrete_map={'ARIMA': '#2E86AB', 'LSTM': '#A23B72'},
        template='plotly_white',
        height=520
    )

    fig.update_layout(xaxis_tickangle=-40, legend_title_text='Model')

    return fig, df

def main():
    # ── Login gate ───────────────────────────────────────────────────────────
    if not st.session_state.get("logged_in", False):
        show_login_page()
        return

    # ── Logged-in header ─────────────────────────────────────────────────────
    user_display = st.session_state.get("display_name", "User")
    role         = st.session_state.get("role", "Viewer")

    st.title("📈 Stock Price Forecasting Dashboard")
    st.markdown(f"### Time-Series Analysis using ARIMA & LSTM")

    # User info + logout in top-right
    hdr_col, logout_col = st.columns([6, 1])
    with logout_col:
        st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#888;'>"
                    f"👤 {user_display}<br><i>{role}</i></div>",
                    unsafe_allow_html=True)
        if st.button("Logout", key="logout_btn"):
            for k in ["logged_in","username","display_name","role","chat_history",
                      "prediction_result","predicted_date"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown("---")

    # Load results
    results = load_results()

    if not results:
        st.error("⚠️ No results found! Please run the model training first.")
        st.stop()

    # Sidebar navigation
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown(f"**👤 {user_display}** ({role})")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select View",
        ["📊 Individual Stock Analysis",
         "🔮 Real-Time Prediction",
         "📈 All Stocks Comparison",
         "🔔 Alerts",
         "🤖 StockBot",
         "ℹ️ About"]
    )

    # ── Pages ────────────────────────────────────────────────────────────────

    if page == "📊 Individual Stock Analysis":
        selected_ticker = st.sidebar.selectbox(
            "Select Stock",
            options=list(COMPANIES.keys()),
            format_func=lambda x: f"{x} - {COMPANIES[x]}"
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📌 Stock Information")
        st.sidebar.info(f"**Ticker:** {selected_ticker}\n\n**Company:** {COMPANIES[selected_ticker]}")

        arima_key = f"{selected_ticker}_ARIMA"
        lstm_key  = f"{selected_ticker}_LSTM"

        if arima_key not in results:
            st.error(f"No ARIMA results found for {selected_ticker}. Run the model pipeline first.")
            st.stop()

        arima_results = results[arima_key]
        lstm_results  = results.get(lstm_key)

        st.header(f"{COMPANIES[selected_ticker]} ({selected_ticker})")
        st.markdown("---")

        available_models = ["ARIMA"]
        if lstm_results:
            available_models.append("LSTM")

        selected_model = st.radio("Select model to display:", available_models, horizontal=True)
        stock_results  = arima_results if selected_model == "ARIMA" else lstm_results
        metrics        = stock_results["metrics"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📉 RMSE",     f"${metrics['RMSE']:.2f}",  help="Lower is better")
        with col2:
            st.metric("📊 MAE",      f"${metrics['MAE']:.2f}",   help="Lower is better")
        with col3:
            st.metric("📈 MAPE",     f"{metrics['MAPE']:.2f}%",  help="Lower is better")
        with col4:
            st.metric("🎯 R² Score", f"{metrics['R2']:.4f}",     help="Closer to 1 is better")

        if lstm_results:
            st.markdown("---")
            st.subheader("⚡ ARIMA vs LSTM — Side-by-Side")
            am, lm = arima_results["metrics"], lstm_results["metrics"]
            cmp_df = pd.DataFrame({
                "Metric": ["RMSE ($)", "MAE ($)", "MAPE (%)", "R²"],
                "ARIMA":  [f"{am['RMSE']:.2f}", f"{am['MAE']:.2f}", f"{am['MAPE']:.2f}", f"{am['R2']:.4f}"],
                "LSTM":   [f"{lm['RMSE']:.2f}", f"{lm['MAE']:.2f}", f"{lm['MAPE']:.2f}", f"{lm['R2']:.4f}"],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader(f"📊 {selected_model} Price Predictions")
        fig_predictions = plot_predictions(selected_ticker, stock_results)
        st.plotly_chart(fig_predictions, use_container_width=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📉 Error Distribution")
            fig_errors = plot_error_distribution(selected_ticker, stock_results)
            st.plotly_chart(fig_errors, use_container_width=True)
        with col2:
            st.subheader("📋 Model Details")
            train_time = stock_results.get("training_time")
            train_str  = f"{train_time:.2f}s" if train_time else "N/A"
            st.markdown(f"""
            **Model:** {selected_model}
            **Training Time:** {train_str}
            **Test Samples:** {len(stock_results['predictions']['y_true'])}

            ---

            **Performance:**
            - {'✅' if metrics['MAPE'] < 5 else '⚠️' if metrics['MAPE'] < 10 else '❌'} MAPE: {metrics['MAPE']:.2f}%
            - {'✅' if metrics['R2'] > 0.8 else '⚠️' if metrics['R2'] > 0.6 else '❌'} R²: {metrics['R2']:.4f}
            """)

        st.markdown("---")
        st.subheader("💾 Download Predictions")
        y_true = stock_results["predictions"]["y_true"]
        y_pred = stock_results["predictions"]["y_pred"]
        test_data = load_stock_data(selected_ticker, "test")
        if test_data is not None:
            y_true_flat = np.array(y_true).flatten().tolist()
            y_pred_flat = np.array(y_pred).flatten().tolist()
            download_df = pd.DataFrame({
                "Date":                      test_data.index[:len(y_true_flat)].tolist(),
                "Actual_Price":              y_true_flat,
                f"{selected_model}_Predicted": y_pred_flat,
                "Error":                     [a - p for a, p in zip(y_true_flat, y_pred_flat)]
            })
            st.download_button(
                label=f"📥 Download {selected_model} Predictions as CSV",
                data=download_df.to_csv(index=False),
                file_name=f"{selected_ticker}_{selected_model}_predictions.csv",
                mime="text/csv"
            )

    elif page == "🔮 Real-Time Prediction":
        st.header("🔮 Real-Time Stock Price Prediction")
        st.markdown("Select a stock, a model, and a target date — the model trains on all data before that date and predicts the closing price for that day.")
        st.markdown("---")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 Settings")
            predict_ticker = st.selectbox(
                "Choose a stock:",
                options=list(COMPANIES.keys()),
                format_func=lambda x: f"{x} - {COMPANIES[x]}",
                key="predict_ticker"
            )
            st.markdown("---")

            # ── Model selector ────────────────────────────────────────────
            model_options = ["ARIMA"]
            if LSTM_AVAILABLE:
                model_options.append("LSTM")
            selected_model_rt = st.radio(
                "🤖 Choose prediction model:",
                options=model_options,
                horizontal=True,
                help=("ARIMA: fast, statistical. "
                      "LSTM: neural network, ~30s training, better for volatile stocks.")
            )
            if not LSTM_AVAILABLE and len(model_options) == 1:
                st.caption("⚠️ LSTM unavailable — install TensorFlow to enable it.")

            st.markdown("---")
            from datetime import date as _date
            today         = _date.today()
            selected_date = st.date_input(
                "📅 Select prediction date:",
                value=today,
                min_value=_date(2022, 1, 1),
                max_value=today,
                help="Model trains on all data BEFORE this date and predicts this date's close."
            )
            if selected_date.weekday() >= 5:
                st.warning("⚠️ Selected date is a weekend. Stock markets are closed.")
            st.markdown("---")

            if selected_model_rt == "LSTM":
                st.info("⏱ LSTM trains a fresh neural network each prediction (~20–40s).")

            if st.button("🎯 Predict Closing Price", type="primary", use_container_width=True):
                if selected_date.weekday() >= 5:
                    st.error("Please select a weekday (Mon–Fri).")
                else:
                    if selected_model_rt == "LSTM":
                        st.session_state["prediction_result"] = predict_next_day_lstm(
                            predict_ticker, target_date=selected_date)
                    else:
                        st.session_state["prediction_result"] = predict_next_day(
                            predict_ticker, target_date=selected_date)
                    st.session_state["predicted_date"]  = selected_date
                    st.session_state["predicted_model"] = selected_model_rt

        with col2:
            if st.session_state.get("prediction_result"):
                result        = st.session_state["prediction_result"]
                pred_date     = st.session_state.get("predicted_date", "N/A")
                used_model    = st.session_state.get("predicted_model", "ARIMA")
                pred_date_str = pred_date.strftime("%Y-%m-%d") if hasattr(pred_date, "strftime") else str(pred_date)
                change_color  = "#27AE60" if result["change"] >= 0 else "#E74C3C"
                arrow         = "▲" if result["change"] >= 0 else "▼"

                # Model-specific info line
                if used_model == "ARIMA":
                    model_info = f"Model: ARIMA{result.get('arima_order','?')} | AIC: {result.get('aic', 0):.2f}"
                else:
                    cfg        = LSTM_STOCK_CONFIGS.get(result["ticker"], {})
                    mode_label = "returns mode" if cfg.get("use_returns") else "price mode"
                    model_info = f"Model: LSTM ({mode_label}) | Lookback: {cfg.get('sequence_length', 60)}d"

                st.markdown(f"""
                <div class="prediction-box">
                <h3>📊 {result["ticker"]} — {COMPANIES[result["ticker"]]}</h3>
                <p><strong>Model used:</strong> {used_model}</p>
                <p><strong>Training data up to:</strong> {result["last_date"].strftime("%Y-%m-%d")}</p>
                <p><strong>Last known close:</strong> ${result["last_price"]:.2f}</p>
                <p><strong>Predicting date:</strong> {pred_date_str}</p>
                <hr>
                <h2 style="color:#1f77b4;">Predicted Close: ${result["predicted_price"]:.2f}</h2>
                <p><strong>95% Confidence Interval:</strong> ${result["confidence_lower"]:.2f} – ${result["confidence_upper"]:.2f}</p>
                <p style="color:{change_color};"><strong>Expected Change:</strong> {arrow} ${abs(result["change"]):.2f} ({result["change_pct"]:+.2f}%)</p>
                <p><small>{model_info}</small></p>
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.get("prediction_result"):
            st.markdown("---")
            st.subheader("📈 Prediction Visualization")
            fig_realtime = plot_realtime_prediction(st.session_state["prediction_result"])
            st.plotly_chart(fig_realtime, use_container_width=True)
            st.markdown("---")
            st.info("💡 The model trains on all historical data before the selected date. Back-test on past dates or forecast for today.")

    elif page == "📈 All Stocks Comparison":
        st.header("All Stocks Performance Comparison")
        st.markdown("---")
        fig_comparison, df_comparison = plot_all_models_comparison(results)
        st.plotly_chart(fig_comparison, use_container_width=True)
        st.subheader("📊 Detailed Metrics Table")
        display_df = df_comparison.copy()
        display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"${x:.2f}")
        display_df["MAE"]  = display_df["MAE"].apply(lambda x: f"${x:.2f}")
        display_df["MAPE"] = display_df["MAPE"].apply(lambda x: f"{x:.2f}%")
        display_df["R²"]   = display_df["R²"].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        st.subheader("🏆 Best Performing Stock")
        raw_df   = df_comparison.copy()
        best_row = raw_df.loc[raw_df["RMSE"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Company:** {best_row['Company']}")
        c2.success(f"**RMSE:** ${best_row['RMSE']:.2f}")
        c3.success(f"**R²:** {best_row['R²']:.4f}")

    elif page == "🔔 Alerts":
        show_alerts_page(COMPANIES)

    elif page == "🤖 StockBot":
        show_chatbot_page()

    else:  # About
        st.header("ℹ️ About This Project")
        st.markdown("---")
        st.markdown("""
        ### 📚 Stock Price Trend Forecasting Using Time-Series Models

        **Final Year Major Project** — Dr. A.P.J. Abdul Kalam Technical University

        #### 🎯 Objectives
        - Historical stock data for 10 major companies
        - ARIMA + LSTM models with comparative analysis
        - Real-time prediction with date selection
        - Stock crash alert monitoring
        - AI chatbot assistant (StockBot)

        #### 🤖 Models
        - ✅ ARIMA (on log returns)
        - ✅ LSTM (60-day lookback, 128→64→32)
        - ✅ Real-Time Prediction with back-testing
        - 🔄 GRU / Prophet (coming soon)

        #### 🛠️ Technologies
        Python 3 · Streamlit · Plotly · TensorFlow · Statsmodels · yfinance

        ---
        **👤 Ayush Singh** | ayushsingh1562004@gmail.com
        """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center'>
        <p style='font-size: 12px; color: #666;'>
        © 2026 StockCast Project<br>
        Built with ❤️ using Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()