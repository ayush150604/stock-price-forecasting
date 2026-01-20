"""
Stock Price Forecasting Dashboard
Interactive Streamlit App for Model Comparison and Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

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
    
    # Actual prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='lines',
        name='Actual Price',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Actual</b>: $%{y:.2f}<extra></extra>'
    ))
    
    # Predicted prices
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
    
    # Create bar chart for RMSE comparison
    fig = px.bar(
        df,
        x='Company',
        y='RMSE',
        color='Model',
        title='RMSE Comparison Across All Stocks',
        labels={'RMSE': 'Root Mean Squared Error'},
        template='plotly_white',
        height=500
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig, df

def main():
    # Title and description
    st.title("📈 Stock Price Forecasting Dashboard")
    st.markdown("### Time-Series Analysis using ARIMA Model")
    st.markdown("---")
    
    # Load results
    results = load_results()
    
    if not results:
        st.error("⚠️ No results found! Please run `python main_simple.py` first to generate predictions.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Select View",
        ["📊 Individual Stock Analysis", "📈 All Stocks Comparison", "ℹ️ About"]
    )
    
    if page == "📊 Individual Stock Analysis":
        # Stock selector
        selected_ticker = st.sidebar.selectbox(
            "Select Stock",
            options=list(COMPANIES.keys()),
            format_func=lambda x: f"{x} - {COMPANIES[x]}"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📌 Stock Information")
        st.sidebar.info(f"**Ticker:** {selected_ticker}\n\n**Company:** {COMPANIES[selected_ticker]}")
        
        # Get results for selected stock
        result_key = f"{selected_ticker}_ARIMA"
        
        if result_key not in results:
            st.error(f"No results found for {selected_ticker}")
            st.stop()
        
        stock_results = results[result_key]
        metrics = stock_results['metrics']
        
        # Display company header
        st.header(f"{COMPANIES[selected_ticker]} ({selected_ticker})")
        st.markdown("---")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📉 RMSE",
                value=f"${metrics['RMSE']:.2f}",
                help="Root Mean Squared Error - Lower is better"
            )
        
        with col2:
            st.metric(
                label="📊 MAE",
                value=f"${metrics['MAE']:.2f}",
                help="Mean Absolute Error - Lower is better"
            )
        
        with col3:
            st.metric(
                label="📈 MAPE",
                value=f"{metrics['MAPE']:.2f}%",
                help="Mean Absolute Percentage Error - Lower is better"
            )
        
        with col4:
            st.metric(
                label="🎯 R² Score",
                value=f"{metrics['R2']:.4f}",
                help="R-squared - Closer to 1 is better"
            )
        
        st.markdown("---")
        
        # Predictions chart
        st.subheader("📊 Price Predictions")
        fig_predictions = plot_predictions(selected_ticker, stock_results)
        st.plotly_chart(fig_predictions, use_container_width=True)
        
        # Error distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📉 Error Distribution")
            fig_errors = plot_error_distribution(selected_ticker, stock_results)
            st.plotly_chart(fig_errors, use_container_width=True)
        
        with col2:
            st.subheader("📋 Model Details")
            st.markdown(f"""
            **Model:** ARIMA  
            **Training Time:** {stock_results.get('training_time', 'N/A'):.2f}s  
            **Test Samples:** {len(stock_results['predictions']['y_true'])}
            
            ---
            
            **Performance Summary:**
            - {'✅' if metrics['MAPE'] < 5 else '⚠️' if metrics['MAPE'] < 10 else '❌'} MAPE: {metrics['MAPE']:.2f}%
            - {'✅' if metrics['R2'] > 0.8 else '⚠️' if metrics['R2'] > 0.6 else '❌'} R²: {metrics['R2']:.4f}
            """)
        
        # Download predictions
        st.markdown("---")
        st.subheader("💾 Download Predictions")
        
        y_true = stock_results['predictions']['y_true']
        y_pred = stock_results['predictions']['y_pred']
        test_data = load_stock_data(selected_ticker, 'test')
        
        if test_data is not None:
            # Ensure y_true and y_pred are flat lists
            y_true_flat = np.array(y_true).flatten().tolist()
            y_pred_flat = np.array(y_pred).flatten().tolist()
            
            download_df = pd.DataFrame({
                'Date': test_data.index[:len(y_true_flat)].tolist(),
                'Actual_Price': y_true_flat,
                'Predicted_Price': y_pred_flat,
                'Error': [a - p for a, p in zip(y_true_flat, y_pred_flat)]
            })
            
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"{selected_ticker}_predictions.csv",
                mime="text/csv"
            )
    
    elif page == "📈 All Stocks Comparison":
        st.header("All Stocks Performance Comparison")
        st.markdown("---")
        
        # Create comparison chart
        fig_comparison, df_comparison = plot_all_models_comparison(results)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Display comparison table
        st.subheader("📊 Detailed Metrics Table")
        
        # Format the dataframe for better display
        display_df = df_comparison.copy()
        display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"${x:.2f}")
        display_df['MAE'] = display_df['MAE'].apply(lambda x: f"${x:.2f}")
        display_df['MAPE'] = display_df['MAPE'].apply(lambda x: f"{x:.2f}%")
        display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Best performing stock
        st.markdown("---")
        st.subheader("🏆 Best Performing Stock")
        
        best_stock = df_comparison.loc[df_comparison['RMSE'].idxmin()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Company:** {best_stock['Company']}")
        
        with col2:
            st.success(f"**RMSE:** ${best_stock['RMSE']:.2f}")
        
        with col3:
            st.success(f"**R² Score:** {best_stock['R²']:.4f}")
    
    else:  # About page
        st.header("ℹ️ About This Project")
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Stock Price Trend Forecasting Using Time-Series Models
        
        This is a **Final Year Major Project** focused on predicting stock prices using various time-series forecasting models.
        
        #### 🎯 Project Objectives
        - Collect historical stock data for 10 major companies
        - Apply multiple machine learning models
        - Compare model performance using statistical metrics
        - Visualize predictions through an interactive dashboard
        
        #### 📊 Companies Analyzed
        """)
        
        for ticker, name in COMPANIES.items():
            st.markdown(f"- **{ticker}**: {name}")
        
        st.markdown("""
        
        #### 🤖 Models Implemented
        - ✅ **ARIMA** (Autoregressive Integrated Moving Average)
        - 🔄 **LSTM** (Coming Soon)
        - 🔄 **GRU** (Coming Soon)
        - 🔄 **Prophet** (Coming Soon)
        
        #### 📈 Evaluation Metrics
        - **RMSE**: Root Mean Squared Error
        - **MAE**: Mean Absolute Error
        - **MAPE**: Mean Absolute Percentage Error
        - **R² Score**: Coefficient of Determination
        
        #### 🛠️ Technologies Used
        - Python
        - Streamlit
        - Plotly
        - Pandas, NumPy
        - Scikit-learn
        - Statsmodels (ARIMA)
        - yfinance (Data Collection)
        
        ---
        
        ### 👨‍💻 Developer Information
        **Project Type:** Final Year Major Project  
        **Domain:** Machine Learning & Time Series Forecasting  
        **Data Source:** Yahoo Finance (5 years historical data)
        
        ---
        
        *For more information or to report issues, please contact the project developer.*
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center'>
        <p style='font-size: 12px; color: #666;'>
        © 2025 Stock Forecasting Project<br>
        Built with ❤️ using Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()