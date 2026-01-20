import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class StockDataLoader:
    """
    A class to download and manage stock data from Yahoo Finance
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 10 companies for the project
        self.companies = {
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
    
    def download_stock_data(self, ticker, start_date=None, end_date=None, period='5y'):
        """
        Download stock data for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period if dates not specified (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with stock data
        """
        try:
            print(f"Downloading data for {ticker} - {self.companies.get(ticker, 'Unknown')}...")
            
            if start_date and end_date:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(ticker, period=period, progress=False)
            
            if data.empty:
                print(f"Warning: No data retrieved for {ticker}")
                return None
            
            # Flatten column names if they're multi-level
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Reset index to make Date a column, then set it back
            data = data.reset_index()
            
            # Save to CSV
            filename = os.path.join(self.data_dir, f"{ticker}.csv")
            data.to_csv(filename, index=False)
            print(f"✓ Data saved to {filename} ({len(data)} records)")
            
            return data
            
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
            return None
    
    def download_all_stocks(self, start_date=None, end_date=None, period='5y'):
        """
        Download data for all companies
        
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        all_data = {}
        
        print(f"\n{'='*60}")
        print(f"Starting download for {len(self.companies)} companies")
        print(f"{'='*60}\n")
        
        for ticker in self.companies.keys():
            data = self.download_stock_data(ticker, start_date, end_date, period)
            if data is not None:
                all_data[ticker] = data
        
        print(f"\n{'='*60}")
        print(f"Download complete! {len(all_data)}/{len(self.companies)} companies retrieved")
        print(f"{'='*60}\n")
        
        return all_data
    
    def load_stock_data(self, ticker):
        """
        Load previously downloaded stock data
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with stock data
        """
        filename = os.path.join(self.data_dir, f"{ticker}.csv")
        
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None
        
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        return data
    
    def get_stock_info(self, ticker):
        """
        Get basic information about a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': self.companies.get(ticker, info.get('longName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0)
            }
        except Exception as e:
            print(f"Error getting info for {ticker}: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    loader = StockDataLoader()
    
    # Download all stocks (last 5 years)
    all_data = loader.download_all_stocks(period='5y')
    
    # Display sample data for first stock
    if all_data:
        first_ticker = list(all_data.keys())[0]
        print(f"\nSample data for {first_ticker}:")
        print(all_data[first_ticker].head())
        print(f"\nShape: {all_data[first_ticker].shape}")
        print(f"\nColumns: {all_data[first_ticker].columns.tolist()}")