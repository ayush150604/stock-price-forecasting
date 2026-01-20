import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class StockDataPreprocessor:
    """
    Preprocess stock data for time series forecasting
    """
    
    def __init__(self, processed_dir='data/processed'):
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        self.scaler = MinMaxScaler()
    
    def clean_data(self, df):
        """
        Clean the stock data
        
        Args:
            df: Raw stock DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy
        df_clean = df.copy()
        
        # Remove Ticker column if it exists
        if 'Ticker' in df_clean.columns:
            df_clean = df_clean.drop('Ticker', axis=1)
        
        # Handle missing values
        df_clean = df_clean.ffill().bfill()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        return df_clean
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators as features
        
        Args:
            df: Stock DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df_tech = df.copy()
        
        # Moving Averages
        df_tech['MA7'] = df_tech['Close'].rolling(window=7).mean()
        df_tech['MA21'] = df_tech['Close'].rolling(window=21).mean()
        df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df_tech['EMA12'] = df_tech['Close'].ewm(span=12, adjust=False).mean()
        df_tech['EMA26'] = df_tech['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df_tech['MACD'] = df_tech['EMA12'] - df_tech['EMA26']
        df_tech['Signal_Line'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df_tech['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_tech['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
        bb_std = df_tech['Close'].rolling(window=20).std()
        df_tech['BB_Upper'] = df_tech['BB_Middle'] + (bb_std * 2)
        df_tech['BB_Lower'] = df_tech['BB_Middle'] - (bb_std * 2)
        
        # Daily Returns
        df_tech['Daily_Return'] = df_tech['Close'].pct_change()
        
        # Volatility
        df_tech['Volatility'] = df_tech['Daily_Return'].rolling(window=21).std()
        
        # Drop NaN values created by indicators
        df_tech = df_tech.dropna()
        
        return df_tech
    
    def create_train_test_split(self, df, train_size=0.8):
        """
        Split data into train and test sets
        
        Args:
            df: Stock DataFrame
            train_size: Proportion of data for training (default: 0.8)
            
        Returns:
            train_df, test_df
        """
        split_index = int(len(df) * train_size)
        
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        
        print(f"Train set: {len(train_df)} records ({df.index[0]} to {train_df.index[-1]})")
        print(f"Test set: {len(test_df)} records ({test_df.index[0]} to {df.index[-1]})")
        
        return train_df, test_df
    
    def scale_data(self, train_df, test_df, columns_to_scale=['Close']):
        """
        Scale the data using MinMaxScaler
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            columns_to_scale: List of columns to scale
            
        Returns:
            scaled_train_df, scaled_test_df, scaler
        """
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        scaler = MinMaxScaler()
        
        # Fit on training data only
        train_scaled[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
        test_scaled[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
        
        return train_scaled, test_scaled, scaler
    
    def prepare_sequences(self, data, n_steps=60, target_col='Close'):
        """
        Prepare sequences for LSTM/GRU models
        
        Args:
            data: DataFrame or numpy array
            n_steps: Number of time steps to look back
            target_col: Target column name
            
        Returns:
            X, y arrays
        """
        if isinstance(data, pd.DataFrame):
            values = data[target_col].values
        else:
            values = data
        
        X, y = [], []
        
        for i in range(n_steps, len(values)):
            X.append(values[i-n_steps:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def preprocess_stock(self, df, ticker, train_size=0.8, add_indicators=True):
        """
        Complete preprocessing pipeline for a stock
        
        Args:
            df: Raw stock DataFrame
            ticker: Stock ticker
            train_size: Train/test split ratio
            add_indicators: Whether to add technical indicators
            
        Returns:
            Dictionary with preprocessed data
        """
        print(f"\nPreprocessing {ticker}...")
        
        # Clean data
        df_clean = self.clean_data(df)
        print(f"✓ Data cleaned ({len(df_clean)} records)")
        
        # Add technical indicators
        if add_indicators:
            df_clean = self.add_technical_indicators(df_clean)
            print(f"✓ Technical indicators added")
        
        # Train/test split
        train_df, test_df = self.create_train_test_split(df_clean, train_size)
        
        # Save processed data
        train_file = os.path.join(self.processed_dir, f"{ticker}_train.csv")
        test_file = os.path.join(self.processed_dir, f"{ticker}_test.csv")
        
        train_df.to_csv(train_file)
        test_df.to_csv(test_file)
        print(f"✓ Saved to {self.processed_dir}")
        
        return {
            'ticker': ticker,
            'train': train_df,
            'test': test_df,
            'full': df_clean
        }

if __name__ == "__main__":
    # Example usage
    from utils.data_loader import StockDataLoader
    
    loader = StockDataLoader()
    preprocessor = StockDataPreprocessor()
    
    # Load a stock
    data = loader.load_stock_data('AAPL')
    
    if data is not None:
        # Preprocess
        result = preprocessor.preprocess_stock(data, 'AAPL')
        
        print(f"\nPreprocessed data shape:")
        print(f"Training: {result['train'].shape}")
        print(f"Testing: {result['test'].shape}")
        print(f"\nFeatures: {result['train'].columns.tolist()}")