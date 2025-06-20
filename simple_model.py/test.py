import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import pickle
import warnings

# Suppress specific future warnings from pandas or other libraries
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Caching Configuration ---
CACHE_DIR = 'data_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- Helper Function for Display Precision ---
def get_display_precision(price_series):
    """
    Determines the appropriate decimal precision for display based on price magnitude.
    Handles pandas Series to find minimum non-zero price for robust precision.
    """
    if price_series.empty:
        return 2 # Default for empty series

    # Filter out zero or NaN prices before finding min
    non_zero_prices = price_series[price_series.notna() & (price_series != 0)]
    
    if non_zero_prices.empty:
        return 2 # Default if all are zero or NaN

    min_price = non_zero_prices.min()

    if min_price >= 100:
        return 2  # e.g., $100.00 -> 2 decimal places
    elif min_price >= 10:
        return 3  # e.g., $10.000 -> 3 decimal places
    elif min_price >= 1:
        return 4  # e.g., $1.0000 -> 4 decimal places
    elif min_price >= 0.1:
        return 5  # e.g., $0.10000 -> 5 decimal places
    elif min_price >= 0.01:
        return 6  # e.g., $0.010000 -> 6 decimal places
    else:
        # For very small prices, dynamically add more precision
        # log10(min_price) gives roughly the order of magnitude
        # -log10(min_price) gives approximate number of zeros after decimal
        return int(np.ceil(-np.log10(min_price))) + 2 # +2 for sig figs after zeros


def fetch_and_prepare_data(tickers, start_date, end_date, interval='1d', force_download=False):
    """
    Fetch historical data for cryptocurrencies/stocks and calculate basic features.
    Includes caching to avoid re-downloading data.

    Args:
        tickers (list): A list of stock or crypto ticker symbols.
        start_date (str): Start date for data fetching (YYYY-MM-DD).
        end_date (str): End date for data fetching (YYYY-MM-DD).
        interval (str): Data interval (e.g., '1d', '1h', '1m', '15m').
        force_download (bool): If True, force re-download data ignoring cache.

    Returns:
        pd.DataFrame: A concatenated DataFrame of historical data for all tickers
                      with basic features, or an empty DataFrame if no data is found.
    """
    # Sanitize dates for filename
    start_str = start_date.replace('-', '')
    end_str = end_date.replace('-', '')
    
    # Generate a unique filename for the cache
    cache_filename = os.path.join(CACHE_DIR, f"{'_'.join(sorted(tickers))}_{start_str}_to_{end_str}_{interval}.pkl")

    # --- Caching Logic ---
    if not force_download and os.path.exists(cache_filename):
        print(f"Loading data from cache: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            all_data = pickle.load(f)
        return all_data

    print("Fetching data from API...")
    all_data = pd.DataFrame()
    
    for ticker in tickers:
        print(f"  -> Fetching {ticker} from {start_date} to {end_date} with interval {interval}...")
        crypto = yf.Ticker(ticker)
        # Yahoo Finance period parameter is often more reliable than start/end with some intervals
        # For '1d' interval, start/end works well. For intraday, using start/end is crucial.
        data = crypto.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            print(f"    No data found for {ticker} in the specified period/interval. Skipping.")
            continue
            
        # Add ticker column
        data['Ticker'] = ticker
        
        # Basic price and volume features
        # Ensure division by zero is handled for returns and ratios if prices are 0
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1)).replace([np.inf, -np.inf], np.nan)
        data['Daily_Return'] = data['Close'].pct_change() # This is now 'Interval_Return'
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Price_Volume_Ratio'] = data['Close'] / (data['Volume'].replace(0, np.nan) + 1e-9) # Avoid div by zero volume
        
        # Add to main dataframe
        all_data = pd.concat([all_data, data], axis=0)
    
    # Sort by index (Date) and then by Ticker for consistent processing
    all_data = all_data.sort_index().sort_values(by='Ticker')
    
    # Drop NaNs after calculating basic features that depend on previous rows
    # This ensures a clean start for feature engineering
    # Note: For intraday data, especially at the start of trading days, NaNs can be common
    all_data = all_data.dropna(subset=['Log_Return', 'Daily_Return', 'Volume_Change', 'Price_Volume_Ratio'])
    
    # Save the fetched data to cache
    if not all_data.empty:
        print(f"Saving data to cache: {cache_filename}")
        with open(cache_filename, 'wb') as f:
            pickle.dump(all_data, f)
    else:
        print(f"No data to save for {', '.join(tickers)} after initial processing.")
    
    return all_data

def calculate_selected_features(data):
    """
    Calculate a comprehensive set of technical indicators from different feature groups.
    These indicators will serve as potential features for the machine learning model.

    Args:
        data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker' columns.

    Returns:
        pd.DataFrame: Original DataFrame with new technical indicator columns added.
    """
    print("Calculating technical indicators...")
    grouped = data.groupby('Ticker')
    result = pd.DataFrame()
    
    for ticker, group in grouped:
        df = group.copy()
        
        # === Moving Averages Group ===
        df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50) # May not have enough data for 50 periods in 20 days
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200) # Definitely not enough for 200 periods in ~20 days of 15m data
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # === Bollinger Bands Group ===
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'], timeperiod=20)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9) # Add epsilon
        
        # === Ichimoku Group (1 representative) ===
        # Note: Ichimoku calculations require a minimum of 26 periods for the base line
        # and 52 for lagging span (not implemented here fully)
        df['Ichimoku_Conversion'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Ichimoku_Base'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        
        # === Volatility Group ===
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Normalized_ATR'] = df['ATR'] / (df['Close'] + 1e-9) # Add epsilon
        df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
        
        # === Momentum Group ===
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

        # === Volume Indicator Group ===
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        # VWAP calculation should be cumulative over the entire period for proper interpretation
        # For grouped data, this is complex if not done on the raw full dataset before grouping.
        # As a simplified proxy, we'll calculate a rolling VWAP
        df['VWAP'] = ((df['Close'] * df['Volume']).rolling(window=20).sum()) / (df['Volume'].rolling(window=20).sum() + 1e-9)
        df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10) # Chaikin Money Flow (ADOSC is Accumulation/Distribution Oscillator)
        
        # === Price Pattern Group ===
        df['Parabolic_SAR'] = talib.SAR(df['High'], df['Low'])
        df['High_Low_Range'] = df['High'] - df['Low']
        df['HL_Range_Ratio'] = df['High_Low_Range'] / (df['Close'] + 1e-9) # Add epsilon
        
        # === Synthetic Features Group ===
        # Buying/Selling pressure based on intra-day range and volume
        df['Buying_Pressure'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)).fillna(0) * df['Volume']
        df['Selling_Pressure'] = ((df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-9)).fillna(0) * df['Volume']
        
        # Proximity to Bollinger Bands (as indicators of relative position within channel)
        df['Support_Proximity'] = (df['Close'] - df['BB_Lower']) / (df['Close'] + 1e-9)
        df['Resistance_Proximity'] = (df['BB_Upper'] - df['Close']) / (df['Close'] + 1e-9)
        
        # --- NEW: Replace infinities with NaN right after calculation for each group ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        result = pd.concat([result, df], axis=0)
    
    # Fill any remaining NaNs created by TA-Lib (e.g., initial periods of MAs)
    # This is a general fill, specific strategies for NaNs might be better
    # For now, fill with 0 or a sensible default, as some NaNs will be dropped later.
    result = result.fillna(method='ffill').fillna(method='bfill') # Forward fill, then backward fill for leading NaNs
    result = result.fillna(0) # Catch any remaining NaNs (e.g., if series was all NaNs)

    return result

def create_prediction_targets(data, forward_period=3):
    """
    Create prediction targets based on future price movements.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' and 'Ticker' columns.
        forward_period (int): Number of intervals into the future to calculate return for target.

    Returns:
        pd.DataFrame: Original DataFrame with 'Future_Return', 'Target', and 'Return_Magnitude' columns.
    """
    print(f"Creating prediction targets for {forward_period} intervals into the future...")
    # Calculate future percentage change in 'Close' price for each Ticker independently
    # .shift(-forward_period) aligns the future return to the current row's date
    data['Future_Return'] = data.groupby('Ticker')['Close'].pct_change(forward_period).shift(-forward_period)
    
    # Define the binary target: 1 if future return is >= 0.005 (0.5%), 0 otherwise
    # Adjust target threshold for intraday smaller movements if needed.
    data['Target'] = (data['Future_Return'] >= 0.005).astype(int) # Example: 0.5% gain
    
    # Calculate the absolute magnitude of the future return (for analysis, not directly used as target)
    data['Return_Magnitude'] = data['Future_Return'].abs()
    
    return data

def select_features_for_model(data):
    """
    Selects a predefined set of features for the model and creates additional
    ratio-based features to capture relationships between indicators and price.

    Args:
        data (pd.DataFrame): DataFrame containing raw data and calculated technical indicators.

    Returns:
        tuple:
            - pd.DataFrame: The input DataFrame with new ratio features added.
            - list: A list of all selected feature column names for model training.
    """
    # Base selected features (these are the TA indicators calculated previously)
    base_selected_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', # Including raw data as features
        'Log_Return', 'Daily_Return', 'Volume_Change', 'Price_Volume_Ratio',
        'SMA_5', 'SMA_20', 'SMA_50', # SMA_200 might not have enough data for 15m intervals
        'EMA_12', 'EMA_26',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        'Ichimoku_Conversion', 'Ichimoku_Base',
        'ATR', 'Normalized_ATR', 'Volatility_20',
        'RSI', 'MACD', 'MACD_Signal', 'MOM', 'Stoch_K', 'Stoch_D', 'ADX',
        'OBV', 'VWAP', 'CMF',
        'Parabolic_SAR', 'High_Low_Range', 'HL_Range_Ratio',
        'Buying_Pressure', 'Selling_Pressure', 'Support_Proximity', 'Resistance_Proximity'
    ]
    
    # Filter out features that are unlikely to have enough data for short intraday periods
    # e.g., SMA_50, SMA_200 periods need at least that many intervals
    # 20 days * 26 intervals/day = 520 intervals. So SMA_50, SMA_200 might be calculable but will have many NaNs
    # For very short training periods, it's better to stick to shorter-period indicators
    filtered_base_features = [f for f in base_selected_features if f in data.columns and f not in ['SMA_200']] 
    # Remove SMA_50 if training data is too short, but for 20 days (~520 intervals), 50 is fine.
    
    # Ensure all base features exist in the data before trying to create ratios
    available_features = [f for f in filtered_base_features if f in data.columns]
    
    # Create additional ratio-based features to capture relationships
    data['SMA5_Ratio'] = data['Close'] / (data['SMA_5'] + 1e-9)
    data['SMA20_Ratio'] = data['Close'] / (data['SMA_20'] + 1e-9)
    # data['SMA50_Ratio'] = data['Close'] / (data['SMA_50'] + 1e-9) # Keep if SMA_50 is included
    # data['SMA200_Ratio'] = data['Close'] / (data['SMA_200'] + 1e-9) # Remove if SMA_200 is removed
    data['EMA12_Ratio'] = data['Close'] / (data['EMA_12'] + 1e-9)
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-9)
    data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']
    data['EMA_Cross_Signal'] = (data['EMA_12'] > data['EMA_26']).astype(int)
    data['RSI_Overbought_Sell'] = (data['RSI'] > 70).astype(int)
    data['RSI_Oversold_Buy'] = (data['RSI'] < 30).astype(int)


    # Combine all selected features
    all_selected_features = available_features + [
        'SMA5_Ratio', 'SMA20_Ratio', # 'SMA50_Ratio', 'SMA200_Ratio',
        'EMA12_Ratio', 'BB_Position', 'MACD_Diff', 
        'EMA_Cross_Signal', 'RSI_Overbought_Sell', 'RSI_Oversold_Buy'
    ]
    
    # Filter out any features that might not have been created due to missing data or calculation issues
    final_features = [f for f in all_selected_features if f in data.columns]

    print(f"Selected {len(final_features)} features for the model.")
    return data, final_features

all_crypto_tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', # Reduced crypto tickers
]
all_stock_tickers = [
        'TSLA', 'NVDA', 'AMC' # Reduced stock tickers
]

ALL_TICKERS_FOR_RUN = all_crypto_tickers + all_stock_tickers

today = datetime.now()
live_data_fetch_start = (today - timedelta(days=2)).strftime('%Y-%m-%d') # Fetch last 2 days of 15m data
live_data_fetch_end = today.strftime('%Y-%m-%d') # Up to today
current_market_data = fetch_and_prepare_data(
                    ALL_TICKERS_FOR_RUN,
                    start_date=live_data_fetch_start,
                    end_date=live_data_fetch_end,
                    interval='15m',
                    force_download=True
                )

current_market_data = calculate_selected_features(current_market_data)
# Create prediction targets for the *live* data is not strictly necessary for generating signals
# but it ensures the DataFrame structure is consistent. Future_Return/Target will be NaN for the latest intervals.
current_market_data = create_prediction_targets(current_market_data, forward_period=3) 

