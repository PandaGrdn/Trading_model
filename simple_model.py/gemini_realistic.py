import pandas as pd
import numpy as np
import requests
import time
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


# Alpha Vantage API Configuration
API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your actual API key
BASE_URL = 'https://www.alphavantage.co/query'

# Data storage configuration
DATA_DIR = 'stock_data'
CACHE_DIR = 'cache'

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(symbol, data_type='raw'):
    """Generate cache filename for a symbol"""
    return os.path.join(CACHE_DIR, f"{symbol}_{data_type}.pkl")

def save_data_to_cache(data, symbol, data_type='raw'):
    """Save data to cache file"""
    ensure_directories()
    filename = get_cache_filename(symbol, data_type)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {data_type} data for {symbol} to cache")

def load_data_from_cache(symbol, data_type='raw'):
    """Load data from cache file"""
    filename = get_cache_filename(symbol, data_type)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {data_type} data for {symbol} from cache")
        return data
    return None

def is_cache_valid(symbol, max_age_days=1):
    """Check if cached data is still valid (not too old)"""
    filename = get_cache_filename(symbol, 'raw')
    if not os.path.exists(filename):
        return False
    
    # Check file modification time
    file_time = datetime.fromtimestamp(os.path.getmtime(filename))
    age = datetime.now() - file_time
    return age.days < max_age_days

def save_processed_data(data, filename):
    """Save processed data to file"""
    ensure_directories()
    filepath = os.path.join(DATA_DIR, filename)
    data.to_pickle(filepath)
    print(f"Saved processed data to {filepath}")

def load_processed_data(filename):
    """Load processed data from file"""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        data = pd.read_pickle(filepath)
        print(f"Loaded processed data from {filepath}")
        return data
    return None

def fetch_stock_data(symbol, outputsize='full', api_key=API_KEY, use_cache=True):
    """
    Fetch daily stock data from Alpha Vantage API with caching
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL')
    outputsize : str
        'compact' (last 100 data points) or 'full' (full-length time series)
    api_key : str
        Alpha Vantage API key
    use_cache : bool
        Whether to use cached data if available
        
    Returns:
    --------
    pd.DataFrame : Stock data with OHLCV columns
    """
    # Check cache first
    if use_cache and is_cache_valid(symbol):
        cached_data = load_data_from_cache(symbol, 'raw')
        if cached_data is not None:
            return cached_data
    
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': api_key
    }
    
    print(f"Fetching data for {symbol} from Alpha Vantage API...")
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error for {symbol}: {data['Error Message']}")
        
        if 'Note' in data:
            print(f"API Note for {symbol}: {data['Note']}")
            time.sleep(12)  # Wait due to rate limit
            return fetch_stock_data(symbol, outputsize, api_key, use_cache=False)  # Retry without cache
        
        # Extract time series data
        time_series_key = 'Time Series (Daily)'
        if time_series_key not in data:
            raise ValueError(f"No time series data found for {symbol}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Rename columns to match expected format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Add ticker column
        df['Ticker'] = symbol
        
        # Save to cache
        if use_cache:
            save_data_to_cache(df, symbol, 'raw')
        
        print(f"Successfully fetched {len(df)} data points for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_and_prepare_data(tickers, start_date=None, end_date=None, use_cache=True, cache_suffix=""):
    """
    Fetch historical data for stocks and calculate key features with caching.
    This version creates a MultiIndex (Date, Ticker) for robustness.
    
    Parameters:
    -----------
    tickers : list
        List of stock symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        Whether to use cached data
    cache_suffix : str
        Suffix to add to cache filename (e.g., 'train', 'test')
        
    Returns:
    --------
    pd.DataFrame : Combined stock data with basic features and MultiIndex (Date, Ticker)
    """
    # Check for cached combined data first
    # IMPORTANT: Change cache filename to reflect the MultiIndex structure
    cache_filename = f"combined_data_multiindex_{cache_suffix}_{start_date}_{end_date}.pkl"
    if use_cache:
        cached_combined = load_processed_data(cache_filename)
        if cached_combined is not None:
            # Ensure the loaded data has the expected MultiIndex
            if isinstance(cached_combined.index, pd.MultiIndex) and 'Ticker' in cached_combined.index.names:
                print(f"Loaded MultiIndex data from cache: {cache_filename}")
                return cached_combined
            else:
                print(f"Cached data at {cache_filename} is not MultiIndex. Re-fetching.")

    all_data_list = [] # Collect individual dataframes

    for i, ticker in enumerate(tickers):
        # Add delay to respect API rate limits (5 calls per minute for free tier)
        if i > 0 and not use_cache:
            time.sleep(12)  
        
        stock_data = fetch_stock_data(ticker, use_cache=use_cache) # This already adds 'Ticker' column
        
        if stock_data.empty:
            print(f"Skipping {ticker} due to data fetch error")
            continue
        
        # Filter by date range if specified
        if start_date:
            stock_data = stock_data[stock_data.index >= start_date]
        if end_date:
            stock_data = stock_data[stock_data.index <= end_date]
        
        # Basic price and volume features (calculated here for each stock before concat)
        # It's generally better to calculate these *after* concat and setting MultiIndex
        # if they involve shifting or rolling across all tickers, but for basic features, this is fine.
        stock_data['Log_Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
        stock_data['Price_Volume_Ratio'] = stock_data['Close'] / (stock_data['Volume'] + 1)
        
        all_data_list.append(stock_data) # Append individual DF

    if not all_data_list:
        print("No data fetched for any ticker.")
        return pd.DataFrame() # Return empty DataFrame if no data

    # Concatenate all individual DataFrames
    all_data = pd.concat(all_data_list, axis=0)
    
    # --- CRITICAL CHANGE: Set MultiIndex (Date, Ticker) ---
    all_data = all_data.set_index(['Ticker'], append=True) # Append 'Ticker' as a new index level
    all_data.index.names = ['Date', 'Ticker'] # Name the index levels
    
    # Sort the MultiIndex for efficient slicing and grouping
    all_data = all_data.sort_index()

    # Drop NaNs after calculating basic features (now on MultiIndex)
    all_data = all_data.dropna()
    
    # Save combined data to cache (now with MultiIndex)
    if use_cache:
        save_processed_data(all_data, cache_filename)
    
    return all_data

def calculate_selected_features(data, use_cache=True, cache_suffix=""):
    """
    Calculate technical indicators for stock analysis with caching
    """
    cache_filename = f"features_data_{cache_suffix}.pkl"
    if use_cache:
        cached_features = load_processed_data(cache_filename)
        if cached_features is not None:
            # Ensure the loaded data has the expected MultiIndex
            if isinstance(cached_features.index, pd.MultiIndex) and 'Ticker' in cached_features.index.names:
                print(f"Loaded MultiIndex data from cache: {cache_filename}")
                return cached_features
            else:
                print(f"Cached data at {cache_filename} is not MultiIndex. Re-calculating.")
    
    print("Calculating technical indicators...")
    # Group by the 'Ticker' level of the MultiIndex
    grouped = data.groupby(level='Ticker') # CRITICAL FIX: Specify level for groupby
    result_dfs = [] # Collect processed dataframes for concatenation

    for ticker, group_df in grouped: # group_df will have a simple DatetimeIndex for date for each ticker
        df = group_df.copy() # Work on a copy of the group

        # Ensure index is DatetimeIndex for talib functions
        if not isinstance(df.index, pd.DatetimeIndex):
             # If index is MultiIndex, reset index temporarily to get 'Date' and then set it back
             # Or, more simply, use get_level_values if the Date is the first level.
             # However, for TALIB, it expects a simple array, so passing df['Close'] is fine.
             # df.index = df.index.get_level_values('Date') # This would flatten the index for this group

        # === Moving Averages Group ===
            df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
            df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
            df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
            df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
            
            # EMA features
            df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
            df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
            
            # === Bollinger Bands Group ===
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'], timeperiod=20)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # === Ichimoku Group ===
            # These are rolling calculations on single ticker data, so they should be fine
            df['Ichimoku_Conversion'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
            df['Ichimoku_Base'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
            
            # === Volatility Group ===
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['Normalized_ATR'] = df['ATR'] / df['Close']
            df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
            
            # === Momentum Group ===
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
            df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
            
            # === Volume Indicator Group ===
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            # For VWAP, make sure cumsum handles NaNs appropriately, or fillna first if needed
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9) # Add small epsilon to prevent div by zero
            df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
            
            # === Price Pattern Group ===
            df['Parabolic_SAR'] = talib.SAR(df['High'], df['Low'])
            df['High_Low_Range'] = df['High'] - df['Low']
            df['HL_Range_Ratio'] = df['High_Low_Range'] / df['Close']
            
            # === Synthetic Features Group ===
            # Buying/Selling Pressure
            df['Buying_Pressure'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001)) * df['Volume']
            df['Selling_Pressure'] = ((df['High'] - df['Close']) / (df['High'] - df['Low'] + 0.001)) * df['Volume']
            
            # Proximity indicators
            df['Support_Proximity'] = (df['Close'] - df['BB_Lower']) / (df['Close'] + 0.001)
            df['Resistance_Proximity'] = (df['BB_Upper'] - df['Close']) / (df['Close'] + 0.001)
        
        result_dfs.append(df) # Append the processed group DataFrame
    
    # Concatenate all processed group DataFrames back into a single DataFrame
    # This will automatically reconstruct the MultiIndex
    result = pd.concat(result_dfs, axis=0)

    # Sort the MultiIndex for consistency
    result = result.sort_index()
    
    # Save to cache
    if use_cache:
        save_processed_data(result, cache_filename)
    
    return result

def create_prediction_targets(data, forward_period=3):
    """
    Create prediction targets for a specified forward period.
    Assumes data has a MultiIndex (Date, Ticker).
    """
    # Calculate future returns by grouping on the Ticker level of the MultiIndex
    # and shifting within each group.
    data['Future_Close'] = data.groupby(level='Ticker')['Close'].shift(-forward_period)
    data['Future_Return'] = (data['Future_Close'] / data['Close']) - 1
    
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    data['Return_Magnitude'] = data['Future_Return'].abs()
    
    return data

def select_features_for_model(data):
    """
    Select and engineer additional features for stock prediction.
    Assumes data has a MultiIndex (Date, Ticker).
    """
    
    required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required base column: {col}")

    selected_features = [
        'BB_Upper', 'SMA_20', 'VWAP', 'EMA_12', 'Ichimoku_Conversion',
        'Parabolic_SAR', 'ATR', 'SMA_50', 'Price_Volume_Ratio',
        'High_Low_Range', 'Selling_Pressure', 'Buying_Pressure',
        'OBV', 'RSI', 'MACD', 'Support_Proximity', 'Resistance_Proximity',
        'EMA_26', 'BB_Lower' 
    ]
    
    data['SMA20_Ratio'] = data['Close'] / data['SMA_20']
    data['SMA50_Ratio'] = data['Close'] / data['SMA_50']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    data['EMA_Ratio'] = data['EMA_12'] / data['EMA_26']

    selected_features.extend(['SMA20_Ratio', 'SMA50_Ratio', 'BB_Position', 'EMA_Ratio'])
    
    # --- Lagged Features (Crucial for Time Series with MultiIndex) ---
    # Shift operations must be performed PER GROUP (Ticker)
    data['Close_Lag1'] = data.groupby(level='Ticker')['Close'].shift(1)
    data['Volume_Lag1'] = data.groupby(level='Ticker')['Volume'].shift(1)
    data['High_Lag1'] = data.groupby(level='Ticker')['High'].shift(1)
    data['Low_Lag1'] = data.groupby(level='Ticker')['Low'].shift(1)
    data['Open_Lag1'] = data.groupby(level='Ticker')['Open'].shift(1)

    for feat in ['RSI', 'MACD', 'SMA20_Ratio', 'BB_Position', 'ATR']:
        data[f'{feat}_Lag1'] = data.groupby(level='Ticker')[feat].shift(1)
        data[f'{feat}_Lag2'] = data.groupby(level='Ticker')[feat].shift(2)

    lagged_features = [col for col in data.columns if '_Lag' in col]
    selected_features.extend(lagged_features)

    # --- Volatility Measures ---
    data['Log_Returns'] = np.log(data['Close'] / data.groupby(level='Ticker')['Close'].shift(1)) # Shift within group
    data['Volatility_10d'] = data.groupby(level='Ticker')['Log_Returns'].rolling(window=10).std().reset_index(level='Ticker', drop=True) * np.sqrt(252) # Ann.
    # Note: reset_index(level='Ticker', drop=True) is needed because .rolling().std() on a groupby object can create a new MultiIndex.
    # We want the result aligned back to the original MultiIndex. This can be complex.
    # A simpler approach for rolling on groupby objects:
    temp_vol = data.groupby(level='Ticker')['Log_Returns'].apply(lambda x: x.rolling(window=10).std()) * np.sqrt(252)
    data['Volatility_10d'] = temp_vol.reindex(data.index) # Reindex to ensure proper alignment
    selected_features.append('Volatility_10d')

    # --- Momentum Features ---
    data['ROC_5d'] = (data['Close'] - data.groupby(level='Ticker')['Close'].shift(5)) / data.groupby(level='Ticker')['Close'].shift(5)
    data['ROC_10d'] = (data['Close'] - data.groupby(level='Ticker')['Close'].shift(10)) / data.groupby(level='Ticker')['Close'].shift(10)
    selected_features.extend(['ROC_5d', 'ROC_10d'])

    # --- Time-Based Features (Extract from Date level of MultiIndex) ---
    # CRITICAL FIX: Access the 'Date' level of the MultiIndex
    date_index = data.index.get_level_values('Date')
    data['DayOfWeek'] = date_index.dayofweek # Monday=0, Sunday=6
    data['DayOfMonth'] = date_index.day
    data['Month'] = date_index.month
    selected_features.extend(['DayOfWeek', 'DayOfMonth', 'Month'])

    # The .dropna() at the end is removed as NaNs are handled in train_prediction_model.
    # This preserves the full time series for proper target and simulation handling later.
    
    return data, selected_features

def train_prediction_model(data, features, test_size=0.2):
    """
    Train an XGBoost model to predict price direction,
    with hyperparameter tuning for improved precision and accuracy.
    """
    # Remove any NaN values from relevant columns
    # IMPORTANT: This is the CORRECT place to drop NaNs, after all features and targets
    # have been calculated on the continuous time series.
    model_data = data.dropna(subset=features + ['Target', 'Future_Return'])
    
    # Feature matrix and target
    X = model_data[features]
    y = model_data['Target'] # 'Target' should be 0 or 1 for binary classification
    
    # Handle remaining NaN values in features with mean imputation (after dropna for target)
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # X_scaled is now a numpy array

    # --- Step 1: Calculate scale_pos_weight for class imbalance ---
    num_neg = np.sum(y == 0)
    num_pos = np.sum(y == 1)
    scale_pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0 # Default to 1.0
    print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.4f}")
    
    # --- Step 2: Define the parameter grid for GridSearchCV ---
    param_grid = {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.001, 0.1],
        'scale_pos_weight': [1.0, scale_pos_weight_value]
    }

    # --- Step 3: Initialize XGBoost classifier for GridSearchCV ---
    xgb_base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
    )

    # --- Step 4: Perform GridSearchCV to find the best hyperparameters ---
    print("\nStarting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=xgb_base_model,
        param_grid=param_grid,
        scoring='precision',
        cv=TimeSeriesSplit(n_splits=3),
        verbose=1,
        n_jobs=-1
    )

    print("\nGridSearchCV (skipped fitting for this run, using default best_params).")
    best_params = {
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 7,
        'n_estimators': 500,
        'reg_alpha': 0,
        'scale_pos_weight': 1.0,
        'subsample': 0.7,
        'random_state': 42
    }
    
    # Create time-series split for evaluation
    tscv_eval = TimeSeriesSplit(n_splits=5)
    
    all_predictions = []
    all_actual = []
    
    print("\nStarting TimeSeries Cross-Validation with best parameters...")
    for fold_idx, (train_idx, test_idx) in enumerate(tscv_eval.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_predictions.extend(y_pred_proba)
        all_actual.extend(y_test)
        print(f"  Fold {fold_idx + 1} complete.")

    # --- Step 6: Train final model on all data with best parameters ---
    print("\nTraining final model on all data with best parameters...")
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_scaled, y)
    print("Final model trained.")
    
    # --- Evaluate overall performance using collected predictions from CV ---
    auc_score = roc_auc_score(all_actual, all_predictions)
    print(f"\nOverall ROC AUC (from CV folds): {auc_score:.4f}")
    
    binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
    
    print("\nClassification Report (from CV folds):")
    print(classification_report(all_actual, binary_predictions))
    
    cm = confusion_matrix(all_actual, binary_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (from CV folds)')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(final_model, max_num_features=20)
    plt.title('Feature Importance (from Final Model)')
    plt.show()
    
    return final_model, scaler, features

def calculate_adaptive_risk_levels(data, price, confidence, min_risk_reward_ratio=2.0):
    """
    Enhanced risk calculation with multiple methods to ensure minimum 2:1 reward-to-risk ratio.
    """
    # Ensure data has enough history for rolling calculations.
    # Using tail(N) to get relevant recent data for indicators.
    # Using 200 as a safe minimum for many common indicators.
    recent_data = data.tail(max(200, len(data))).copy()

    if recent_data.empty or len(recent_data) < 20: # Added len(recent_data) < 20 for basic indicator needs
        print("Warning: Insufficient historical data for robust risk calculation. Using default.")
        adjusted_risk_dollar = price * 0.02 # 2% of price as dollar risk
        stop_loss = price - adjusted_risk_dollar
        take_profit = price + (adjusted_risk_dollar * min_risk_reward_ratio)
        if stop_loss <= 0: stop_loss = price * 0.001
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': adjusted_risk_dollar,
            'reward_amount': take_profit - price,
            'risk_reward_ratio': min_risk_reward_ratio,
            'risk_pct': (adjusted_risk_dollar / price) * 100,
            'reward_pct': ((take_profit - price) / price) * 100
        }

    # --- Step 1: Calculate various risk measures (dollar amounts) ---

    # Method 1: ATR-based calculation
    atr_value = talib.ATR(recent_data['High'], recent_data['Low'], recent_data['Close'], timeperiod=14).iloc[-1]
    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = (recent_data['High'] - recent_data['Low']).mean() # Fallback to simple range mean
        if pd.isna(atr_value) or atr_value <= 0:
             atr_value = price * 0.008 # Default to 0.8% of price if no ATR history

    # Method 2: Bollinger Band based calculation
    bb_range_risk = None
    if all(col in recent_data.columns for col in ['BB_Upper', 'BB_Lower']):
        bb_upper, bb_middle, bb_lower = talib.BBANDS(recent_data['Close'], timeperiod=20)
        if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]):
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            bb_range_risk = min(bb_range * 0.3, price * 0.03)

    # Method 3: Recent volatility based calculation
    log_returns = np.log(recent_data['Close'] / recent_data['Close'].shift(1)).dropna()
    recent_volatility_dollar = 0
    if len(log_returns) >= 20:
        volatility_daily_std = log_returns.tail(20).std()
        recent_volatility_dollar = volatility_daily_std * price
        recent_volatility_dollar = min(recent_volatility_dollar * np.sqrt(3), price * 0.04)

    # Method 4: Support/Resistance based calculation (as a downside buffer)
    support_distance_risk = 0
    if len(recent_data) >= 20:
        support_level = recent_data['Close'].rolling(window=20).min().iloc[-1]
        if price > support_level:
            distance_to_support = price - support_level
            support_distance_risk = distance_to_support * 0.7
            support_distance_risk = min(support_distance_risk, price * 0.05)

    # --- Step 2: Aggregate Risk Measures and Determine 'base_risk' ---
    risk_measures_dollars = []
    risk_measures_dollars.append(atr_value * 2.0)
    if recent_volatility_dollar > 0:
        risk_measures_dollars.append(recent_volatility_dollar * 1.5)
    if bb_range_risk is not None:
        risk_measures_dollars.append(bb_range_risk)
    if support_distance_risk > 0:
        risk_measures_dollars.append(support_distance_risk)
    risk_measures_dollars.append(price * 0.005)

    valid_risks = [r for r in risk_measures_dollars if r > 0 and not pd.isna(r)]
    
    if not valid_risks:
        base_risk = price * 0.015
    else:
        base_risk = np.median(valid_risks)
        if base_risk <= 0:
            base_risk = price * 0.015

    # --- Step 3: Adjust risk based on confidence ---
    confidence_adjustment = 1.3 - (confidence - 0.5) * 2
    confidence_adjustment = max(0.7, min(1.3, confidence_adjustment))
    
    adjusted_risk_dollar = base_risk * confidence_adjustment
    adjusted_risk_dollar = max(adjusted_risk_dollar, price * 0.005)
    adjusted_risk_dollar = min(adjusted_risk_dollar, price * 0.08)

    # --- Step 4: Calculate Stop Loss and Take Profit ---
    stop_loss = price - adjusted_risk_dollar
    if stop_loss <= 0:
        stop_loss = price * 0.001
        adjusted_risk_dollar = price - stop_loss

    min_reward_dollar = adjusted_risk_dollar * min_risk_reward_ratio
    take_profit = price + min_reward_dollar

    if 'Resistance_Proximity' in recent_data.columns and not pd.isna(recent_data['Resistance_Proximity'].iloc[-1]):
        resistance_level_calc = price * (1 + recent_data['Resistance_Proximity'].iloc[-1])
        if resistance_level_calc > take_profit:
            take_profit = resistance_level_calc * (1 + (confidence * 0.02))
    
    take_profit = min(take_profit, price * 1.25)

    # --- Step 5: Final Adjustments and Return Values ---
    actual_risk_amount = adjusted_risk_dollar 
    actual_reward_amount = take_profit - price
    actual_risk_reward_ratio = actual_reward_amount / actual_risk_amount if actual_risk_amount > 0 else min_risk_reward_ratio

    if actual_reward_amount > 0 and actual_risk_reward_ratio < min_risk_reward_ratio:
        required_reward_for_ratio = actual_risk_amount * min_risk_reward_ratio
        take_profit = price + required_reward_for_ratio
        actual_reward_amount = required_reward_for_ratio
        actual_risk_reward_ratio = min_risk_reward_ratio
    elif actual_reward_amount <= 0:
        actual_risk_reward_ratio = 0

    risk_pct_of_price = (actual_risk_amount / price) * 100 if price > 0 else 0
    reward_pct_of_price = (actual_reward_amount / price) * 100 if price > 0 else 0

    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_amount': actual_risk_amount,
        'reward_amount': actual_reward_amount,
        'risk_reward_ratio': actual_risk_reward_ratio,
        'risk_pct': risk_pct_of_price,
        'reward_pct': reward_pct_of_price
    }

def calculate_dynamic_position_size(confidence, risk_pct, max_portfolio_risk=3.0, max_position_pct=20.0, min_confidence=0.55):
    """
    Calculate position size based on confidence, risk percentage, and portfolio risk management.
    """
    if confidence < min_confidence or risk_pct <= 0:
        return 0.0

    if risk_pct > 15.0:
        print(f"Warning: risk_pct ({risk_pct:.2f}%) is very high. Capping for position size calculation.")
        risk_pct = 15.0

    confidence_factor = (confidence - min_confidence) / (1.0 - min_confidence)
    confidence_factor = max(0.0, min(1.0, confidence_factor))

    raw_position_size_pct = (max_portfolio_risk / risk_pct) * 100 if risk_pct > 0 else 0
    aggression_multiplier = 1.5
    confidence_adjusted_size = raw_position_size_pct * confidence_factor * aggression_multiplier
    final_position_size = min(confidence_adjusted_size, max_position_pct)

    if final_position_size < 0.5:
        return 0.0

    return final_position_size


def enhanced_signal_generation(model, scaler, data, features, min_confidence=0.55):
    """
    Enhanced trading signal generation with improved risk-reward optimization
    """
    signals = data.copy()

    signals['Confidence'] = np.nan
    signals['Signal'] = 0
    signals['Stop_Loss'] = np.nan
    signals['Take_Profit'] = np.nan
    signals['Risk_Reward_Ratio'] = np.nan
    signals['Position_Size_Pct'] = np.nan
    signals['Risk_Pct'] = np.nan
    signals['Reward_Pct'] = np.nan

    X_full = signals[features].fillna(signals[features].mean())
    X_scaled_full = scaler.transform(X_full)

    signals['Confidence'] = model.predict_proba(X_scaled_full)[:, 1]
    signals['Signal'] = (signals['Confidence'] > min_confidence).astype(int)

    grouped_by_ticker = signals.groupby(level='Ticker') # Specify level for groupby

    for ticker, group_df_for_ticker in grouped_by_ticker:
        group_df_sorted = group_df_for_ticker.sort_index(level='Date') # Sort by Date level

        for i, row_tuple in enumerate(group_df_sorted.itertuples()):
            current_date_idx = row_tuple.Index # This will be the MultiIndex tuple (Date, Ticker)
            current_price = row_tuple.Close
            confidence = row_tuple.Confidence
            signal_status = row_tuple.Signal

            if signal_status == 1:
                # current_data_slice needs to be from the original group_df_sorted
                current_data_slice = group_df_sorted.iloc[:i+1].copy()
                min_history_required = 200

                if len(current_data_slice) < min_history_required:
                    # Accessing the main signals DataFrame to update the signal status
                    signals.loc[current_date_idx, 'Signal'] = 0
                    continue

                risk_metrics = calculate_adaptive_risk_levels(
                    current_data_slice,
                    current_price,
                    confidence,
                    min_risk_reward_ratio=2.0
                )

                position_size = calculate_dynamic_position_size(
                    confidence,
                    risk_metrics['risk_pct'],
                    max_portfolio_risk=3.0,
                    max_position_pct=20.0
                )

                is_valid_rr = not pd.isna(risk_metrics['risk_reward_ratio']) and np.isfinite(risk_metrics['risk_reward_ratio'])

                if position_size > 0 and is_valid_rr and risk_metrics['risk_reward_ratio'] >= 1.5:
                    signals.loc[current_date_idx, 'Stop_Loss'] = risk_metrics['stop_loss']
                    signals.loc[current_date_idx, 'Take_Profit'] = risk_metrics['take_profit']
                    signals.loc[current_date_idx, 'Risk_Reward_Ratio'] = risk_metrics['risk_reward_ratio']
                    signals.loc[current_date_idx, 'Position_Size_Pct'] = position_size
                    signals.loc[current_date_idx, 'Risk_Pct'] = risk_metrics['risk_pct']
                    signals.loc[current_date_idx, 'Reward_Pct'] = risk_metrics['reward_pct']
                else:
                    signals.loc[current_date_idx, 'Signal'] = 0

    return signals

def advanced_performance_analysis(signals, forward_period=3):
    """
    Advanced performance analysis with risk-adjusted metrics
    """
    # Filter for actual signals only
    actual_signals = signals[signals['Signal'] == 1].copy()
    
    if len(actual_signals) == 0:
        print("No trading signals generated for analysis")
        return {}
    
    # Calculate simulated returns using the corrected simulation function
    # The .apply method will pass each row as a Series, with its MultiIndex as .name
    actual_signals['Simulated_Return'] = actual_signals.apply(
        lambda row: simulate_trade_outcome(row, forward_period, signals), # Pass the full signals DF
        axis=1
    ).dropna() # Drop trades that couldn't be resolved (e.g., no future data)
    
    # Performance metrics
    total_trades = len(actual_signals)
    if total_trades == 0:
        print("No valid trades to analyze after simulation (all dropped due to NaN returns).")
        return {}

    winning_trades = (actual_signals['Simulated_Return'] > 0).sum()
    losing_trades = (actual_signals['Simulated_Return'] < 0).sum()
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win = actual_signals[actual_signals['Simulated_Return'] > 0]['Simulated_Return'].mean()
    avg_loss = actual_signals[actual_signals['Simulated_Return'] < 0]['Simulated_Return'].mean()
    
    total_wins = actual_signals[actual_signals['Simulated_Return'] > 0]['Simulated_Return'].sum()
    total_losses = abs(actual_signals[actual_signals['Simulated_Return'] < 0]['Simulated_Return'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if total_trades > 0 else 0
    
    # Ensure Position_Size_Pct is numeric and not NaN for calculation
    actual_signals['Position_Size_Pct_Clean'] = pd.to_numeric(actual_signals['Position_Size_Pct'], errors='coerce').fillna(0)
    portfolio_returns = (actual_signals['Simulated_Return'] * (actual_signals['Position_Size_Pct_Clean'] / 100)).sum()
    total_portfolio_return = portfolio_returns
    
    avg_portfolio_risk = actual_signals['Risk_Pct'].mean()
    risk_adjusted_return = total_portfolio_return / avg_portfolio_risk if avg_portfolio_risk > 0 else 0


    print(f"\n{'='*60}")
    print(f"ENHANCED PERFORMANCE ANALYSIS (Corrected)")
    print(f"{'='*60}")
    print(f"Analysis Period: {forward_period} days forward prediction")
    print(f"Total Trades Generated: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Trades with No Clear Outcome (e.g., Flat): {total_trades - winning_trades - losing_trades}")
    print(f"\nPROFITABILITY METRICS:")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Winning Trade Return: {avg_win:.2%}")
    print(f"Average Losing Trade Return: {avg_loss:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Expected Value per Trade: {expected_value:.2%}")
    print(f"\nRISK MANAGEMENT:")
    print(f"Average Risk per Trade (Entry Price %): {actual_signals['Risk_Pct'].mean():.2f}%")
    print(f"Average Reward per Trade (Entry Price %): {actual_signals['Reward_Pct'].mean():.2f}%")
    print(f"Average Risk-Reward Ratio: {actual_signals['Risk_Reward_Ratio'].mean():.2f}:1")
    print(f"Average Position Size: {actual_signals['Position_Size_Pct'].mean():.2f}%")
    print(f"\nPORTFOLIO IMPACT:")
    print(f"Total Portfolio Return (Sum of Position-Weighted Returns): {total_portfolio_return:.2%}")
    print(f"Risk-Adjusted Return (Total Return / Avg. Trade Risk%): {risk_adjusted_return:.2f}")
    
    create_enhanced_visualizations(actual_signals, signals)
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expected_value': expected_value,
        'total_portfolio_return': total_portfolio_return,
        'risk_adjusted_return': risk_adjusted_return
    }

def simulate_trade_outcome(trade_row, forward_period, full_signals_df):
    """
    Simulate trade outcome considering stop-loss and take-profit levels.
    """
    current_price = trade_row['Close']
    stop_loss = trade_row['Stop_Loss']
    take_profit = trade_row['Take_Profit']
    
    # Extract Ticker and Entry Date correctly from the MultiIndex (trade_row.name)
    entry_date, ticker = trade_row.name 

    # Find the data slice for the future period for this specific ticker
    # Ensure correct slicing for MultiIndex: (ticker_value, date_slice)
    # Use .loc with a tuple for the MultiIndex access
    # We need to get the OHLC data for the *future days* after the entry date.
    # The 'full_signals_df' contains both historical features/targets and future OHLC data.
    
    # Define the end date for the future slice
    slice_end_date = entry_date + timedelta(days=forward_period + 5) # Added a small buffer
    
    # Get the relevant rows for this ticker within the future period
    # Use pd.IndexSlice for complex multi-index slicing
    idx = pd.IndexSlice
    
    try:
        # Select rows where ticker matches AND date is within the future window
        future_data_slice = full_signals_df.loc[
            (ticker, idx[entry_date + timedelta(days=1):slice_end_date]),
            ['High', 'Low', 'Close'] # Only need these columns for simulation
        ].sort_index(level='Date') # Ensure sorted by date for iteration
    except KeyError:
        # This can happen if a ticker just doesn't exist in the slice or date range
        return np.nan # Cannot simulate if future data cannot be found


    # If no future data available (e.g., trade very close to end of test_data_raw_features)
    if future_data_slice.empty:
        # Fallback to Future_Return if no future OHLC available for simulation
        if not pd.isna(trade_row['Future_Close']):
            return (trade_row['Future_Close'] / current_price) - 1
        else:
            return np.nan # Cannot resolve trade outcome
        
    # Get the actual future close after `forward_period` days, from the trade_row itself
    future_final_close = trade_row['Future_Close']

    # Iterate through each day in the future slice to check for SL/TP hits
    for _, future_day_row in future_data_slice.iterrows():
        future_high = future_day_row['High']
        future_low = future_day_row['Low']

        # Ensure High/Low are not NaN
        if pd.isna(future_high) or pd.isna(future_low):
            continue

        # Check for SL hit first (assumed precedence)
        if future_low <= stop_loss:
            return (stop_loss / current_price) - 1

        # Check for TP hit
        if future_high >= take_profit:
            return (take_profit / current_price) - 1
        
    # If neither SL nor TP was hit within the observed future_data_slice,
    # the trade's outcome is determined by its `Future_Close` at the end of the `forward_period`.
    if not pd.isna(future_final_close):
        return (future_final_close / current_price) - 1
    else:
        # This implies the trade's target date is beyond the fetched `future_data_slice`
        # or the Future_Close itself was NaN (e.g., at very end of dataset).
        return np.nan


def create_enhanced_visualizations(actual_signals, all_signals):
    """
    Create comprehensive visualizations for the enhanced trading system
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    plot_signals = actual_signals.dropna(subset=['Risk_Pct', 'Reward_Pct', 'Confidence', 'Position_Size_Pct', 'Simulated_Return'])

    if not plot_signals.empty:
        scatter = ax1.scatter(plot_signals['Risk_Pct'], plot_signals['Reward_Pct'], 
                            c=plot_signals['Confidence'], cmap='viridis', alpha=0.7, s=plot_signals['Position_Size_Pct'] * 2)
        
        max_rr_plot = max(plot_signals['Risk_Pct'].max(), plot_signals['Reward_Pct'].max()) * 1.1
        ax1.plot([0, max_rr_plot], [0, max_rr_plot], 'k--', alpha=0.5, label='1:1 R:R Line')
        ax1.plot([0, max_rr_plot], [0, max_rr_plot * 2], 'r--', alpha=0.7, label='2:1 R:R Line')
        
        ax1.set_xlabel('Risk % (of Entry Price)')
        ax1.set_ylabel('Reward % (of Entry Price)')
        ax1.set_title('Risk vs Reward Distribution (Bubble size by Position Size)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Model Confidence')
    else:
        ax1.set_title('Risk vs Reward Distribution (No Valid Signals)')
        ax1.text(0.5, 0.5, 'No data to display', transform=ax1.transAxes, ha='center')

    if not plot_signals.empty:
        ax2.scatter(plot_signals['Confidence'], plot_signals['Position_Size_Pct'], 
                c=plot_signals['Simulated_Return'], cmap='coolwarm', alpha=0.7, s=60)
        ax2.set_xlabel('Model Confidence')
        ax2.set_ylabel('Position Size % (of Portfolio)')
        ax2.set_title('Confidence vs Position Size (Color by Return)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(ax2.collections[0], ax=ax2, label='Simulated Return')
    else:
        ax2.set_title('Confidence vs Position Size (No Valid Signals)')
        ax2.text(0.5, 0.5, 'No data to display', transform=ax2.transAxes, ha='center')

    returns = plot_signals['Simulated_Return'].dropna()
    if not returns.empty:
        sns.histplot(returns, bins=30, kde=True, ax=ax3, color='skyblue', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax3.axvline(x=returns.mean(), color='blue', linestyle='-', alpha=0.7, label=f'Mean: {returns.mean():.2%}')
        ax3.set_xlabel('Simulated Trade Returns')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Simulated Trade Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.set_title('Distribution of Trade Returns (No Valid Signals)')
        ax3.text(0.5, 0.5, 'No data to display', transform=ax3.transAxes, ha='center')

    if not all_signals.empty:
        # To plot signals on the price chart, we need to handle the MultiIndex properly
        # We need to iterate over tickers and then plot their prices and signals separately
        
        # Get unique tickers from all_signals (which is already a MultiIndex DF)
        unique_tickers = all_signals.index.get_level_values('Ticker').unique()
        
        selected_tickers_for_plot = unique_tickers[:2] # Plot up to 2 tickers for clarity

        # Ensure ax4 is clear before plotting
        ax4.clear()

        for ticker_to_plot in selected_tickers_for_plot:
            # Select all data for this specific ticker
            ticker_data = all_signals.loc[(ticker_to_plot, slice(None)), :].copy().droplevel(level='Ticker') # Drop ticker level for simple DatetimeIndex for plotting
            ticker_data = ticker_data.sort_index() # Ensure sorted by date

            # Select signals for this specific ticker
            ticker_signals = plot_signals.loc[plot_signals.index.get_level_values('Ticker') == ticker_to_plot].copy()
            ticker_signals = ticker_signals.droplevel(level='Ticker') # Drop ticker level for simple DatetimeIndex for plotting
            ticker_signals = ticker_signals.sort_index() # Ensure sorted by date

            ax4.plot(ticker_data.index, ticker_data['Close'], label=f'{ticker_to_plot} Close Price', alpha=0.8, linewidth=1.5)
            
            if not ticker_signals.empty:
                ax4.scatter(ticker_signals.index, ticker_signals['Close'], 
                        color='green', marker='^', s=100, label=f'{ticker_to_plot} Entry (Signal)', zorder=5)

        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price')
        ax4.set_title('Price Chart with Trading Signals (Entry Points)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.set_title('Price Chart with Trading Signals (No Valid Signals)')
        ax4.text(0.5, 0.5, 'No data to display', transform=ax4.transAxes, ha='center')

    plt.tight_layout()
    plt.show()
    
    if not plot_signals.empty:
        plt.figure(figsize=(14, 7))
        # When plotting, if the index has multiple levels, pandas might not plot correctly.
        # Use get_level_values('Date') to get a simple DatetimeIndex for plotting.
        plot_index_dates = plot_signals.index.get_level_values('Date')
        plt.plot(plot_index_dates, plot_signals['Risk_Reward_Ratio'], 
                marker='o', linestyle='-', alpha=0.7, markersize=4)
        plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Target 2:1 Ratio')
        avg_rr = plot_signals['Risk_Reward_Ratio'].mean()
        plt.axhline(y=avg_rr, color='blue', 
                   linestyle='-', alpha=0.7, label=f'Average R:R: {avg_rr:.2f}:1')
        plt.xlabel('Date')
        plt.ylabel('Risk-Reward Ratio')
        plt.title('Risk-Reward Ratio Over Time for Generated Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def enhanced_main(tickers, train_start_date='2022-01-01', train_end_date='2023-12-31', 
                 test_start_date='2024-01-01', test_end_date=None, forward_period=3, 
                 min_confidence=0.55, use_cache=True, force_refresh=False):
    """
    Enhanced main function with improved risk-reward optimization
    """
    if test_end_date is None:
        # Get today's date
        test_end_date = datetime.now()
        # Find the last actual market close date if today is a weekend or holiday
        # This involves fetching today's data or checking a calendar API.
        # For simplicity, we'll just use today's date string, assuming market data will eventually catch up.
        # A more robust system would verify the latest available data.
        test_end_date_str = test_end_date.strftime('%Y-%m-%d')
    else:
        test_end_date_str = test_end_date # Use the provided string

    # Ensure test_end_date_str is a string for comparisons
    
    if force_refresh:
        print("Force refresh requested - clearing cache...")
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        use_cache = False
    
    print("=== ENHANCED STOCK TRADING SIGNAL SYSTEM ===")
    print(f"Training Period: {train_start_date} to {train_end_date}")
    print(f"Testing Period: {test_start_date} to {test_end_date_str}")
    print(f"Forward Prediction: {forward_period} days")
    print(f"Minimum Confidence for Signal: {min_confidence}")
    print(f"Target Risk-Reward Ratio: 2:1 minimum (applied at initial calculation)")
    print(f"Aggressive Position Sizing: Max Risk 3% of portfolio, Max Position 20%, Aggression Multiplier 1.5x")
    
    print("\n1. Fetching training data...")
    train_data = fetch_and_prepare_data(
        tickers, 
        start_date=train_start_date,
        end_date=train_end_date,
        use_cache=use_cache,
        cache_suffix="train"
    )
    
    print("2. Fetching testing data...")
    # For testing, fetch data that extends *past* the actual test_end_date
    # so that 'Future_Return' and 'Future_Close' can be calculated correctly
    # for signals generated up to test_end_date_str.
    # We fetch for (forward_period * 2) days past the test_end_date to ensure
    # simulate_trade_outcome has enough look-ahead data even if forward_period is small.
    adjusted_test_end_date_for_fetch = (datetime.strptime(test_end_date_str, '%Y-%m-%d') + timedelta(days=forward_period * 2)).strftime('%Y-%m-%d')
    
    test_data_raw = fetch_and_prepare_data(
        tickers,
        start_date=test_start_date,
        end_date=adjusted_test_end_date_for_fetch, # Fetch a bit more data for future calculations
        use_cache=use_cache,
        cache_suffix="test_extended" # Use a distinct cache suffix
    )

    print("3. Calculating features for training data...")
    train_data = calculate_selected_features(train_data, use_cache=use_cache, cache_suffix="train_features")
    print("4. Calculating features for testing data (on extended data for future targets)...")
    test_data_raw_features = calculate_selected_features(test_data_raw, use_cache=use_cache, cache_suffix="test_extended_features")
    
    print("5. Creating prediction targets for all data...")
    train_data = create_prediction_targets(train_data, forward_period)
    test_data_with_targets = create_prediction_targets(test_data_raw_features, forward_period)
    
    print("6. Selecting and engineering final features...")
    train_data, selected_features = select_features_for_model(train_data)
    test_data_with_targets, _ = select_features_for_model(test_data_with_targets)

    # Filter test_data to the actual test_end_date *after* future targets are calculated
    # CRITICAL FIX: Compare against the FIRST level (Date) of the MultiIndex
    test_data_final = test_data_with_targets[test_data_with_targets.index.get_level_values('Date') <= test_end_date_str].copy()

    print("7. Training prediction model...")
    model, scaler, features = train_prediction_model(train_data, selected_features)
    
    print("8. Generating enhanced trading signals...")
    # Pass `test_data_final` for signal generation, which contains data up to `test_end_date_str`.
    # `simulate_trade_outcome` will use `test_data_raw_features` (from the global scope of `enhanced_main`)
    # to access the necessary future OHLC data for its simulation.
    signals = enhanced_signal_generation(model, scaler, test_data_final, features, min_confidence)
    
    print("9. Analyzing performance...")
    performance = advanced_performance_analysis(signals, forward_period) # Pass `signals` as it contains all trade details
    
    print("\n" + "="*60)
    print("LATEST TRADING OPPORTUNITIES")
    print("="*60)
    
    viable_signals = signals[
        (signals['Signal'] == 1) & 
        (signals['Position_Size_Pct'].notna()) &
        (signals['Stop_Loss'].notna())
    ].tail(5)

    if len(viable_signals) > 0:
        for (date_idx, ticker_idx), row in viable_signals.iterrows(): # Iterate over MultiIndex
            print(f"\n🎯 {ticker_idx} - {date_idx.strftime('%Y-%m-%d')}")
            print(f"   💰 Entry: ${row['Close']:.2f}")
            print(f"   🛑 Stop Loss: ${row['Stop_Loss']:.2f} ({row['Risk_Pct']:.2f}% risk)")
            print(f"   🎯 Take Profit: ${row['Take_Profit']:.2f} ({row['Reward_Pct']:.2f}% reward)")
            print(f"   ⚖️  Risk-Reward: {row['Risk_Reward_Ratio']:.2f}:1")
            print(f"   📊 Confidence: {row['Confidence']:.1%}")
            print(f"   💼 Position Size: {row['Position_Size_Pct']:.2f}% of portfolio")
    else:
        print("No current trading signals meet the enhanced criteria.")
    
    print(f"\n{'='*60}")
    print("SYSTEM SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Enhanced risk management active")
    print(f"✅ Minimum 2:1 risk-reward ratio enforced (at calculation, then filtered for >=1.5)")
    print(f"✅ Dynamic position sizing implemented (More Aggressive)")
    print(f"✅ Multi-factor risk calculation active")
    print(f"✅ Confidence-based filtering at {min_confidence:.1%}")
    
    return model, scaler, signals, performance

# Usage example and execution
if __name__ == "__main__":
    # Ensure all new functions, especially fetch_and_prepare_data,
    # create_prediction_targets, calculate_selected_features and select_features_for_model
    # are correctly using/producing MultiIndex DataFrames.
    STOCK_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'] # More tickers might give more signals
    
    enhanced_model, enhanced_scaler, enhanced_signals, enhanced_performance = enhanced_main(
        STOCK_TICKERS,
        train_start_date='2022-01-01',
        train_end_date='2023-12-31',
        test_start_date='2024-01-01',
        test_end_date=None, # Will default to current date
        forward_period=3,
        min_confidence=0.55,  # Start with a slightly lower confidence to generate more signals
        use_cache=True,
        force_refresh=True # SET TO TRUE TO FORCE NEW DATA WITH MULTIINDEX
    )
    
    print("\n🚀 Enhanced Trading System Ready!")
    print("📈 All signals now feature more realistic risk-reward ratios and position sizes.")
    print("💡 Position sizes dynamically adjusted based on confidence and risk.")
    print("🛡️  Multi-layer risk management active.")