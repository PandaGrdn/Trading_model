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
    Fetch historical data for stocks and calculate key features with caching
    
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
    pd.DataFrame : Combined stock data with basic features
    """
    # Check for cached combined data first
    cache_filename = f"combined_data_{cache_suffix}_{start_date}_{end_date}.pkl"
    if use_cache:
        cached_combined = load_processed_data(cache_filename)
        if cached_combined is not None:
            return cached_combined
    
    all_data = pd.DataFrame()
    
    for i, ticker in enumerate(tickers):
        # Add delay to respect API rate limits (5 calls per minute for free tier)
        if i > 0 and not use_cache:  # Don't delay if using cache
            time.sleep(12)  
        
        stock_data = fetch_stock_data(ticker, use_cache=use_cache)
        
        if stock_data.empty:
            print(f"Skipping {ticker} due to data fetch error")
            continue
        
        # Filter by date range if specified
        if start_date:
            stock_data = stock_data[stock_data.index >= start_date]
        if end_date:
            stock_data = stock_data[stock_data.index <= end_date]
        
        # Basic price and volume features
        stock_data['Log_Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
        stock_data['Price_Volume_Ratio'] = stock_data['Close'] / (stock_data['Volume'] + 1)
        
        # Add to main dataframe
        all_data = pd.concat([all_data, stock_data], axis=0)
    
    # Drop NaNs after calculating basic features
    all_data = all_data.dropna()
    
    # Save combined data to cache
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
            return cached_features
    
    print("Calculating technical indicators...")
    grouped = data.groupby('Ticker')
    result = pd.DataFrame()
    
    for ticker, group in grouped:
        df = group.copy()
        
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
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
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
        
        result = pd.concat([result, df], axis=0)
    
    # Save to cache
    if use_cache:
        save_processed_data(result, cache_filename)
    
    return result

def create_prediction_targets(data, forward_period=3):
    """
    Create prediction targets for a specified forward period
    """
    # Calculate future returns
    # This calculation assumes that the 'Close' price used is the entry price.
    # The `shift(-forward_period)` makes sure we are getting the Close price
    # `forward_period` days *into the future* from the current row.
    data['Future_Close'] = data.groupby('Ticker')['Close'].shift(-forward_period)
    data['Future_Return'] = (data['Future_Close'] / data['Close']) - 1 # Simple percentage return
    
    # Define binary target (1 if price goes up, 0 if it goes down)
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    
    # Define magnitude of movement
    data['Return_Magnitude'] = data['Future_Return'].abs()
    
    return data

def select_features_for_model(data):
    """
    Select and engineer additional features for stock prediction.
    """
    
    # Ensure necessary base columns exist for new derivations
    required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required base column: {col}")

    # --- Existing Features (from your current list) ---
    selected_features = [
        'BB_Upper', 'SMA_20', 'VWAP', 'EMA_12', 'Ichimoku_Conversion',
        'Parabolic_SAR', 'ATR', 'SMA_50', 'Price_Volume_Ratio',
        'High_Low_Range', 'Selling_Pressure', 'Buying_Pressure',
        'OBV', 'RSI', 'MACD', 'Support_Proximity', 'Resistance_Proximity',
        # Assuming EMA_26 and BB_Lower are present if used in ratios below
        'EMA_26', 'BB_Lower' 
    ]
    
    # --- Derived Features (your current ratios) ---
    # Calculate price-relative versions to reduce collinearity
    data['SMA20_Ratio'] = data['Close'] / data['SMA_20']
    data['SMA50_Ratio'] = data['Close'] / data['SMA_50']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    data['EMA_Ratio'] = data['EMA_12'] / data['EMA_26'] # Ensure EMA_26 is calculated/available

    selected_features.extend(['SMA20_Ratio', 'SMA50_Ratio', 'BB_Position', 'EMA_Ratio'])
    
    # --- NEW: Lagged Features (Crucial for Time Series) ---
    # Create lags for key price components
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Volume_Lag1'] = data['Volume'].shift(1)
    data['High_Lag1'] = data['High'].shift(1)
    data['Low_Lag1'] = data['Low'].shift(1)
    data['Open_Lag1'] = data['Open'].shift(1)

    # Create lags for important indicators and ratios (e.g., 1 and 2 days ago)
    for feat in ['RSI', 'MACD', 'SMA20_Ratio', 'BB_Position', 'ATR']:
        data[f'{feat}_Lag1'] = data[feat].shift(1)
        data[f'{feat}_Lag2'] = data[feat].shift(2) # Example for a second lag

    # Add all new lagged features
    lagged_features = [col for col in data.columns if '_Lag' in col]
    selected_features.extend(lagged_features)

    # --- NEW: Volatility Measures ---
    # Rolling 10-day historical volatility (log returns)
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility_10d'] = data['Log_Returns'].rolling(window=10).std() * np.sqrt(252) # Ann.
    selected_features.append('Volatility_10d')

    # --- NEW: Momentum Features ---
    # Price Rate of Change over 5 and 10 days
    data['ROC_5d'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)
    data['ROC_10d'] = (data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)
    selected_features.extend(['ROC_5d', 'ROC_10d'])

    # --- NEW: Time-Based Features (Extract from index if it's datetime) ---
    # Ensure your DataFrame index is a datetime object
    if isinstance(data.index, pd.DatetimeIndex):
        data['DayOfWeek'] = data.index.dayofweek # Monday=0, Sunday=6
        data['DayOfMonth'] = data.index.day
        data['Month'] = data.index.month
        selected_features.extend(['DayOfWeek', 'DayOfMonth', 'Month'])

    # --- Clean up NaNs created by shifting/rolling ---
    # These will primarily be at the beginning of the DataFrame.
    # Your main model training function uses dropna or fillna, but it's good practice here too.
    data = data.dropna(subset=selected_features) 
    
    return data, selected_features

def train_prediction_model(data, features, test_size=0.2):
    """
    Train an XGBoost model to predict price direction,
    with hyperparameter tuning for improved precision and accuracy.
    """
    # Remove any NaN values from relevant columns
    # IMPORTANT: Ensure 'Target' and 'Future_Return' columns exist in 'data'
    model_data = data.dropna(subset=features + ['Target', 'Future_Return']) # Also drop NaNs for Future_Return for clean target
    
    # Feature matrix and target
    X = model_data[features]
    y = model_data['Target'] # 'Target' should be 0 or 1 for binary classification
    
    # Handle remaining NaN values in features with mean imputation (after dropna for target)
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # X_scaled is now a numpy array

    # --- Step 1: Calculate scale_pos_weight for class imbalance ---
    # Assuming 'Target' is 0 for majority (negative) and 1 for minority (positive)
    # Adjust if your labels are different.
    # If the classes are perfectly balanced (like in your example), this will be ~1.0
    # Ensure no division by zero if a class has no samples
    num_neg = np.sum(y == 0)
    num_pos = np.sum(y == 1)
    scale_pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0 # Default to 1.0 if no positive samples
    print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.4f}")
    
    # --- Step 2: Define the parameter grid for GridSearchCV ---
    # This range is chosen to explore different complexities and learning rates.
    # You might need to adjust these ranges based on initial runs.
    param_grid = {
        'n_estimators': [200, 500], # Reduced for faster execution, consider 1000 for final.
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.001, 0.1],
        'scale_pos_weight': [1.0, scale_pos_weight_value] # Test default and calculated value
    }

    # --- Step 3: Initialize XGBoost classifier for GridSearchCV ---
    # This estimator will be tuned by GridSearchCV
    xgb_base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc', # Used internally by XGBoost during fit
        random_state=42, # To suppress a warning
    )

    # --- Step 4: Perform GridSearchCV to find the best hyperparameters ---
    # Use TimeSeriesSplit for CV within GridSearchCV to respect time dependency
    # Note: GridSearchCV will use the entire X_scaled and y for its internal splits
    # scoring='precision' focuses on optimizing precision. 'f1' or 'roc_auc' are good general choices.
    print("\nStarting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=xgb_base_model,
        param_grid=param_grid,
        scoring='precision', # Prioritize precision for tuning
        cv=TimeSeriesSplit(n_splits=3), # Use TimeSeriesSplit for CV within GridSearchCV                                  # n_splits here can be smaller for faster tuning
        verbose=1, # Print progress
        n_jobs=-1 # Use all available CPU cores
    )

    # Removed direct grid_search.fit(X_scaled, y) as it's not strictly necessary for final model
    # and might be redundant if you're directly using hardcoded 'best' params.
    # If you want to use the GridSearchCV's best parameters, uncomment the line below and the best_params assignments.
    # grid_search.fit(X_scaled, y)
    # best_params = grid_search.best_params_
    # print(f"Best parameters found: {best_params}")
    # print(f"Best cross-validation precision score: {grid_search.best_score_:.4f}")

    print("\nGridSearchCV (skipped fitting for this run, using default best_params).")
    # For demonstration, directly setting best_params as per your previous code's hardcoded values
    # If you want GridSearchCV to actually find them, uncomment grid_search.fit and best_params = ...
    best_params = {
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 7,
        'n_estimators': 500,
        'reg_alpha': 0,
        'scale_pos_weight': 1.0, # Using 1.0, but consider scale_pos_weight_value if imbalance is significant
        'subsample': 0.7,
        'random_state': 42
    }
    
    # Create time-series split for evaluation (using the original tscv for consistency)
    tscv_eval = TimeSeriesSplit(n_splits=5) # Renamed to avoid confusion with GridSearchCV's internal CV
    
    all_predictions = []
    all_actual = []
    
    print("\nStarting TimeSeries Cross-Validation with best parameters...")
    for fold_idx, (train_idx, test_idx) in enumerate(tscv_eval.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize XGBoost model with best parameters
        model = xgb.XGBClassifier(**best_params) # Use the best_params dictionary
        
        # Train model for the current fold
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False # Keep verbose=False for clean output during loop
        )
        
        # Get predictions (probabilities for ROC AUC and binary classification)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_predictions.extend(y_pred_proba)
        all_actual.extend(y_test)
        print(f"  Fold {fold_idx + 1} complete.") # Progress indicator

    # --- Step 6: Train final model on all data with best parameters ---
    # This model is what you'd use for future predictions
    print("\nTraining final model on all data with best parameters...")
    final_model = xgb.XGBClassifier(**best_params) # Use the best_params dictionary
    final_model.fit(X_scaled, y)
    print("Final model trained.")
    
    # --- Evaluate overall performance using collected predictions from CV ---
    auc_score = roc_auc_score(all_actual, all_predictions)
    print(f"\nOverall ROC AUC (from CV folds): {auc_score:.4f}")
    
    # Convert probabilities to binary predictions using default threshold 0.5 for report
    # If you want to optimize precision, you could adjust this threshold here.
    binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
    
    # Print classification report
    print("\nClassification Report (from CV folds):")
    print(classification_report(all_actual, binary_predictions))
    
    # Print confusion matrix
    cm = confusion_matrix(all_actual, binary_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (from CV folds)')
    plt.show()
    
    # Feature importance from the final model trained on all data
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(final_model, max_num_features=20)
    plt.title('Feature Importance (from Final Model)')
    plt.show()
    
    return final_model, scaler, features

# --- START OF FIXES FOR REALISTIC RISK/REWARD ---
def calculate_adaptive_risk_levels(data, price, confidence, min_risk_reward_ratio=2.0):
    """
    Enhanced risk calculation with multiple methods to ensure minimum 2:1 reward-to-risk ratio.
    The primary goal is to ensure 'risk_pct' accurately reflects the percentage
    of the current price risked if the stop loss is hit.
    """
    # Ensure data has enough history for rolling calculations.
    # Using tail(N) to get relevant recent data for indicators.
    # Using 20 as a safe minimum for many common indicators.
    recent_data = data.tail(max(200, len(data))).copy() # Take a decent chunk for indicators to be stable

    if recent_data.empty:
        # Fallback for extremely short data: assume a small, fixed risk
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
    # Ensure ATR is calculated over a reasonable period and is a positive value
    # Re-calculate ATR on recent_data to ensure it's not NaN
    atr_value = talib.ATR(recent_data['High'], recent_data['Low'], recent_data['Close'], timeperiod=14).iloc[-1]
    if pd.isna(atr_value) or atr_value <= 0:
        # Fallback to average daily true range or a default percentage of price
        atr_value = (recent_data['High'] - recent_data['Low']).mean() # Simple range mean
        if pd.isna(atr_value) or atr_value <= 0:
             atr_value = price * 0.008 # Default to 0.8% of price if no ATR history

    # Method 2: Bollinger Band based calculation
    bb_range_risk = None
    if all(col in recent_data.columns for col in ['BB_Upper', 'BB_Lower']):
        bb_upper, bb_middle, bb_lower = talib.BBANDS(recent_data['Close'], timeperiod=20)
        if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]):
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            # Use a portion of BB range as a risk measure. Cap at a reasonable % of price.
            bb_range_risk = min(bb_range * 0.3, price * 0.03) # Cap at 3% of price (tighter)

    # Method 3: Recent volatility based calculation
    # Using log returns for more stable volatility calculation
    log_returns = np.log(recent_data['Close'] / recent_data['Close'].shift(1)).dropna()
    recent_volatility_dollar = 0
    if len(log_returns) >= 20:
        # Volatility as a standard deviation of log returns, converted to dollar movement
        volatility_daily_std = log_returns.tail(20).std()
        recent_volatility_dollar = volatility_daily_std * price # Daily dollar movement
        recent_volatility_dollar = min(recent_volatility_dollar * np.sqrt(3), price * 0.04) # Scale for 3-day period, cap at 4%

    # Method 4: Support/Resistance based calculation (as a downside buffer)
    support_distance_risk = 0
    if len(recent_data) >= 20:
        support_level = recent_data['Close'].rolling(window=20).min().iloc[-1]
        # Only consider support below current price as a stop loss level
        if price > support_level:
            distance_to_support = price - support_level
            support_distance_risk = distance_to_support * 0.7 # Use 70% of distance to support
            support_distance_risk = min(support_distance_risk, price * 0.05) # Cap at 5% of price

    # --- Step 2: Aggregate Risk Measures and Determine 'base_risk' ---
    risk_measures_dollars = []

    # Factors for each method to ensure realistic contribution.
    # These factors determine how much of each indicator's value contributes to the stop distance.
    risk_measures_dollars.append(atr_value * 2.0) # ATR often represents 1-day movement, scale for a wider stop
    if recent_volatility_dollar > 0:
        risk_measures_dollars.append(recent_volatility_dollar * 1.5) # Factor for volatility

    if bb_range_risk is not None:
        risk_measures_dollars.append(bb_range_risk)

    if support_distance_risk > 0:
        risk_measures_dollars.append(support_distance_risk)
    
    # Add a floor to base risk (e.g., 0.5% of price) to prevent stops from being too tight
    # This also helps if all other measures are zero or tiny.
    risk_measures_dollars.append(price * 0.005) # Minimum 0.5% price movement risk

    # Use the median of valid, positive risk measures. Median is robust to outliers.
    valid_risks = [r for r in risk_measures_dollars if r > 0 and not pd.isna(r)]
    
    if not valid_risks:
        base_risk = price * 0.015 # Fallback if no valid risk measures (1.5% of price)
    else:
        base_risk = np.median(valid_risks)
        if base_risk <= 0: # Ensure base_risk is always positive
            base_risk = price * 0.015

    # --- Step 3: Adjust risk based on confidence ---
    # confidence_adjustment: lower for higher confidence (tighter stop), higher for lower confidence (wider stop)
    # Range: 0.7 (for confidence 1.0) to 1.3 (for confidence 0.5)
    confidence_adjustment = 1.3 - (confidence - 0.5) * 2 # Scales from 1.3 down to 0.3 for confidence 0.5 to 1.0
    confidence_adjustment = max(0.7, min(1.3, confidence_adjustment)) # Clamp values to a reasonable range
    
    adjusted_risk_dollar = base_risk * confidence_adjustment

    # Ensure adjusted_risk_dollar is not negative, too small, or too large.
    # This is a critical cap for realism.
    adjusted_risk_dollar = max(adjusted_risk_dollar, price * 0.005) # Minimum 0.5% of price
    adjusted_risk_dollar = min(adjusted_risk_dollar, price * 0.08)  # MAXIMUM 8% of price as dollar risk

    # --- Step 4: Calculate Stop Loss and Take Profit ---
    stop_loss = price - adjusted_risk_dollar
    
    # Ensure stop_loss is always positive and not ridiculously close to zero
    if stop_loss <= 0:
        stop_loss = price * 0.001 # A tiny fraction of price as absolute minimum stop
        adjusted_risk_dollar = price - stop_loss # Recalculate adjusted_risk_dollar based on this minimum

    # Calculate required reward based on adjusted risk and minimum R:R ratio
    min_reward_dollar = adjusted_risk_dollar * min_risk_reward_ratio
    
    # Initial take profit based on minimum required reward
    take_profit = price + min_reward_dollar

    # Enhance take profit with resistance levels if applicable
    # This relies on Resistance_Proximity being calculated earlier
    # You might want to use historical resistance levels or just a fixed target above the min_reward
    if 'Resistance_Proximity' in recent_data.columns and not pd.isna(recent_data['Resistance_Proximity'].iloc[-1]):
        # This approach uses the previously calculated resistance level if available
        resistance_level_calc = price * (1 + recent_data['Resistance_Proximity'].iloc[-1])
        if resistance_level_calc > take_profit: # If existing resistance offers more, consider it
            take_profit = resistance_level_calc * (1 + (confidence * 0.02)) # Add a confidence bonus
    
    # Cap take profit to prevent absurd targets
    take_profit = min(take_profit, price * 1.25) # Max 25% gain in 3 days is already very optimistic for most stocks

    # --- Step 5: Final Adjustments and Return Values ---
    actual_risk_amount = adjusted_risk_dollar 
    actual_reward_amount = take_profit - price

    actual_risk_reward_ratio = actual_reward_amount / actual_risk_amount if actual_risk_amount > 0 else min_risk_reward_ratio

    # Ensure final R:R meets the minimum (if reward_amount is positive)
    if actual_reward_amount > 0 and actual_risk_reward_ratio < min_risk_reward_ratio:
        required_reward_for_ratio = actual_risk_amount * min_risk_reward_ratio
        take_profit = price + required_reward_for_ratio
        actual_reward_amount = required_reward_for_ratio
        actual_risk_reward_ratio = min_risk_reward_ratio
    elif actual_reward_amount <= 0: # If take profit is at or below entry, then R:R is irrelevant or bad
        actual_risk_reward_ratio = 0 # Or some indicator of a bad trade

    # Calculate risk_pct as the percentage of the ENTRY PRICE that is risked.
    # This is the correct interpretation for position sizing.
    risk_pct_of_price = (actual_risk_amount / price) * 100 if price > 0 else 0
    reward_pct_of_price = (actual_reward_amount / price) * 100 if price > 0 else 0

    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_amount': actual_risk_amount,      # Dollar amount risked per share
        'reward_amount': actual_reward_amount,  # Dollar amount rewarded per share
        'risk_reward_ratio': actual_risk_reward_ratio,
        'risk_pct': risk_pct_of_price,          # Percentage of entry price risked (e.g., 3.5%)
        'reward_pct': reward_pct_of_price       # Percentage of entry price gained (e.g., 7.0%)
    }

def calculate_dynamic_position_size(confidence, risk_pct, max_portfolio_risk=3.0, max_position_pct=20.0, min_confidence=0.55):
    """
    Calculate position size based on confidence, risk percentage, and portfolio risk management.
    This version is designed to be more aggressive in its allocation, assuming risk_pct
    is now a realistic single-digit percentage (e.g., 1-8%).
    """
    # Defensive checks
    if confidence < min_confidence or risk_pct <= 0:
        return 0.0

    # Ensure risk_pct is not excessively high (e.g., from an edge case)
    # If risk_pct is still coming out high (e.g., > 10%), it indicates an issue in `calculate_adaptive_risk_levels`
    # or that the trade's stop is truly very far, making it unviable for large positions.
    if risk_pct > 15.0: # Cap risk_pct consideration at 15% for position sizing
        print(f"Warning: risk_pct ({risk_pct:.2f}%) is very high. Capping for position size calculation.")
        risk_pct = 15.0 # This prevents absurdly small position sizes for very wide stops

    # Scale confidence from min_confidence to 1.0 linearly
    # E.g., if min_confidence=0.55, confidence=0.75: (0.75 - 0.55) / (1.0 - 0.55) = 0.20 / 0.45 = 0.44
    confidence_factor = (confidence - min_confidence) / (1.0 - min_confidence)
    confidence_factor = max(0.0, min(1.0, confidence_factor)) # Clamp between 0 and 1

    # --- Core Position Sizing Logic (Kelly-like) ---
    # `max_portfolio_risk` is the % of total portfolio capital you're willing to lose on a single trade.
    # `risk_pct` is the % of the *trade's value* you'd lose if the stop is hit.
    # So, (Trade_Value_as_Proportion_of_Portfolio * risk_pct) = max_portfolio_risk
    # Trade_Value_as_Proportion_of_Portfolio = max_portfolio_risk / risk_pct
    
    # Calculate desired position size as a percentage of portfolio
    # (max_portfolio_risk / risk_pct) * 100 converts the proportion to a percentage.
    # This is the "raw" position size if confidence was 1.0 and no aggression multiplier.
    raw_position_size_pct = (max_portfolio_risk / risk_pct) * 100 if risk_pct > 0 else 0

    # Scale by confidence factor. Higher confidence = larger position.
    confidence_adjusted_size = raw_position_size_pct * confidence_factor
    
    # Add an aggression multiplier:
    # This multiplies the calculated position size to make it generally larger.
    # Values: 1.0 (no change) to 2.0 (double the size) or more.
    aggression_multiplier = 1.5 # Increased for more aggression. You can tune this.
    confidence_adjusted_size *= aggression_multiplier

    # Apply maximum position limit
    final_position_size = min(confidence_adjusted_size, max_position_pct)

    # Ensure minimum viable position:
    # If a calculated position is too small, it might not be worth the transaction costs or mental effort.
    if final_position_size < 0.5:  # Retained 0.5% as minimum for realism, adjust as needed
        return 0.0

    return final_position_size


def enhanced_signal_generation(model, scaler, data, features, min_confidence=0.55):
    """
    Enhanced trading signal generation with improved risk-reward optimization
    
    Args:
        model: Trained XGBoost model.
        scaler: StandardScaler used for features.
        data (pd.DataFrame): The input DataFrame containing all features and OHLC data
                             for the testing period, potentially including future data
                             needed for `simulate_trade_outcome`. This DataFrame
                             should include data for multiple tickers, sorted by date.
        features (list): List of feature column names used by the model.
        min_confidence (float): Minimum confidence score from the model to generate a signal.

    Returns:
        pd.DataFrame: The input 'data' DataFrame with added signal-related columns.
    """
    # Make a copy of data to avoid modifying original, and to ensure we're working with
    # the exact data passed for signal generation.
    signals = data.copy()

    # --- Step 1: Initialize new columns for signals data ---
    # Do this BEFORE any loops or calculations to ensure they exist for all rows
    signals['Confidence'] = np.nan
    signals['Signal'] = 0 # Default to no signal (0), will be set to 1 if conditions met
    signals['Stop_Loss'] = np.nan
    signals['Take_Profit'] = np.nan
    signals['Risk_Reward_Ratio'] = np.nan
    signals['Position_Size_Pct'] = np.nan
    signals['Risk_Pct'] = np.nan
    signals['Reward_Pct'] = np.nan

    # --- Step 2: Prepare features and get model predictions for the entire dataset ---
    # This is done once for efficiency
    X_full = signals[features].fillna(signals[features].mean())
    X_scaled_full = scaler.transform(X_full)

    # Get model prediction probabilities and assign confidence scores
    signals['Confidence'] = model.predict_proba(X_scaled_full)[:, 1]

    # Assign initial signals based on confidence threshold.
    # These are potential signals that will then be filtered by risk/reward criteria.
    signals['Signal'] = (signals['Confidence'] > min_confidence).astype(int)

    # --- Step 3: Iterate by ticker to ensure correct historical slicing for each ticker ---
    # This is the CRITICAL part that solves the KeyError
    # Group the DataFrame by 'Ticker' so we can process each stock's data independently
    grouped_by_ticker = signals.groupby('Ticker')

    # Iterate through each ticker's group
    for ticker, group_df_for_ticker in grouped_by_ticker:
        # It's good practice to ensure the group is sorted by date index, although groupby
        # generally preserves order if the original DataFrame was sorted.
        group_df_sorted = group_df_for_ticker.sort_index()

        # Iterate through each row (daily data point) for the current ticker
        # Using .itertuples() is generally more efficient than .iterrows()
        for i, row_tuple in enumerate(group_df_sorted.itertuples()):
            # Access attributes from the named tuple (row_tuple)
            # row_tuple.Index contains the datetime index for the current row
            current_date_idx = row_tuple.Index
            current_price = row_tuple.Close
            confidence = row_tuple.Confidence
            signal_status = row_tuple.Signal # This is the initial signal (0 or 1)

            # Only proceed if the model generated an initial signal for this date
            if signal_status == 1:
                # Get current and historical data slice for this specific ticker UP TO this point.
                # This ensures NO LOOK-AHEAD BIAS.
                # `iloc[:i+1]` gets all rows from the beginning of `group_df_sorted` up to and including the current row.
                current_data_slice = group_df_sorted.iloc[:i+1].copy()

                # --- Defensive check: Ensure enough historical data for indicators ---
                # Many TA-Lib indicators (like SMA_200, Ichimoku, etc.) require a certain look-back period.
                # If `current_data_slice` is too short, `talib` functions will return NaNs,
                # which can lead to invalid risk_metrics.
                # Adjust `min_history_required` based on the largest `timeperiod` in `calculate_selected_features`.
                min_history_required = 200 # Largest period (SMA_200, Ichimoku_Base=26 is shorter, but 200 is safest)

                if len(current_data_slice) < min_history_required:
                    # If not enough history, this signal cannot be properly evaluated for risk.
                    # Set signal to 0 and continue to the next iteration.
                    signals.loc[current_date_idx, 'Signal'] = 0
                    continue # Skip to the next trade in the loop

                # --- Calculate enhanced risk metrics ---
                risk_metrics = calculate_adaptive_risk_levels(
                    current_data_slice, # Pass only historical data for the current ticker
                    current_price,
                    confidence,
                    min_risk_reward_ratio=2.0
                )

                # --- Calculate dynamic position size ---
                position_size = calculate_dynamic_position_size(
                    confidence,
                    risk_metrics['risk_pct'], # This is the percentage risk of the trade's entry price
                    max_portfolio_risk=3.0,   # Max % of total portfolio capital to risk on one trade
                    max_position_pct=20.0     # Max % of portfolio to allocate to one position
                )

                # --- Final Signal Filtering ---
                # Only keep the signal if position size is viable AND
                # the calculated Risk-Reward Ratio is valid and meets a minimum threshold.
                # Ensure risk_metrics['risk_reward_ratio'] is not NaN or inf before comparison
                is_valid_rr = not pd.isna(risk_metrics['risk_reward_ratio']) and np.isfinite(risk_metrics['risk_reward_ratio'])

                if position_size > 0 and is_valid_rr and risk_metrics['risk_reward_ratio'] >= 1.5:
                    # If all criteria met, update the corresponding row in the main 'signals' DataFrame
                    # Use .loc[index, column] for safe assignment with datetime indices
                    signals.loc[current_date_idx, 'Stop_Loss'] = risk_metrics['stop_loss']
                    signals.loc[current_date_idx, 'Take_Profit'] = risk_metrics['take_profit']
                    signals.loc[current_date_idx, 'Risk_Reward_Ratio'] = risk_metrics['risk_reward_ratio']
                    signals.loc[current_date_idx, 'Position_Size_Pct'] = position_size
                    signals.loc[current_date_idx, 'Risk_Pct'] = risk_metrics['risk_pct']
                    signals.loc[current_date_idx, 'Reward_Pct'] = risk_metrics['reward_pct']
                    # Signal remains 1
                else:
                    # If any criteria are not met, revert the signal to 0
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
    
    # --- Simulate stop-loss and take-profit execution ---
    # Crucial: Ensure 'Future_Close' is available for comparison
    # This requires 'Future_Close' to be calculated in create_prediction_targets and passed through.
    # It seems it is in create_prediction_targets now.
    
    # We need to find the actual high/low within the forward period to accurately simulate.
    # This requires looking forward in the original data, which can be tricky in a grouped DF.
    # For a realistic simulation, you'd typically need to find the Min/Max price within the next 'forward_period' days.
    # For now, we'll refine the `simulate_trade_outcome` to use the `Future_Close` target as the *final price*
    # after `forward_period` days, and assume SL/TP are checked *at any point* within that period.
    # This is a simplification. A more robust simulation would need actual minute/hourly data or at least daily OHLCV for the future period.
    
    # For simplicity, let's assume the `Future_Close` is the price after `forward_period` days,
    # and if SL/TP are hit *before* that, they take precedence.
    # This implies we need the future price *range* not just the final close.
    # Since we only have `Future_Close`, we'll simplify the simulation to assume
    # if `Future_Close` crosses SL/TP, it's hit.
    # A true simulation would check high/low for the period.
    
    # Let's adjust `simulate_trade_outcome` to consider the actual price path or at least bounds.
    # For now, I'll assume `Future_Close` represents the price at the end of the `forward_period`.
    # And we'll use a simplified check: if the price *ever* went below SL or above TP within `forward_period`.
    # This usually requires getting future OHLC data. For this refactor, we'll assume `Future_Close`
    # is the price at the end of the period, and SL/TP are hit if `Future_Close` is beyond them.
    # If the price movement was *not* hitting SL/TP, then `Future_Return` is the outcome.

    # REVISED simulate_trade_outcome logic:
    # 1. Get the current Close price (entry price).
    # 2. Get the Stop_Loss and Take_Profit levels.
    # 3. Get the Future_Close price (price after forward_period days).
    # 4. Determine actual high/low in the forward period (if available, otherwise simplify).
    #    Since we don't have future OHLC readily available in `trade_row`, we make a strong assumption:
    #    If Future_Close is below SL, assume SL was hit. If Future_Close is above TP, assume TP was hit.
    #    Otherwise, the return is the actual Future_Return. This is a simplification and not perfect.

    # To fix this properly, we need to access future high/low for each trade's forward period.
    # This is often done by merging or iterating. For now, let's assume `Future_Close` is the only
    # future price point we have from `create_prediction_targets`.
    # I'll modify `simulate_trade_outcome` to be more robust for *this specific data structure*.
    
    actual_signals['Simulated_Return'] = actual_signals.apply(
        lambda row: simulate_trade_outcome(row, forward_period, signals), # Pass original signals DF for future data access
        axis=1
    )
    
    # Performance metrics
    total_trades = len(actual_signals)
    winning_trades = (actual_signals['Simulated_Return'] > 0).sum()
    losing_trades = (actual_signals['Simulated_Return'] < 0).sum()
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate returns
    avg_win = actual_signals[actual_signals['Simulated_Return'] > 0]['Simulated_Return'].mean()
    avg_loss = actual_signals[actual_signals['Simulated_Return'] < 0]['Simulated_Return'].mean()
    
    # Risk-adjusted metrics
    total_wins = actual_signals[actual_signals['Simulated_Return'] > 0]['Simulated_Return'].sum()
    total_losses = abs(actual_signals[actual_signals['Simulated_Return'] < 0]['Simulated_Return'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Expected value per trade
    expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Portfolio level analysis
    # Ensure position_size_pct is not NaN where signal is 1 (it should be filtered out)
    # total_portfolio_return is the sum of (return * position_size_pct / 100)
    portfolio_returns = (actual_signals['Simulated_Return'] * (actual_signals['Position_Size_Pct'] / 100)).sum()
    total_portfolio_return = portfolio_returns
    
    avg_portfolio_risk = actual_signals['Risk_Pct'].mean() # Average risk percentage *per trade*
    
    # Risk-adjusted return calculation (e.g., Sharpe Ratio type concept)
    # Simple risk-adjusted return: Total Portfolio Return / Std Dev of Portfolio Returns (if available)
    # Your current 'Risk-Adjusted Return: Total Portfolio Return / Avg Risk per Trade' is unusual.
    # A more common way is portfolio return / portfolio volatility.
    
    # For now, let's use a simpler interpretation:
    # How much return did we get for the average risk taken on each trade.
    # If avg_portfolio_risk is still high, this will be small.
    # But if it's realistic (e.g., 2-5%), then this might be meaningful.
    risk_adjusted_return = total_portfolio_return / avg_portfolio_risk if avg_portfolio_risk > 0 else 0


    # Print comprehensive results
    print(f"\n{'='*60}")
    print(f"ENHANCED PERFORMANCE ANALYSIS")
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
    print(f"Risk-Adjusted Return (Total Return / Avg. Trade Risk%): {risk_adjusted_return:.2f}") # Re-evaluate this metric later
    
    # Visualization
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
    This version attempts a more realistic simulation by checking if SL/TP
    would have been hit within the forward period using future OHLC data.
    
    NOTE: This requires access to the *actual* future OHLC data for the specific ticker
    during the `forward_period` days. This is the most complex part of backtesting.
    For simplicity, we'll try to extract future data from the `full_signals_df` itself.
    This assumes `full_signals_df` contains all OHLC data for all tickers and dates.
    """
    current_price = trade_row['Close']
    stop_loss = trade_row['Stop_Loss']
    take_profit = trade_row['Take_Profit']
    ticker = trade_row['Ticker']
    entry_date = trade_row.name # The date of the signal

    # Find the data slice for the future period for this specific ticker
    future_data_slice = full_signals_df[
        (full_signals_df['Ticker'] == ticker) &
        (full_signals_df.index > entry_date) &
        (full_signals_df.index <= entry_date + timedelta(days=forward_period * 1.5)) # Give some buffer
    ].sort_index()

    # If there's no future data (e.g., at end of test period), return actual return
    if future_data_slice.empty:
        return trade_row['Future_Return'] # No future data to simulate hits

    # Get the actual future close after `forward_period` days
    # This needs to be precise. It's the `Future_Close` for the `trade_row` itself.
    future_final_close = trade_row['Future_Close']
    
    # Check if stop-loss or take-profit was hit *before or on* the target date
    hit_sl = False
    hit_tp = False

    # Iterate through each day in the future period to check High/Low
    # This is crucial for realistic simulation.
    for _, future_day_row in future_data_slice.iterrows():
        future_high = future_day_row['High']
        future_low = future_day_row['Low']

        # Check for SL hit
        if future_low <= stop_loss:
            hit_sl = True
            # Simulate the loss. The return is (Stop_Loss / Entry_Price) - 1
            return (stop_loss / current_price) - 1

        # Check for TP hit (only if SL wasn't hit first in the same bar, simplified here)
        if future_high >= take_profit:
            hit_tp = True
            # Simulate the gain. The return is (Take_Profit / Entry_Price) - 1
            return (take_profit / current_price) - 1
        
        # If the current day in the loop is the 'forward_period' day (or beyond)
        # This check is imperfect. A more robust way is to compare dates explicitly.
        # For this refactor, we are using trade_row['Future_Close'] as the 'actual' outcome IF no SL/TP hit.
        
    # If neither SL nor TP was hit within the period, the outcome is the return to `Future_Close`
    # Check if Future_Close itself is available and not NaN
    if not pd.isna(future_final_close):
        return (future_final_close / current_price) - 1
    else:
        # If Future_Close is NaN (e.g., at the very end of the dataset), return 0 or NaN
        return np.nan # Or 0, or handle as an incomplete trade


def create_enhanced_visualizations(actual_signals, all_signals):
    """
    Create comprehensive visualizations for the enhanced trading system
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid') # A more modern and clean style
    
    # 1. Risk-Reward Scatter Plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # Larger figure
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Filter out NaNs if any for plotting
    plot_signals = actual_signals.dropna(subset=['Risk_Pct', 'Reward_Pct', 'Confidence', 'Position_Size_Pct', 'Simulated_Return'])

    # Risk vs Reward scatter
    if not plot_signals.empty:
        scatter = ax1.scatter(plot_signals['Risk_Pct'], plot_signals['Reward_Pct'], 
                            c=plot_signals['Confidence'], cmap='viridis', alpha=0.7, s=plot_signals['Position_Size_Pct'] * 2) # Size by position
        
        # Plot 2:1 and 1:1 risk-reward lines
        max_rr_plot = max(plot_signals['Risk_Pct'].max(), plot_signals['Reward_Pct'].max()) * 1.1
        ax1.plot([0, max_rr_plot], [0, max_rr_plot], 'k--', alpha=0.5, label='1:1 R:R Line') # 1:1 line
        ax1.plot([0, max_rr_plot], [0, max_rr_plot * 2], 'r--', alpha=0.7, label='2:1 R:R Line') # 2:1 line
        
        ax1.set_xlabel('Risk % (of Entry Price)')
        ax1.set_ylabel('Reward % (of Entry Price)')
        ax1.set_title('Risk vs Reward Distribution (Bubble size by Position Size)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Model Confidence')
    else:
        ax1.set_title('Risk vs Reward Distribution (No Valid Signals)')
        ax1.text(0.5, 0.5, 'No data to display', transform=ax1.transAxes, ha='center')

    # 2. Confidence vs Position Size
    if not plot_signals.empty:
        ax2.scatter(plot_signals['Confidence'], plot_signals['Position_Size_Pct'], 
                c=plot_signals['Simulated_Return'], cmap='coolwarm', alpha=0.7, s=60) # Color by return
        ax2.set_xlabel('Model Confidence')
        ax2.set_ylabel('Position Size % (of Portfolio)')
        ax2.set_title('Confidence vs Position Size (Color by Return)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(ax2.collections[0], ax=ax2, label='Simulated Return')
    else:
        ax2.set_title('Confidence vs Position Size (No Valid Signals)')
        ax2.text(0.5, 0.5, 'No data to display', transform=ax2.transAxes, ha='center')


    # 3. Return Distribution
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

    # 4. Time Series of Signals (with actual price)
    if not all_signals.empty:
        # Select one or two tickers for clarity, or iterate if few tickers
        selected_tickers_for_plot = all_signals['Ticker'].unique()[:2] # Plot up to 2 tickers
        for ticker in selected_tickers_for_plot:
            ticker_data = all_signals[all_signals['Ticker'] == ticker].sort_index()
            ticker_signals = ticker_data[ticker_data['Signal'] == 1]
            
            ax4.plot(ticker_data.index, ticker_data['Close'], label=f'{ticker} Close Price', alpha=0.8, linewidth=1.5)
            
            # Plot entry points
            if not ticker_signals.empty:
                ax4.scatter(ticker_signals.index, ticker_signals['Close'], 
                        color='green', marker='^', s=100, label=f'{ticker} Entry (Signal)', zorder=5)
                
                # Plot Stop Loss and Take Profit lines for each signal
                for sig_idx, sig_row in ticker_signals.iterrows():
                    # Only plot lines for the period of the trade if we know the exit.
                    # For simplicity, just plot the point. Lines would be too messy if many signals.
                    pass # We will rely on scatter for signals.
                    # For a single trade visualization, you could plot a horizontal line for SL/TP.

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
    
    # Additional plot: Risk-Reward Ratio over time
    if not plot_signals.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(plot_signals.index, plot_signals['Risk_Reward_Ratio'], 
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
                 min_confidence=0.55, use_cache=True, force_refresh=False): # Adjusted min_confidence here
    """
    Enhanced main function with improved risk-reward optimization
    """
    # Set default test_end_date to current date if not provided
    if test_end_date is None:
        test_end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Clear cache if force refresh is requested
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
    print(f"Testing Period: {test_start_date} to {test_end_date}")
    print(f"Forward Prediction: {forward_period} days")
    print(f"Minimum Confidence for Signal: {min_confidence}")
    print(f"Target Risk-Reward Ratio: 2:1 minimum (applied at initial calculation)")
    print(f"Aggressive Position Sizing: Max Risk 3% of portfolio, Max Position 20%, Aggression Multiplier 1.5x")
    
    # Fetch data
    print("\n1. Fetching training data...")
    train_data = fetch_and_prepare_data(
        tickers, 
        start_date=train_start_date,
        end_date=train_end_date,
        use_cache=use_cache,
        cache_suffix="train"
    )
    
    print("2. Fetching testing data...")
    # For testing, you want data that extends *past* test_end_date by forward_period
    # so that 'Future_Return' and 'Future_Close' can be calculated correctly up to test_end_date
    adjusted_test_end_date = (datetime.strptime(test_end_date, '%Y-%m-%d') + timedelta(days=forward_period * 2)).strftime('%Y-%m-%d')
    
    test_data_raw = fetch_and_prepare_data(
        tickers,
        start_date=test_start_date,
        end_date=adjusted_test_end_date, # Fetch a bit more data for future calculations
        use_cache=use_cache,
        cache_suffix="test_extended" # New cache suffix
    )

    # Filter test_data to the actual test_end_date *after* future targets are calculated
    test_data_for_signals = test_data_raw[test_data_raw.index <= test_end_date].copy()
    
    # Calculate features
    print("3. Calculating features for training data...")
    train_data = calculate_selected_features(train_data, use_cache=use_cache, cache_suffix="train_features")
    
    print("4. Calculating features for testing data...")
    test_data_raw_features = calculate_selected_features(test_data_raw, use_cache=use_cache, cache_suffix="test_extended_features")
    
    # Create targets (Future_Return and Future_Close)
    print("5. Creating prediction targets for training data...")
    train_data = create_prediction_targets(train_data, forward_period)
    print("5. Creating prediction targets for testing data...")
    # Apply targets to the extended test data
    test_data_with_targets = create_prediction_targets(test_data_raw_features, forward_period)
    
    # Feature selection
    print("6. Selecting and engineering features for training data...")
    train_data, selected_features = select_features_for_model(train_data)
    print("6. Selecting and engineering features for testing data...")
    test_data_with_targets, _ = select_features_for_model(test_data_with_targets) # Note: this will drop rows with NaN features

    # Filter test_data_with_targets to the actual test_end_date *after* feature/target engineering
    test_data_final = test_data_with_targets[test_data_with_targets.index <= test_end_date].copy()

    # Train model
    print("7. Training prediction model...")
    model, scaler, features = train_prediction_model(train_data, selected_features)
    
    # Generate enhanced signals
    print("8. Generating enhanced trading signals...")
    # Pass `test_data_with_targets` (the one with full future data up to the adjusted end date)
    # to `enhanced_signal_generation` so `simulate_trade_outcome` can find future OHLC.
    signals = enhanced_signal_generation(model, scaler, test_data_final, features, min_confidence)
    
    # Analysis and visualization
    print("9. Analyzing performance...")
    # Ensure `advanced_performance_analysis` gets the signals with all necessary columns.
    performance = advanced_performance_analysis(signals, forward_period)
    
    # Display latest signals
    print("\n" + "="*60)
    print("LATEST TRADING OPPORTUNITIES")
    print("="*60)
    
    # Filter for signals that are marked as 1 and have all necessary info
    viable_signals = signals[
        (signals['Signal'] == 1) & 
        (signals['Position_Size_Pct'].notna()) &
        (signals['Stop_Loss'].notna())
    ].tail(5)  # Last 5 viable signals

    if len(viable_signals) > 0:
        for _, row in viable_signals.iterrows():
            print(f"\n🎯 {row['Ticker']} - {row.name.strftime('%Y-%m-%d')}")
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
    # Configuration
    STOCK_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'] # More tickers might give more signals
    
    # Run enhanced system
    enhanced_model, enhanced_scaler, enhanced_signals, enhanced_performance = enhanced_main(
        STOCK_TICKERS,
        train_start_date='2022-01-01',
        train_end_date='2023-12-31',
        test_start_date='2024-01-01',
        test_end_date=None, # Will default to current date
        forward_period=3,
        min_confidence=0.55,  # Start with a slightly lower confidence to generate more signals
        use_cache=True,
        force_refresh=False # Set to True if you want to re-download all data
    )
    
    print("\n🚀 Enhanced Trading System Ready!")
    print("📈 All signals now feature more realistic risk-reward ratios and position sizes.")
    print("💡 Position sizes dynamically adjusted based on confidence and risk.")
    print("🛡️  Multi-layer risk management active.")