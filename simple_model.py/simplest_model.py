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

# --- Caching Configuration ---
CACHE_DIR = 'crypto_cache'
DATA_DIR = 'processed_crypto_data' # For storing processed (feature-engineered) data

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def get_cache_filename(base_name, data_type='raw'):
    """Generate a cache filename for a given base name and data type."""
    return os.path.join(CACHE_DIR, f"{base_name}_{data_type}.pkl")

def save_data_to_cache(data, base_name, data_type='raw'):
    """Save data to a cache file using pickle."""
    ensure_directories()
    filename = get_cache_filename(base_name, data_type)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    # print(f"Saved {data_type} data for {base_name} to cache: {filename}")

def load_data_from_cache(base_name, data_type='raw', max_age_hours=24):
    """
    Load data from a cache file if it exists and is not too old.
    Returns the data if valid, None otherwise.
    """
    filename = get_cache_filename(base_name, data_type)
    if os.path.exists(filename):
        file_time = datetime.fromtimestamp(os.path.getmtime(filename))
        age = datetime.now() - file_time
        if age < timedelta(hours=max_age_hours):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            # print(f"Loaded {data_type} data for {base_name} from cache: {filename} (age: {age})")
            return data
        else:
            print(f"Cached {data_type} data for {base_name} is too old ({age}). Re-fetching.")
    return None

def save_processed_data(data, filename):
    """Save processed (e.g., feature-engineered) data to a designated directory."""
    ensure_directories()
    filepath = os.path.join(DATA_DIR, filename)
    data.to_pickle(filepath)
    print(f"Saved processed data to {filepath}")

def load_processed_data(filename):
    """Load processed data from a designated directory."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        data = pd.read_pickle(filepath)
        print(f"Loaded processed data from {filepath}")
        return data
    return None

# Function to fetch and prepare data
def fetch_and_prepare_data(tickers, period='1y', interval='3d', start_date=None, end_date=None, force_download=False):
    """
    Fetch historical data for cryptocurrencies and calculate basic features.
    Includes caching functionality.
    """
    all_data_list = []
    
    # Create a unique cache key based on tickers, period, and interval
    # Note: start_date/end_date are less reliable for YF as it prioritizes period
    cache_base_name = f"yf_data_{'_'.join(tickers)}_{period}_{interval}"
    
    if not force_download:
        cached_all_data = load_data_from_cache(cache_base_name, data_type='raw_combined')
        if cached_all_data is not None:
            # If using start_date/end_date filter, apply it after loading
            if start_date:
                cached_all_data = cached_all_data[cached_all_data.index.get_level_values('Date') >= start_date]
            if end_date:
                cached_all_data = cached_all_data[cached_all_data.index.get_level_values('Date') <= end_date]
            print(f"Loaded combined raw data from cache for {', '.join(tickers)}")
            return cached_all_data

    print(f"Fetching raw data for {', '.join(tickers)} from Yahoo Finance...")
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        try:
            crypto = yf.Ticker(ticker)
            # Use period and interval for fetching directly
            data = crypto.history(period=period, interval=interval)
            
            if data.empty:
                print(f"  No data fetched for {ticker}. Skipping.")
                continue

            # Add ticker column
            data['Ticker'] = ticker
            
            # Basic price and volume features - calculate before concat to avoid NaNs across tickers
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Daily_Return'] = data['Close'].pct_change()
            data['Volume_Change'] = data['Volume'].pct_change()
            data['Price_Volume_Ratio'] = data['Close'] / (data['Volume'] + 1e-9) # Add small epsilon to prevent div by zero
            
            all_data_list.append(data)
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}. Skipping.")
            continue
    
    if not all_data_list:
        print("No data fetched for any ticker. Returning empty DataFrame.")
        return pd.DataFrame()

    all_data = pd.concat(all_data_list, axis=0)
    
    # --- Set MultiIndex (Date, Ticker) ---
    # Convert index name to 'Date' if it's default 'Datetime' for clarity
    if all_data.index.name != 'Date':
        all_data.index.name = 'Date'
    all_data = all_data.set_index('Ticker', append=True)
    all_data = all_data.sort_index() # Sort by MultiIndex for efficient slicing

    # Drop NaNs after calculating basic features
    all_data = all_data.dropna()

    # Apply date filtering here if it's not handled by YF period/interval directly
    if start_date:
        all_data = all_data[all_data.index.get_level_values('Date') >= start_date]
    if end_date:
        all_data = all_data[all_data.index.get_level_values('Date') <= end_date]
    
    save_data_to_cache(all_data, cache_base_name, data_type='raw_combined')
    print(f"Successfully fetched and cached data for {', '.join(tickers)}")
    return all_data

def calculate_selected_features(data, cache_key_suffix="", force_recalculate=False):
    """
    Calculate selected technical indicators. Includes caching functionality.
    """
    # Create a unique cache key based on the input data's date range and ticker list
    # Use the first and last dates in the index, and the unique tickers
    if not data.empty:
        start_date = data.index.get_level_values('Date').min().strftime('%Y-%m-%d')
        end_date = data.index.get_level_values('Date').max().strftime('%Y-%m-%d')
        tickers_str = '_'.join(sorted(data.index.get_level_values('Ticker').unique()))
        feature_cache_name = f"features_{tickers_str}_{start_date}_{end_date}_{cache_key_suffix}"
    else:
        print("Input data is empty for feature calculation. Skipping.")
        return pd.DataFrame()

    if not force_recalculate:
        cached_features = load_data_from_cache(feature_cache_name, data_type='features')
        if cached_features is not None:
            print(f"Loaded features from cache: {feature_cache_name}")
            return cached_features

    print(f"Calculating features for {len(data.index.get_level_values('Ticker').unique())} tickers from {start_date} to {end_date}...")
    grouped = data.groupby(level='Ticker')
    result_dfs = []
    
    for ticker, group_df in grouped:
        df = group_df.copy()
        
        # Ensure that df.index is a DatetimeIndex for TALIB functions
        # For a MultiIndex dataframe grouped by ticker, the group_df's index
        # will actually be a DatetimeIndex (the first level of the original MultiIndex).
        # So, no explicit `.get_level_values('Date')` is needed here for TALIB inputs.

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
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9)
        
        # === Ichimoku Group ===
        df['Ichimoku_Conversion'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Ichimoku_Base'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        
        # === Volatility Group ===
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Normalized_ATR'] = df['ATR'] / (df['Close'] + 1e-9)
        df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
        
        # === Momentum Group ===
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
        
        # === Volume Indicator Group ===
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
        df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
        
        # === Price Pattern Group ===
        df['Parabolic_SAR'] = talib.SAR(df['High'], df['Low'])
        df['High_Low_Range'] = df['High'] - df['Low']
        df['HL_Range_Ratio'] = df['High_Low_Range'] / (df['Close'] + 1e-9)
        
        # === Synthetic Features Group ===
        df['Buying_Pressure'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)) * df['Volume']
        df['Selling_Pressure'] = ((df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-9)) * df['Volume']
        
        df['Support_Proximity'] = (df['Close'] - df['BB_Lower']) / (df['Close'] + 1e-9)
        df['Resistance_Proximity'] = (df['BB_Upper'] - df['Close']) / (df['Close'] + 1e-9)
        
        result_dfs.append(df)
    
    result = pd.concat(result_dfs, axis=0)
    result = result.sort_index() # Re-sort the MultiIndex after concatenation

    save_data_to_cache(result, feature_cache_name, data_type='features')
    print(f"Successfully calculated and cached features.")
    return result

def create_prediction_targets(data, forward_period=3):
    """
    Create prediction targets for a specified forward period.
    Assumes data has a MultiIndex (Date, Ticker).
    """
    # Calculate future returns by grouping on the Ticker level of the MultiIndex
    # and shifting within each group.
    # We add 1e-9 to Close to prevent division by zero for Future_Return calculation
    data['Future_Close'] = data.groupby(level='Ticker')['Close'].shift(-forward_period)
    data['Future_Return'] = (data['Future_Close'] / (data['Close'] + 1e-9)) - 1
    
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    data['Return_Magnitude'] = data['Future_Return'].abs()
    
    return data

def select_features_for_model(data):
    """
    Select and engineer additional features for prediction.
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
        'EMA_26', 'BB_Lower' # Need BB_Lower for BB_Position
    ]
    
    # Calculate price-relative versions of some indicators to reduce collinearity
    data['SMA20_Ratio'] = data['Close'] / (data['SMA_20'] + 1e-9)
    data['SMA50_Ratio'] = data['Close'] / (data['SMA_50'] + 1e-9)
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / ((data['BB_Upper'] - data['BB_Lower']) + 1e-9)
    data['EMA_Ratio'] = data['EMA_12'] / (data['EMA_26'] + 1e-9)

    selected_features.extend([
        'SMA20_Ratio', 'SMA50_Ratio', 'BB_Position', 'EMA_Ratio'
    ])
    
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
    data['Log_Returns'] = np.log(data['Close'] / (data.groupby(level='Ticker')['Close'].shift(1) + 1e-9))
    # Rolling on groupby objects: use .apply and then reindex
    temp_vol = data.groupby(level='Ticker')['Log_Returns'].apply(lambda x: x.rolling(window=10).std()) * np.sqrt(252)
    data['Volatility_10d'] = temp_vol.reindex(data.index) # Reindex to ensure proper alignment
    selected_features.append('Volatility_10d')

    # --- Momentum Features ---
    data['ROC_5d'] = (data['Close'] - data.groupby(level='Ticker')['Close'].shift(5)) / (data.groupby(level='Ticker')['Close'].shift(5) + 1e-9)
    data['ROC_10d'] = (data['Close'] - data.groupby(level='Ticker')['Close'].shift(10)) / (data.groupby(level='Ticker')['Close'].shift(10) + 1e-9)
    selected_features.extend(['ROC_5d', 'ROC_10d'])

    # --- Time-Based Features (Extract from Date level of MultiIndex) ---
    date_index = data.index.get_level_values('Date')
    data['DayOfWeek'] = date_index.dayofweek
    data['DayOfMonth'] = date_index.day
    data['Month'] = date_index.month
    selected_features.extend(['DayOfWeek', 'DayOfMonth', 'Month'])

    # Filter selected_features to only include those that actually exist in the DataFrame
    # This prevents errors if some indicators couldn't be calculated due to insufficient data
    final_features_list = [f for f in selected_features if f in data.columns]
    
    return data, final_features_list


def train_prediction_model(data, features, test_size=0.2):
    """
    Train an XGBoost model to predict price direction with confidence scores.
    Handles NaN values and standardizes features.
    """
    # Remove any NaN values in features, target, or future_return
    model_data = data.dropna(subset=features + ['Target', 'Future_Return'])
    
    if model_data.empty:
        print("Warning: No sufficient data after dropping NaNs for model training. Returning None.")
        return None, None, None

    # Feature matrix and target
    X = model_data[features]
    y = model_data['Target']
    
    # Handle remaining NaN values in features with mean imputation
    X = X.fillna(X.mean()) # Impute NaNs *after* dropping rows with NaNs in critical columns

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate scale_pos_weight for class imbalance
    num_neg = np.sum(y == 0)
    num_pos = np.sum(y == 1)
    scale_pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.4f}")
    
    # Fixed XGBoost parameters from previous optimization
    best_params = {
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 7,
        'n_estimators': 500,
        'reg_alpha': 0,
        'scale_pos_weight': scale_pos_weight_value, # Use calculated value here
        'subsample': 0.7,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42
    }
    
    # Time Series Cross-Validation for evaluation
    tscv_eval = TimeSeriesSplit(n_splits=5)
    all_predictions = []
    all_actual = []
    
    print("\nStarting TimeSeries Cross-Validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(tscv_eval.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
            early_stopping_rounds=50 # Add early stopping
        )
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_predictions.extend(y_pred_proba)
        all_actual.extend(y_test)
        print(f"  Fold {fold_idx + 1} complete.")

    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_scaled, y, early_stopping_rounds=50, eval_set=[(X_scaled, y)], verbose=False) # Fit on all data for deployment
    print("Final model trained.")
    
    # Evaluate overall performance
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
    The primary goal is to ensure 'risk_pct' accurately reflects the percentage
    of the current price risked if the stop loss is hit.
    """
    # Ensure data has enough history for rolling calculations.
    # Using tail(N) to get relevant recent data for indicators.
    # Using 200 as a safe minimum for many common indicators.
    # For a MultiIndex, 'data' here is a slice for a single ticker (from enhanced_signal_generation)
    recent_data = data.tail(max(200, len(data))).copy()

    if recent_data.empty or len(recent_data) < 20: # Added len(recent_data) < 20 for basic indicator needs
        # Fallback for insufficient historical data for robust calculation
        print("Warning: Insufficient historical data for robust risk calculation. Using default.")
        adjusted_risk_dollar = price * 0.02 # 2% of price as dollar risk
        stop_loss = price - adjusted_risk_dollar
        take_profit = price + (adjusted_risk_dollar * min_risk_reward_ratio)
        if stop_loss <= 0: stop_loss = price * 0.001 # Ensure stop_loss is always positive
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': adjusted_risk_dollar,
            'reward_amount': take_profit - price,
            'risk_reward_ratio': min_risk_reward_ratio,
            'risk_pct': (adjusted_risk_dollar / (price + 1e-9)) * 100,
            'reward_pct': ((take_profit - price) / (price + 1e-9)) * 100
        }

    # --- Step 1: Calculate various risk measures (dollar amounts) ---

    # Method 1: ATR-based calculation
    # Ensure ATR is calculated over a reasonable period and is a positive value
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
            bb_range_risk = min(bb_range * 0.3, price * 0.03) # Cap at 3% of price (tighter)

    # Method 3: Recent volatility based calculation
    log_returns = np.log(recent_data['Close'] / (recent_data['Close'].shift(1) + 1e-9)).dropna()
    recent_volatility_dollar = 0
    if len(log_returns) >= 20:
        volatility_daily_std = log_returns.tail(20).std()
        recent_volatility_dollar = volatility_daily_std * price
        recent_volatility_dollar = min(recent_volatility_dollar * np.sqrt(3), price * 0.04) # Scale for 3-day period, cap at 4%

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
    risk_measures_dollars.append(price * 0.005) # Minimum 0.5% price movement risk

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
    adjusted_risk_dollar = min(adjusted_risk_dollar, price * 0.08) # MAX 8% of price as dollar risk

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
    
    take_profit = min(take_profit, price * 1.25) # Max 25% gain cap

    # --- Step 5: Final Adjustments and Return Values ---
    actual_risk_amount = adjusted_risk_dollar 
    actual_reward_amount = take_profit - price
    actual_risk_reward_ratio = actual_reward_amount / (actual_risk_amount + 1e-9) if actual_risk_amount > 0 else min_risk_reward_ratio

    if actual_reward_amount > 0 and actual_risk_reward_ratio < min_risk_reward_ratio:
        required_reward_for_ratio = actual_risk_amount * min_risk_reward_ratio
        take_profit = price + required_reward_for_ratio
        actual_reward_amount = required_reward_for_ratio
        actual_risk_reward_ratio = min_risk_reward_ratio
    elif actual_reward_amount <= 0: # If take profit is at or below entry, then R:R is irrelevant or bad
        actual_risk_reward_ratio = 0

    risk_pct_of_price = (actual_risk_amount / (price + 1e-9)) * 100 if price > 0 else 0
    reward_pct_of_price = (actual_reward_amount / (price + 1e-9)) * 100 if price > 0 else 0

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

    raw_position_size_pct = (max_portfolio_risk / (risk_pct + 1e-9)) * 100 if risk_pct > 0 else 0 # Add epsilon
    aggression_multiplier = 1.5
    confidence_adjusted_size = raw_position_size_pct * confidence_factor * aggression_multiplier
    final_position_size = min(confidence_adjusted_size, max_position_pct)

    if final_position_size < 0.5:
        return 0.0

    return final_position_size


def enhanced_signal_generation(model, scaler, data, features, min_confidence=0.55):
    """
    Generate trading signals based on model predictions.
    This version processes data by ticker to ensure correct historical slicing for each.
    """
    signals = data.copy()

    # Initialize new columns
    signals['Confidence'] = np.nan
    signals['Signal'] = 0
    signals['Stop_Loss'] = np.nan
    signals['Take_Profit'] = np.nan
    signals['Risk_Reward_Ratio'] = np.nan
    signals['Position_Size_Pct'] = np.nan
    signals['Risk_Pct'] = np.nan
    signals['Reward_Pct'] = np.nan

    # Prepare features and get model predictions for the entire dataset
    X_full = signals[features].fillna(signals[features].mean())
    X_scaled_full = scaler.transform(X_full)

    signals['Confidence'] = model.predict_proba(X_scaled_full)[:, 1]
    signals['Signal'] = (signals['Confidence'] > min_confidence).astype(int)

    # Iterate by ticker to ensure correct historical slicing
    grouped_by_ticker = signals.groupby(level='Ticker')

    for ticker, group_df_for_ticker in grouped_by_ticker:
        # Ensure the group is sorted by date index
        group_df_sorted = group_df_for_ticker.sort_index(level='Date')

        for i, row_tuple in enumerate(group_df_sorted.itertuples()):
            current_date_idx = row_tuple.Index # This is the MultiIndex tuple (Date, Ticker)
            current_price = row_tuple.Close
            confidence = row_tuple.Confidence
            signal_status = row_tuple.Signal

            if signal_status == 1:
                # Get current and historical data slice for this specific ticker UP TO this point.
                current_data_slice = group_df_sorted.iloc[:i+1].copy()
                min_history_required = 200 # Based on SMA_200, adjust if needed

                if len(current_data_slice) < min_history_required:
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
    actual_signals = signals[signals['Signal'] == 1].copy()
    
    if len(actual_signals) == 0:
        print("No trading signals generated for analysis.")
        return {}
    
    # Calculate simulated returns using the corrected simulation function
    actual_signals['Simulated_Return'] = actual_signals.apply(
        lambda row: simulate_trade_outcome(row, forward_period, signals), # Pass the full signals DF for future data
        axis=1
    ).dropna() # Drop trades that couldn't be resolved (e.g., no future data)
    
    total_trades = len(actual_signals)
    if total_trades == 0:
        print("No valid trades to analyze after simulation (all dropped due to NaN returns).")
        return {}

    winning_trades = (actual_signals['Simulated_Return'] > 0).sum()
    losing_trades = (actual_signals['Simulated_Return'] < 0).sum()
    
    win_rate = winning_trades / total_trades
    
    avg_win = actual_signals[actual_signals['Simulated_Return'] > 0]['Simulated_Return'].mean()
    avg_loss = actual_signals[actual_signals['Simulated_Return'] < 0]['Simulated_Return'].mean()
    
    total_wins = actual_signals[actual_signals['Simulated_Return'] > 0]['Simulated_Return'].sum()
    total_losses = abs(actual_signals[actual_signals['Simulated_Return'] < 0]['Simulated_Return'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
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
    # Use pd.IndexSlice for robust multi-index slicing
    idx = pd.IndexSlice
    
    # Define the end date for the future slice (up to forward_period, plus some buffer)
    # This buffer is important because 'interval' might not be daily, e.g., '3d'.
    # So we need enough "interval" points to cover 'forward_period' days.
    # A generous buffer: forward_period * 2, or min_data_points for interval * interval_length_days
    # For interval='3d', a 3-day forward period needs at least 1 interval.
    # To be safe, look for a period covering up to 5 intervals after current date.
    slice_start_date = entry_date + timedelta(days=1)
    slice_end_date = entry_date + timedelta(days=forward_period * 5) # Increased buffer for intervals

    try:
        # Select rows where ticker matches AND date is within the future window
        # The .loc accessor on a MultiIndex DataFrame works with a tuple (level0_value, level1_slice)
        future_data_slice = full_signals_df.loc[
            (ticker, idx[slice_start_date:slice_end_date]),
            ['High', 'Low', 'Close'] # Only need these columns for simulation
        ]
        # It's already sorted by Date due to the MultiIndex structure
    except KeyError:
        # This can happen if a ticker just doesn't exist in the slice or date range
        return np.nan # Cannot simulate if future data cannot be found


    # If no future data available (e.g., trade very close to end of `full_signals_df`)
    if future_data_slice.empty:
        # Fallback to Future_Return if no future OHLC available for simulation
        if not pd.isna(trade_row['Future_Close']):
            return (trade_row['Future_Close'] / (current_price + 1e-9)) - 1
        else:
            return np.nan # Cannot resolve trade outcome
        
    # Get the actual future close after `forward_period` days, from the trade_row itself.
    # This is the 'ideal' target if no SL/TP is hit.
    future_final_close = trade_row['Future_Close']

    # Iterate through each day/interval in the future slice to check for SL/TP hits
    for _, future_day_row in future_data_slice.iterrows():
        future_high = future_day_row['High']
        future_low = future_day_row['Low']

        # Ensure High/Low are not NaN
        if pd.isna(future_high) or pd.isna(future_low):
            continue

        # Check for SL hit first (assumed precedence for simplicity)
        if future_low <= stop_loss:
            return (stop_loss / (current_price + 1e-9)) - 1

        # Check for TP hit
        if future_high >= take_profit:
            return (take_profit / (current_price + 1e-9)) - 1
        
    # If neither SL nor TP was hit within the observed future_data_slice,
    # the trade's outcome is determined by its `Future_Close` at the end of the `forward_period`.
    # Ensure future_final_close is not NaN before calculating return
    if not pd.isna(future_final_close):
        return (future_final_close / (current_price + 1e-9)) - 1
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
    
    # Filter out NaNs for plotting, specifically those from unresolved trades
    plot_signals = actual_signals.dropna(subset=['Risk_Pct', 'Reward_Pct', 'Confidence', 'Position_Size_Pct', 'Simulated_Return'])
    # Ensure Position_Size_Pct is numeric for size parameter
    plot_signals['Position_Size_Pct_Scaled'] = plot_signals['Position_Size_Pct'] * 2 # Scale for visibility

    if not plot_signals.empty:
        # Scatter plot for Risk vs Reward
        scatter = ax1.scatter(plot_signals['Risk_Pct'], plot_signals['Reward_Pct'], 
                            c=plot_signals['Confidence'], cmap='viridis', alpha=0.7, s=plot_signals['Position_Size_Pct_Scaled'])
        
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
        # Scatter plot for Confidence vs Position Size
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
        # Histogram for Return Distribution
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
        # Time Series of Signals
        unique_tickers = all_signals.index.get_level_values('Ticker').unique()
        selected_tickers_for_plot = unique_tickers[:2] # Plot up to 2 tickers for clarity
        ax4.clear() # Clear existing content if any

        for ticker_to_plot in selected_tickers_for_plot:
            # Select all data for this specific ticker and get its Date level index
            ticker_data = all_signals.loc[(ticker_to_plot, slice(None)), :]
            # Sort by Date (already sorted, but good to be explicit for consistency)
            ticker_data_dates = ticker_data.index.get_level_values('Date').to_numpy() # Get dates as numpy array
            
            # Select signals for this specific ticker from plot_signals
            ticker_signals = plot_signals.loc[(ticker_to_plot, slice(None)), :]
            ticker_signals_dates = ticker_signals.index.get_level_values('Date').to_numpy() # Get dates as numpy array

            ax4.plot(ticker_data_dates, ticker_data['Close'], label=f'{ticker_to_plot} Close Price', alpha=0.8, linewidth=1.5)
            
            if not ticker_signals.empty:
                ax4.scatter(ticker_signals_dates, ticker_signals['Close'], 
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
        # When plotting, use get_level_values('Date') to get a simple DatetimeIndex for plotting.
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


def main(tickers, train_start_date='2022-01-01', train_end_date='2023-12-31', 
         test_start_date='2024-01-01', test_end_date=None, forward_period=3, min_confidence=0.55,
         force_download=False, force_recalculate_features=False): # Added force flags
    """
    Main function to run the entire prediction and trading signal generation
    
    Args:
        tickers (list): List of ticker symbols
        train_start_date (str): Start date for training data (YYYY-MM-DD)
        train_end_date (str): End date for training data (YYYY-MM-DD)
        test_start_date (str): Start date for testing/backtesting data (YYYY-MM-DD)
        test_end_date (str): End date for testing/backtesting data (YYYY-MM-DD), defaults to current date
        forward_period (int): Number of days to look forward for predictions
        min_confidence (float): Minimum confidence threshold for signals
        force_download (bool): If True, forces re-download of raw data bypassing cache.
        force_recalculate_features (bool): If True, forces recalculation of features bypassing cache.
    """
    # Set default test_end_date to current date if not provided
    if test_end_date is None:
        test_end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=== CRYPTOCURRENCY TRADING SIGNAL SYSTEM ===")
    print(f"Training Period: {train_start_date} to {train_end_date}")
    print(f"Testing Period: {test_start_date} to {test_end_date}")
    print(f"Forward Prediction: {forward_period} days")
    print(f"Minimum Confidence for Signal: {min_confidence}")
    
    # --- Data Fetching ---
    print("\n1. Fetching training data...")
    train_data = fetch_and_prepare_data(
        tickers, 
        start_date=train_start_date,
        end_date=train_end_date,
        interval='3d', # Keep this interval for consistency
        force_download=force_download
    )
    
    print("\n2. Fetching testing data...")
    # For testing, fetch data that extends *past* the actual test_end_date
    # so that 'Future_Return' and 'Future_Close' can be calculated correctly
    # for signals generated up to test_end_date.
    # We fetch enough data to cover the forward period for simulation,
    # plus some buffer for rolling windows/indicators at the very end.
    # Yahoo Finance 'period' parameter is relative, so we use it with 'interval'
    # and then filter by 'start_date'/'end_date' if specified.
    # To ensure future data is available for `simulate_trade_outcome`, we fetch data
    # up to the current date if test_end_date is None, or a bit beyond provided test_end_date.
    
    # Determine period for yfinance to capture enough data
    # Calculate days between test_start_date and now/test_end_date + buffer
    start_dt = datetime.strptime(test_start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(test_end_date, '%Y-%m-%d')
    days_in_test_period = (end_dt - start_dt).days
    # Add buffer for forward_period and indicator calculations
    total_days_needed = days_in_test_period + forward_period * 5 # Generous buffer

    # YF period accepts '1y', '5y', 'max', etc. or timedelta strings like '7d'
    # It's usually easier to specify a long period and then filter by start/end_date
    # If test_end_date is current, period='max' or a very long period is safest
    yf_period = 'max' # Fetch max historical data and then filter
    
    test_data_raw = fetch_and_prepare_data(
        tickers,
        period=yf_period, # Fetch max data
        interval='3d',
        start_date=test_start_date, # Filter after fetching
        end_date=None, # Let YF fetch up to current, then filter later
        force_download=force_download
    )
    
    # --- Feature Calculation ---
    print("\n3. Calculating features for training data...")
    train_data = calculate_selected_features(train_data, cache_key_suffix="train", force_recalculate=force_recalculate_features)
    
    print("\n4. Calculating features for testing data...")
    # This calculation should also be on the raw, extended test data for accurate future returns
    test_data_raw_features = calculate_selected_features(test_data_raw, cache_key_suffix="test_extended", force_recalculate=force_recalculate_features)
    
    # --- Create Prediction Targets ---
    print("\n5. Creating prediction targets for training data...")
    train_data = create_prediction_targets(train_data, forward_period)
    
    print("\n5. Creating prediction targets for testing data (on extended data for future targets)...")
    test_data_with_targets = create_prediction_targets(test_data_raw_features, forward_period)
    
    # --- Select and Engineer Final Features ---
    print("\n6. Selecting and engineering final features for training data...")
    train_data, selected_features = select_features_for_model(train_data)
    
    print("\n6. Selecting and engineering final features for testing data...")
    test_data_with_targets, _ = select_features_for_model(test_data_with_targets)
    
    # --- Filter test_data to the actual test_end_date *after* all features and targets are calculated ---
    # This ensures that 'signals' DataFrame used for analysis only contains signals
    # within the specified test_start_date to test_end_date.
    # The 'test_data_raw_features' (passed to simulate_trade_outcome via 'signals' later)
    # still retains the future data needed for simulation.
    test_data_final = test_data_with_targets[test_data_with_targets.index.get_level_values('Date') <= test_end_date].copy()

    # --- Train Model ---
    print("\n7. Training prediction model...")
    model, scaler, features = train_prediction_model(train_data, selected_features)
    
    # Handle case where training fails (e.g., no data)
    if model is None:
        print("Model training failed. Cannot proceed with signal generation or analysis.")
        return None, None, None, None

    # --- Generate Trading Signals ---
    print("\n8. Generating enhanced trading signals...")
    # Pass 'test_data_with_targets' (the extended one) to `enhanced_signal_generation`
    # because `simulate_trade_outcome` needs future OHLC for its look-ahead logic.
    signals = enhanced_signal_generation(model, scaler, test_data_final, features, min_confidence)
    
    # --- Analyze Performance ---
    print("\n9. Analyzing performance...")
    # `advanced_performance_analysis` uses `signals` which has `Simulated_Return` calculated
    performance = advanced_performance_analysis(signals, forward_period)
    
    # --- Display Latest Signals ---
    print("\n" + "="*60)
    print("LATEST TRADING OPPORTUNITIES")
    print("="*60)
    
    # Filter for viable signals that were actually generated
    viable_signals = signals[
        (signals['Signal'] == 1) & 
        (signals['Position_Size_Pct'].notna()) &
        (signals['Stop_Loss'].notna())
    ].tail(5)

    if len(viable_signals) > 0:
        for (date_idx, ticker_idx), row in viable_signals.iterrows(): # Iterate over MultiIndex
            print(f"\n🎯 {ticker_idx} - {date_idx.strftime('%Y-%m-%d')}")
            print(f"   💰 Entry: ${row['Close']:.2f}")
            print(f"   🛑 Stop Loss: ${row['Stop_Loss']:.2f} ({((row['Stop_Loss'] / row['Close']) - 1) * 100:.2f}% risk)")
            print(f"   🎯 Take Profit: ${row['Take_Profit']:.2f} ({((row['Take_Profit'] / row['Close']) - 1) * 100:.2f}% reward)")
            print(f"   ⚖️  Risk-Reward: {row['Risk_Reward_Ratio']:.2f}:1")
            print(f"   📊 Confidence: {row['Confidence']:.1%}")
            print(f"   💼 Position Size: {row['Position_Size_Pct']:.2f}% of portfolio")
    else:
        print("No current trading signals meet the enhanced criteria.")
    
    print(f"\n{'='*60}")
    print("SYSTEM SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Multi-Index DataFrame structure implemented for robustness.")
    print(f"✅ Comprehensive caching for raw and feature-engineered data.")
    print(f"✅ Enhanced risk management active (realistic bounds).")
    print(f"✅ Minimum 2:1 risk-reward ratio enforced (at calculation, then filtered for >=1.5).")
    print(f"✅ Dynamic position sizing implemented (More Aggressive).")
    print(f"✅ Multi-factor risk calculation active.")
    print(f"✅ Confidence-based filtering at {min_confidence:.1%}.")
    
    return model, scaler, signals, performance


# Usage example and execution
if __name__ == "__main__":
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 
                      'AVAX-USD', 'LINK-USD', 'DOGE-USD']
    
    # Ensure cache directories exist
    ensure_directories()

    model, scaler, signals, performance = main(
        crypto_tickers, 
        train_start_date='2023-01-01',  # Training data from 2023
        train_end_date='2024-03-31',    # Training data until end of March 2024
        test_start_date='2024-04-01',   # Backtesting from start of April 2024
        test_end_date=None,             # Backtesting until current date (today)
        forward_period=3,               # 3-day prediction window
        min_confidence=0.55,            # Minimum confidence threshold for signals
        force_download=False,           # Set to True to re-download all raw data
        force_recalculate_features=False # Set to True to re-calculate all features
    )
    
    print("\n🚀 Cryptocurrency Trading System Ready!")
    print("📈 All signals now feature realistic risk-reward ratios and position sizes.")
    print("💡 Position sizes dynamically adjusted based on confidence and risk.")
    print("🛡️  Multi-layer risk management active.")