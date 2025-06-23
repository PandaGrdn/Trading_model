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


# Function to fetch and prepare data with caching
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

    # --- Inter-asset features (calculated BEFORE grouping, then merged) ---
    # Calculate average market return (simple average of all daily returns)
    # This needs to be done on the full dataset before grouping by ticker
    # Adjusting for interval data: Calculate mean daily return
    # First, ensure index is datetime
    data.index = pd.to_datetime(data.index)

    # Calculate daily returns for each ticker and then average across tickers for each timestamp
    # Use 'Daily_Return' (which is actually 'Interval_Return' in this context)
    market_avg_return = data.groupby(data.index)['Daily_Return'].mean().rename('Market_Avg_Return')

    # Merge market average return back into the original data, ensuring alignment by index
    data = data.merge(market_avg_return, left_index=True, right_index=True, how='left')

    # --- Time-based features (calculated BEFORE grouping) ---
    data['Hour_Of_Day'] = data.index.hour

    # Now, proceed with grouped calculations for ticker-specific features
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
        df['VWAP'] = ((df['Close'] * df['Volume']).rolling(window=20).sum()) / (df['Volume'].rolling(window=20).sum() + 1e-9)
        df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10) # Chaikin Money Flow (ADOSC is Accumulation/Distribution Oscillator)

        # === Price Pattern Group ===
        df['Parabolic_SAR'] = talib.SAR(df['High'], df['Low'])
        df['High_Low_Range'] = df['High'] - df['Low']
        df['HL_Range_Ratio'] = df['High_Low_Range'] / (df['Close'] + 1e-9) # Add epsilon

        # === Synthetic Features Group ===
        df['Buying_Pressure'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)).fillna(0) * df['Volume']
        df['Selling_Pressure'] = ((df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-9)).fillna(0) * df['Volume']
        df['Support_Proximity'] = (df['Close'] - df['BB_Lower']) / (df['Close'] + 1e-9)
        df['Resistance_Proximity'] = (df['BB_Upper'] - df['Close']) / (df['Close'] + 1e-9)

        # --- NEW: Lagged Features (Adjust periods as needed for your interval) ---
        # Lagged returns
        df['Log_Return_Lag1'] = df['Log_Return'].shift(1)
        df['Daily_Return_Lag1'] = df['Daily_Return'].shift(1)
        df['Daily_Return_Lag2'] = df['Daily_Return'].shift(2)

        # Lagged RSI
        df['RSI_Lag1'] = df['RSI'].shift(1)
        df['RSI_Lag2'] = df['RSI'].shift(2)

        # Lagged MACD
        df['MACD_Lag1'] = df['MACD'].shift(1)
        df['MACD_Signal_Lag1'] = df['MACD_Signal'].shift(1)

        # Lagged Volume Change
        df['Volume_Change_Lag1'] = df['Volume_Change'].shift(1)

        # --- NEW: Replace infinities with NaN right after calculation for each group ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        result = pd.concat([result, df], axis=0)

    # Re-merge the market average return and time features into the final result dataframe
    # This ensures they are present for all rows after the grouped operations
    result = result.merge(market_avg_return, left_index=True, right_index=True, how='left')
    result['Hour_Of_Day'] = result.index.hour

    # Fill any remaining NaNs created by TA-Lib (e.g., initial periods of MAs)
    result = result.fillna(method='ffill').fillna(method='bfill') # Forward fill, then backward fill for leading NaNs
    result = result.fillna(0) # Catch any remaining NaNs (e.g., if series was all NaNs)

    return result

def create_prediction_targets(data, forward_period, profit_threshold=0.01, stop_loss_threshold=0.02):
    """
    Create prediction targets based on the 'first touch' of a profit or stop-loss threshold
    within a specified forward period.

    Args:
        data (pd.DataFrame): DataFrame with 'Close' and 'Ticker' columns.
        forward_period (int): Number of intervals into the future to look for a touch.
        profit_threshold (float): The percentage gain at which to consider a 'profit touch'.
                                  E.g., 0.01 for 1% gain.
        stop_loss_threshold (float): The percentage loss at which to consider a 'stop loss touch'.
                                     E.g., 0.005 for 0.5% loss.

    Returns:
        pd.DataFrame: Original DataFrame with 'Target' (1 for profit, 0 for stop loss),
                      'Future_Return' (the actual return at the time of touch or end of period),
                      and 'Return_Magnitude' columns.
    """
    print(f"Creating prediction targets based on first touch within {forward_period} intervals...")
    data['Target'] = np.nan
    data['Future_Return'] = np.nan
    data['Return_Magnitude'] = np.nan

    grouped = data.groupby('Ticker')

    for ticker, group in grouped:
        for i in range(len(group) - forward_period):
            current_close = group['Close'].iloc[i]
            
            # Slice the future data for the current row's forward period
            future_prices = group['Close'].iloc[i+1 : i+1+forward_period]
            
            if future_prices.empty:
                continue

            # Calculate the upper and lower bounds for profit/stop loss
            profit_price = current_close * (1 + profit_threshold)
            stop_loss_price = current_close * (1 - stop_loss_threshold)

            target_set = False
            for j, future_price in enumerate(future_prices):
                # Check for profit touch
                if future_price >= profit_price:
                    data.loc[group.index[i], 'Target'] = 1
                    data.loc[group.index[i], 'Future_Return'] = (future_price - current_close) / current_close
                    target_set = True
                    break # Target hit, move to next initial row

                # Check for stop loss touch
                if future_price <= stop_loss_price:
                    data.loc[group.index[i], 'Target'] = 0
                    data.loc[group.index[i], 'Future_Return'] = (future_price - current_close) / current_close
                    target_set = True
                    break # Target hit, move to next initial row
            
            
            # If neither threshold is hit within the forward period, target remains NaN.
            # This implicitly removes these instances from training when NaNs are dropped.
            
    # Calculate the absolute magnitude of the future return for analysis
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
        'Buying_Pressure', 'Selling_Pressure', 'Support_Proximity', 'Resistance_Proximity',

        # --- NEW: Added Features ---
        'Market_Avg_Return', # Inter-asset feature
        'Hour_Of_Day',       # Time-based
        'Log_Return_Lag1',   # Lagged feature
        'Daily_Return_Lag1', # Lagged feature
        'Daily_Return_Lag2', # Lagged feature
        'RSI_Lag1',          # Lagged feature
        'RSI_Lag2',          # Lagged feature
        'MACD_Lag1',         # Lagged feature
        'MACD_Signal_Lag1',  # Lagged feature
        'Volume_Change_Lag1' # Lagged feature
    ]

    # Filter out features that are unlikely to have enough data for short intraday periods
    filtered_base_features = [f for f in base_selected_features if f in data.columns and f not in ['SMA_200']]

    # Ensure all base features exist in the data before trying to create ratios
    available_features = [f for f in filtered_base_features if f in data.columns]

    # Create additional ratio-based features to capture relationships
    data['SMA5_Ratio'] = data['Close'] / (data['SMA_5'] + 1e-9)
    data['SMA20_Ratio'] = data['Close'] / (data['SMA_20'] + 1e-9)
    data['EMA12_Ratio'] = data['Close'] / (data['EMA_12'] + 1e-9)
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-9)
    data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']
    data['EMA_Cross_Signal'] = (data['EMA_12'] > data['EMA_26']).astype(int)
    data['RSI_Overbought_Sell'] = (data['RSI'] > 70).astype(int)
    data['RSI_Oversold_Buy'] = (data['RSI'] < 30).astype(int)


    # Combine all selected features
    all_selected_features = available_features + [
        'SMA5_Ratio', 'SMA20_Ratio',
        'EMA12_Ratio', 'BB_Position', 'MACD_Diff',
        'EMA_Cross_Signal', 'RSI_Overbought_Sell', 'RSI_Oversold_Buy'
    ]

    # Filter out any features that might not have been created due to missing data or calculation issues
    final_features = [f for f in all_selected_features if f in data.columns]

    print(f"Selected {len(final_features)} features for the model.")
    return data, final_features

def train_prediction_model(data, features):
    """
    Train an XGBoost classifier using TimeSeriesSplit for robust validation.
    The model predicts the 'Target' (future price direction) and outputs
    confidence scores.

    Args:
        data (pd.DataFrame): DataFrame containing features and 'Target' column.
        features (list): List of feature column names to use for training.

    Returns:
        tuple:
            - xgb.XGBClassifier: The trained XGBoost model.
            - StandardScaler: The fitted scaler used for feature normalization.
            - list: The list of features used for training.
    """
    # Convert any infinity or very large values to NaN before dropping
    # This specifically addresses the ValueError: Input X contains infinity
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows where 'Target', 'Future_Return' or any of the selected features are NaN
    # This is crucial as time series features like MAs have leading NaNs
    model_data = data.dropna(subset=features + ['Target', 'Future_Return'])
    
    # Ensure there's enough data after dropping NaNs
    if model_data.empty:
        raise ValueError("No valid data remaining after dropping NaNs for model training. Check data range and feature calculation.")

    # Separate features (X) and target (y)
    X = model_data[features]
    y = model_data['Target']

    # Handle any remaining NaNs in features by filling with the mean
    # (Though dropna should have removed most, some edge cases might remain)
    X = X.fillna(X.mean())
    
    # Capture the exact feature names that go into the scaler and then the model
    features_for_model_training = X.columns.tolist()

    # Initialize StandardScaler for feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Fit and transform on the entire training set

    # TimeSeriesSplit for robust validation
    # Splits the data into 5 folds, ensuring no data leakage from future to past
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_predictions = [] # Store probabilities for AUC calculation
    all_actual = []      # Store actual targets

    print("Training model with TimeSeriesSplit (5 folds)...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        print(f"  Fold {fold + 1}/{tscv.n_splits}")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Ensure y_train has both classes for scale_pos_weight
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print(f"    Warning: Skipping fold {fold+1} due to single class in y_train.")
            continue

        # Initialize and train XGBoost Classifier for each fold
        # Parameters are tuned for classification task
        model = xgb.XGBClassifier(
            n_estimators=100,             # Number of boosting rounds
            learning_rate=0.05,           # Step size shrinkage to prevent overfitting
            max_depth=4,                  # Maximum depth of a tree
            subsample=0.8,                # Subsample ratio of the training instance
            colsample_bytree=0.8,         # Subsample ratio of columns when constructing each tree
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(), # Handles class imbalance
            objective='binary:logistic',  # Binary classification
            eval_metric='auc',            # Evaluation metric during training (Area Under Curve)
            use_label_encoder=False,      # Suppress warning for older sklearn versions
            random_state=42               # For reproducibility
        )
        # Removed 'feature_names' argument from fit call as it caused TypeError in some XGBoost versions
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False) 
        
        # Collect predictions and actual values for overall metrics
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (Target=1)
        all_predictions.extend(y_pred_proba)
        all_actual.extend(y_test)
    
    # Train a final model on the entire dataset for deployment
    print("Training final model on full dataset...")
    # Ensure y has both classes for scale_pos_pos_weight for final model
    if y.sum() == 0 or y.sum() == len(y):
        print("Warning: Final model training skipped due to single class in full dataset. Returning last trained fold model.")
        if 'model' in locals(): # Check if a model was trained in any fold
            return model, scaler, features_for_model_training # Return the last successful fold model with correct features
        else:
            raise ValueError("No valid training folds completed to train a final model.")


    final_model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=(len(y) - y.sum()) / y.sum(),
        objective='binary:logistic', eval_metric='auc', use_label_encoder=False,
        random_state=42
    )
    # Removed 'feature_names' argument from fit call as it caused TypeError in some XGBoost versions
    final_model.fit(X_scaled, y)
    
    # --- Performance Metrics on Validation Folds ---
    if all_actual and all_predictions: # Ensure lists are not empty
        auc_score = roc_auc_score(all_actual, all_predictions)
        print(f"\nOverall ROC AUC (from cross-validation): {auc_score:.4f}")
        
        # Convert probabilities to binary predictions for classification report
        binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
        print("\nClassification Report (from cross-validation):")
        print(classification_report(all_actual, binary_predictions, zero_division=0)) # zero_division=0 to handle cases with no true/predicted 0s/1s
        
        # --- Visualization: Confusion Matrix ---
        cm = confusion_matrix(all_actual, binary_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Aggregated from CV Folds)')
        plt.show()
    else:
        print("\nNo predictions or actuals collected from cross-validation to report metrics.")
    
    # --- Visualization: Feature Importance ---
    plt.figure(figsize=(12, 8))
    # Use the precisely captured feature names from the X DataFrame before scaling
    feature_importances = pd.Series(final_model.feature_importances_, index=features_for_model_training)
    feature_importances.nlargest(20).plot(kind='barh') # Display top 20
    plt.title('Feature Importance (from Final Model)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
    
    return final_model, scaler, features_for_model_training # Return the correct feature names

def calculate_position_size(confidence, max_position_pct=5.0, min_confidence=0.55):
    """
    Calculates the percentage of portfolio to allocate based on model confidence.
    A higher confidence leads to a larger position size, up to max_position_pct.

    Args:
        confidence (float): The model's prediction probability for the positive class (0.0 to 1.0).
        max_position_pct (float): Maximum allowed position size as a percentage (e.g., 5.0 for 5%).
        min_confidence (float): Minimum confidence required to initiate a trade.

    Returns:
        float: The calculated position size percentage.
    """
    if confidence < min_confidence:
        return 0.0 # No position if confidence is below threshold
    
    # Scale confidence linearly from min_confidence to 1.0, mapping to 0.0 to max_position_pct
    scaled_confidence = (confidence - min_confidence) / (1.0 - min_confidence + 1e-9) # Add epsilon
    position_size = scaled_confidence * max_position_pct
    
    # Ensure position size does not exceed the maximum allowed
    return min(position_size, max_position_pct)

def calculate_risk_metrics(data_slice, price, confidence, volatility_metric='Normalized_ATR'):
    """
    Calculates stop-loss and take-profit levels based on current price,
    model confidence, and a volatility metric (e.g., ATR).

    Args:
        data_slice (pd.DataFrame): A slice of the historical data up to the current point,
                                   needed to get the latest volatility.
        price (float): The current entry price of the asset.
        confidence (float): The model's confidence score for the trade.
        volatility_metric (str): The name of the volatility column to use.

    Returns:
        dict: A dictionary containing 'stop_loss', 'take_profit', and 'risk_reward_ratio'.
              Returns NaN for values if volatility data is missing.
    """
    if data_slice.empty or volatility_metric not in data_slice.columns:
        # print(f"Warning: Missing data or volatility metric '{volatility_metric}' for risk metrics calculation.")
        return {'stop_loss': np.nan, 'take_profit': np.nan, 'risk_reward_ratio': np.nan}

    volatility = data_slice[volatility_metric].iloc[-1]
    
    if pd.isna(volatility) or volatility == 0:
        # print(f"Warning: Volatility for {data_slice['Ticker'].iloc[-1]} is NaN or zero. Cannot calculate risk metrics.")
        return {'stop_loss': np.nan, 'take_profit': np.nan, 'risk_reward_ratio': np.nan}

    # Adjust confidence factor: higher confidence reduces stop distance slightly, assuming better signal quality
    # Confidence > 0.5 makes confidence_factor less than 1, reducing stop_distance.
    # Confidence <= 0.5 makes confidence_factor 1.0.
    confidence_factor = 1.0 - ((confidence - 0.5) * 0.5) if confidence > 0.5 else 1.0
    
    # Stop distance is a multiple of volatility and price, adjusted by confidence
    # 2.5 is a common multiplier for ATR-based stops
    stop_distance = volatility * price * 2.5 * confidence_factor
    
    # Stop Loss for a BUY signal: below the entry price
    stop_loss = price - stop_distance
    
    # Risk/Reward Ratio: dynamically adjust based on confidence
    # Higher confidence can justify a slightly higher risk/reward target
    risk_reward_ratio = 1.5 + (confidence * 1.0) # Scale confidence (0-1) to add to 1.5-2.5
    
    # Take Profit for a BUY signal: above the entry price by risk_reward_ratio * stop_distance
    take_profit = price + (stop_distance * risk_reward_ratio)

    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward_ratio': risk_reward_ratio
    }


def generate_trading_signals(model, scaler, data, features, min_confidence=0.55):
    """
    Generates trading signals with correctly calculated, ticker-specific risk metrics.
    This version uses a precise assignment method to prevent write errors and
    correctly filters data to prevent data contamination.
    """
    signals = data.copy()
    
    # Ensure all required features are present
    X = signals.reindex(columns=features, fill_value=np.nan)

    # Robust NaN/inf handling
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(method='ffill').fillna(method='bfill')
    if X.isnull().values.any():
        X = X.fillna(0)

    X_scaled = scaler.transform(X)
    
    signals['Confidence'] = model.predict_proba(X_scaled)[:, 1]
    signals['Signal'] = (signals['Confidence'] > min_confidence).astype(int)
    signals['Position_Size_Pct'] = signals['Confidence'].apply(
        lambda x: calculate_position_size(x, min_confidence=min_confidence)
    )
    
    signals['Stop_Loss'] = np.nan
    signals['Take_Profit'] = np.nan
    signals['Risk_Reward'] = np.nan

    # --- DEFINITIVE FIX FOR CALCULATION AND ASSIGNMENT ---
    for i in range(len(signals)):
        if signals['Signal'].iloc[i] == 1:
            # 1. Get data for the current row
            current_ticker = signals['Ticker'].iloc[i]
            current_date = signals.index[i]
            
            # 2. **CRITICAL FIX**: Correctly filter for THIS ticker's history ONLY
            ticker_history = signals[(signals['Ticker'] == current_ticker) & (signals.index <= current_date)]
            
            if not ticker_history.empty:
                # 3. Calculate risk metrics using the clean history
                risk_metrics = calculate_risk_metrics(
                    ticker_history, 
                    signals['Close'].iloc[i], 
                    signals['Confidence'].iloc[i]
                )
                
                # 4. Use precise .iloc assignment to prevent write errors
                col_loc_sl = signals.columns.get_loc('Stop_Loss')
                col_loc_tp = signals.columns.get_loc('Take_Profit')
                col_loc_rr = signals.columns.get_loc('Risk_Reward')

                signals.iloc[i, col_loc_sl] = risk_metrics['stop_loss']
                signals.iloc[i, col_loc_tp] = risk_metrics['take_profit']
                signals.iloc[i, col_loc_rr] = risk_metrics['risk_reward_ratio']

    return signals

def visualize_signals(signals, ticker):
    ticker_signals = signals[signals['Ticker'] == ticker].copy()
    if ticker_signals.empty:
        return

    # Ensure the index is sorted and unique
    ticker_signals = ticker_signals[~ticker_signals.index.duplicated(keep='first')]
    ticker_signals = ticker_signals.sort_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(ticker_signals.index, ticker_signals['Close'], label='Close Price', color='blue')
    buy_signals = ticker_signals[ticker_signals['Signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
    
    for i, row in buy_signals.iterrows():
        ax1.plot([i, i], [row['Close'], row['Stop_Loss']], 'r--', alpha=0.5)
        ax1.scatter(i, row['Stop_Loss'], color='red', marker='_', s=100)
        ax1.plot([i, i], [row['Close'], row['Take_Profit']], 'g--', alpha=0.5)
        ax1.scatter(i, row['Take_Profit'], color='green', marker='_', s=100)
        
    ax2.bar(ticker_signals.index, ticker_signals['Confidence'], color='purple', alpha=0.7, label='Confidence Score')
    ax2.axhline(y=0.55, color='r', linestyle='--', label='Min Confidence Threshold')
    ax1.set_title(f'{ticker} Price with Trading Signals'); ax1.set_ylabel('Price'); ax1.legend(); ax1.grid(True)
    ax2.set_xlabel('Date'); ax2.set_ylabel('Confidence Score'); ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()


def analyze_performance(signals, forward_period):
    """
    Analyzes the backtested performance of the trading signals.

    Args:
        signals (pd.DataFrame): DataFrame containing generated signals and 'Future_Return'.
        forward_period (int): The forward period used for calculating returns (now intervals).

    Returns:
        dict: A dictionary of performance metrics (win rate, average win/loss, profit factor).
    """
    # Filter for actual trades (where a BUY signal was generated)
    actual_trades = signals[signals['Signal'] == 1].copy()
    
    if actual_trades.empty:
        print("\nNo trades were generated to analyze performance.")
        return {}
    
    # Drop NaNs from Future_Return for performance analysis
    actual_trades = actual_trades.dropna(subset=['Future_Return'])

    if actual_trades.empty:
        print("\nNo complete trades with Future_Return available for performance analysis.")
        return {}

    # Calculate performance metrics
    # Win: Future_Return > 0
    # Loss: Future_Return < 0
    # Break-even: Future_Return == 0 (can be counted as neither win nor loss, or grouped with losses)
    
    win_count = (actual_trades['Future_Return'] > 0).sum()
    loss_count = (actual_trades['Future_Return'] < 0).sum()
    
    total_trades = len(actual_trades)
    
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    avg_win = actual_trades[actual_trades['Future_Return'] > 0]['Future_Return'].mean()
    avg_loss = actual_trades[actual_trades['Future_Return'] < 0]['Future_Return'].mean()
    
    total_wins = actual_trades[actual_trades['Future_Return'] > 0]['Future_Return'].sum()
    total_losses = actual_trades[actual_trades['Future_Return'] < 0]['Future_Return'].sum() # Keep negative sign for summing

    # Profit Factor: (Sum of Gross Profits) / (Sum of Gross Losses)
    # Use abs() for total_losses in the denominator for the profit factor calculation
    profit_factor = total_wins / abs(total_losses) if total_losses < 0 else float('inf') # Avoid div by zero if no losses

    print(f"\nPerformance Analysis (Forward Period: {forward_period} intervals)") # Changed to intervals
    print(f"Number of Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}") # This will correctly show negative value
    print(f"Total Gross Profit: {total_wins:.2%}")
    print(f"Total Gross Loss: {total_losses:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # --- Visualization: Distribution of Trade Returns ---
    plt.figure(figsize=(12, 6))
    sns.histplot(actual_trades['Future_Return'], kde=True, bins=30, color='skyblue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Return')
    plt.title('Distribution of Trade Returns', fontsize=16)
    plt.xlabel(f'Return over {forward_period} Intervals', fontsize=12) # Changed to intervals
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    # --- Visualization: Model Confidence vs Actual Returns ---
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_trades['Confidence'], actual_trades['Future_Return'], alpha=0.6, s=50, color='darkgreen')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero Return Line')
    plt.xlabel('Model Confidence (Probability of Target=1)', fontsize=12)
    plt.ylabel(f'Actual Return over {forward_period} Intervals', fontsize=12) # Changed to intervals
    plt.title('Model Confidence vs Actual Returns for BUY Signals', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    return {
        'win_rate': win_rate, 
        'avg_win': avg_win, 
        'avg_loss': avg_loss, 
        'profit_factor': profit_factor,
        'total_trades': total_trades
    }

def main(all_tickers, train_start_date, train_end_date, test_start_date, 
         test_end_date, forward_period, min_confidence, force_download, 
         unseen_test_ticker=None, interval_data='1d'): # Added interval_data parameter
    """
    Main function to run the entire trading signal generation and analysis pipeline.
    Supports splitting data for unseen ticker testing.

    Args:
        all_tickers (list): A list of all tickers, including the one for unseen testing if applicable.
        train_start_date (str): Start date for training data.
        train_end_date (str): End date for training data.
        test_start_date (str): Start date for testing/backtesting data.
        test_end_date (str): End date for testing/backtesting data (can be current date).
        forward_period (int): Number of intervals into the future for target calculation.
        min_confidence (float): Minimum confidence for a BUY signal.
        force_download (bool): Flag to force data download, ignoring cache.
        unseen_test_ticker (str, optional): A ticker to exclude from training data
                                            but include in testing data. Defaults to None.
        interval_data (str): The interval for data fetching (e.g., '1d', '15m').

    Returns:
        tuple: (model, scaler, signals, performance_metrics, features_trained_on)
               Returns (None, None, None, None, None) if data fetching fails.
    """
    
    # --- 1. Prepare Training Data ---
    print("--- Preparing Training Data ---")
    # Exclude unseen_test_ticker from training data if specified
    train_tickers = [t for t in all_tickers if t != unseen_test_ticker]
    if not train_tickers:
        print("Error: No tickers left for training after excluding unseen_test_ticker.")
        return None, None, None, None, None

    train_data = fetch_and_prepare_data(
        train_tickers, train_start_date, train_end_date, interval=interval_data, force_download=force_download
    )
    
    if train_data.empty:
        print("Could not fetch sufficient data for training. Exiting.")
        return None, None, None, None, None

    train_data = calculate_selected_features(train_data)
    train_data = create_prediction_targets(train_data, forward_period)
    train_data, selected_features = select_features_for_model(train_data)

    # --- 2. Prepare Testing Data ---
    print("\n--- Preparing Testing Data ---")
    # Test data always includes all tickers (for comprehensive backtesting and unseen ticker evaluation)
    test_data = fetch_and_prepare_data(
        all_tickers, test_start_date, test_end_date, interval=interval_data, force_download=force_download
    )
    
    if test_data.empty:
        print("Could not fetch sufficient data for testing. Exiting.")
        return None, None, None, None, None

    test_data = calculate_selected_features(test_data)
    test_data = create_prediction_targets(test_data, forward_period)
    test_data, _ = select_features_for_model(test_data) # Use the same feature selection logic

    # Ensure common features are used for both training and testing X_scaled data
    # (features list returned by select_features_for_model should be used)
    common_features_for_model = [f for f in selected_features if f in test_data.columns]
    if not common_features_for_model:
        print("Error: No common features for model training/prediction. Check feature selection.")
        return None, None, None, None, None
    
    # Filter train_data to only include common features, in case some were not generated for all training tickers
    train_data_filtered_features = train_data[common_features_for_model + ['Target', 'Future_Return', 'Ticker', 'Close']].copy()

    # --- 3. Model Training ---
    print("\n--- Model Training ---")
    model, scaler, features_trained_on = train_prediction_model(train_data_filtered_features, common_features_for_model)
    
    # --- 4. Generate Trading Signals on Test Data ---
    print("\n--- Generating Trading Signals on Test Data ---")
    signals = generate_trading_signals(model, scaler, test_data, features_trained_on, min_confidence)
    
    # --- 5. Visualize Signals (for all tickers in the test set) ---
    print("\n--- Visualizing Signals ---")
    for ticker in all_tickers: # Visualize all tickers in the test set
        visualize_signals(signals, ticker)
    
    # --- 6. Analyze Performance ---
    print("\n--- Analyzing Performance ---")
    overall_performance = analyze_performance(signals, forward_period)
    
    if unseen_test_ticker and not signals.empty:
        print(f"\n--- Detailed Performance for Unseen Ticker: {unseen_test_ticker} ---")
        unseen_ticker_signals = signals[signals['Ticker'] == unseen_test_ticker].copy()
        if not unseen_ticker_signals.empty:
            analyze_performance(unseen_ticker_signals, forward_period)
        else:
            print(f"No signals or data for unseen ticker {unseen_test_ticker} in the test period.")
            
        print(f"\n--- Detailed Performance for Seen Tickers (excluding {unseen_test_ticker}) ---")
        seen_tickers_signals = signals[signals['Ticker'] != unseen_test_ticker].copy()
        if not seen_tickers_signals.empty:
            analyze_performance(seen_tickers_signals, forward_period)
        else:
            print(f"No signals or data for seen tickers in the test period (after excluding {unseen_test_ticker}).")
    
    print("\n--- Process Complete ---")
    return model, scaler, signals, overall_performance, features_trained_on # NEW: Return features_trained_on

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define your cryptocurrencies and some volatile stocks for comprehensive testing
    all_crypto_tickers = [
         # Reduced crypto tickers
    ]
    all_stock_tickers = [
        'TSLA', 'NVDA', 'AMD', 'GOOGL', 'AAPL', 'AMZN', 'PFE', 'CSCO' # Reduced stock tickers
    ]
    
    # Combine all tickers for the overall test run
    # Remember: if unseen_test_ticker is set, it will be excluded from the training set only.
    ALL_TICKERS_FOR_RUN = all_crypto_tickers + all_stock_tickers
    
    # --- Configuration ---
    # Set this to True to force re-download all data, ignoring the cache.
    # Set to False to use cached data if available.
    FORCE_API_DOWNLOAD_FOR_BACKTEST = False 
    
    # Set this to True to force re-download for the *live* signal generation
    # (Highly recommended to be True for live signals)
    FORCE_API_DOWNLOAD_FOR_LIVE_SIGNALS = True

    # Define the minimum confidence threshold for a BUY signal
    MIN_CONFIDENCE_THRESHOLD = 0.55 # New variable defined here

    # Define the ticker you want to exclude from TRAINING but include in TESTING
    # Set to None if you want to train on and test on all tickers provided
    UNSEEN_TICKER_FOR_TEST = 'PLUG' # Example: Test PLUG as an unseen stock.

    FORWARD = 4

    # NOTE: 'PLUG' is currently not in ALL_TICKERS_FOR_RUN. If you intend to test 'PLUG'
    # as an unseen ticker, please ensure it's added to ALL_TICKERS_FOR_RUN.
    # Otherwise, set UNSEEN_TICKER_FOR_TEST to None or to one of the tickers in the list.
    if UNSEEN_TICKER_FOR_TEST and UNSEEN_TICKER_FOR_TEST not in ALL_TICKERS_FOR_RUN:
        print(f"Warning: UNSEEN_TICKER_FOR_TEST '{UNSEEN_TICKER_FOR_TEST}' is not in the ALL_TICKERS_FOR_RUN list. Setting it to None.")
        UNSEEN_TICKER_FOR_TEST = None

    try:
        # Calculate dates for 15-minute intervals
        today = datetime.now()-timedelta(days=3)
        
        # Test data: Last 10 days
        test_end_date_intraday = today.strftime('%Y-%m-%d')
        test_start_date_intraday = (today - timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Train data: 20 days prior to the test period
        train_end_date_intraday = (datetime.strptime(test_start_date_intraday, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        train_start_date_intraday = (datetime.strptime(train_end_date_intraday, '%Y-%m-%d') - timedelta(days=20)).strftime('%Y-%m-%d')

        # --- Stage 1: Backtesting and Model Training (Historical Data) ---
        # This stage trains the model and evaluates its performance on a historical test set.
        # The test_end_date here is historical to allow for 'Future_Return' calculation.
        print("\n" + "="*50)
        print("STAGE 1: TRAINING MODEL & BACKTESTING HISTORICAL PERFORMANCE (15-MINUTE DATA)")
        print("="*50)
        
        # Modified assignment to capture features_trained_on
        trained_model, trained_scaler, historical_signals, historical_performance, features_trained_on = main(
            all_tickers=ALL_TICKERS_FOR_RUN,
            train_start_date=train_start_date_intraday,
            train_end_date=train_end_date_intraday, 
            test_start_date=test_start_date_intraday,
            test_end_date=test_end_date_intraday,
            forward_period=FORWARD, # Predicting 3 15-minute intervals (45 minutes) into the future
            min_confidence=MIN_CONFIDENCE_THRESHOLD, # Use the defined constant
            force_download=FORCE_API_DOWNLOAD_FOR_BACKTEST,
            unseen_test_ticker=UNSEEN_TICKER_FOR_TEST,
            interval_data='15m' # Specify 15-minute intervals
        )
        
        print("\n" + "="*50)
        print("STAGE 1 COMPLETE: HISTORICAL ANALYSIS CONCLUDED")
        print("="*50 + "\n")

        # --- Save Historical Signals to File (like trading_model.py) ---
        if historical_signals is not None and not historical_signals.empty:
            # Ensure Signal column is integer for backtester compatibility
            historical_signals['Signal'] = historical_signals['Signal'].astype(int)
            print("\n--- Saving Final Signals ---")
            SIGNALS_CACHE_DIR = 'signals_cache'
            if not os.path.exists(SIGNALS_CACHE_DIR):
                os.makedirs(SIGNALS_CACHE_DIR)
            signals_filename = os.path.join(
                SIGNALS_CACHE_DIR,
                f"signals_{test_start_date_intraday}_to_{test_end_date_intraday}.pkl"
            )
            # Delete all files in the signals cache directory
            for filename in os.listdir(SIGNALS_CACHE_DIR):
                file_path = os.path.join(SIGNALS_CACHE_DIR, filename)
                os.remove(file_path)
            print(f"Saving signals to: {signals_filename}")
            historical_signals.to_pickle(signals_filename)
            print("Signals saved successfully.")

        # --- Stage 2: Generate Live Trading Signals (Using the Trained Model) ---
        # This stage applies the trained model to the most recent data available (up to today).
        # This is for identifying potential trades right now.
        if trained_model is not None and trained_scaler is not None:
            # Check if features_trained_on is defined (it should be if Stage 1 completed successfully)
            if 'features_trained_on' not in locals() or features_trained_on is None:
                print("Error: 'features_trained_on' not available from Stage 1. Cannot proceed with live signal generation.")
            else:
                print("\n" + "="*50)
                print("STAGE 2: GENERATING LIVE TRADING SIGNALS (15-MINUTE DATA)")
                print("="*50)

                # For live signals, fetch only the most recent data needed (e.g., last 1-2 days of 15m data)
                # This ensures the features are calculated on fresh data for current predictions.
                live_data_fetch_start = (today - timedelta(days=2)).strftime('%Y-%m-%d') # Fetch last 2 days of 15m data
                live_data_fetch_end = today.strftime('%Y-%m-%d') # Up to today

                # Skipping live data fetch to avoid rate limit
                # Proceed directly to signal generation using cached/historical data
                current_market_data = pd.DataFrame()
                live_signals = pd.DataFrame()
                
                if not current_market_data.empty:
                    current_market_data = calculate_selected_features(current_market_data)
                    # Create prediction targets for the *live* data is not strictly necessary for generating signals
                    # but it ensures the DataFrame structure is consistent. Future_Return/Target will be NaN for the latest intervals.
                    current_market_data = create_prediction_targets(current_market_data, forward_period=FORWARD) 
                    current_market_data, live_features = select_features_for_model(current_market_data)

                    # Ensure only relevant features are passed and handle any missing columns if live_features differs from features_trained_on
                    # This ensures the model receives data in the format it was trained on.
                    missing_live_features = [f for f in features_trained_on if f not in current_market_data.columns]
                    if missing_live_features:
                        print(f"Warning: Missing features in live data for model prediction: {missing_live_features}. These will be filled with appropriate values.")
                        # This part of the code now relies on the more robust cleaning in generate_trading_signals
                        # No need for explicit filling here, as generate_trading_signals handles it.
                    
                    # Filter current_market_data to only include features the model was trained on
                    # This step is critical to ensure the DataFrame passed to generate_trading_signals
                    # contains only the columns the model expects, even if some values are NaN for now.
                    current_market_data_filtered = current_market_data[features_trained_on + ['Ticker', 'Close']].copy()

                    live_signals = generate_trading_signals(
                        trained_model, trained_scaler, current_market_data_filtered, features_trained_on, MIN_CONFIDENCE_THRESHOLD # Use the defined constant
                    )

                    # --- NEW: Save Live Signals to File ---
                    if live_signals is not None and not live_signals.empty:
                        print("\n--- Saving Final Signals ---")
                        SIGNALS_CACHE_DIR = 'signals_cache'
                        if not os.path.exists(SIGNALS_CACHE_DIR):
                            os.makedirs(SIGNALS_CACHE_DIR)
                        
                        # Create a consistent filename for the signals
                        signals_filename = os.path.join(
                            SIGNALS_CACHE_DIR,
                            f"signals_{test_start_date_intraday}_to_{test_end_date_intraday}.pkl"
                        )
                        
                        # Delete all files in the signals cache directory
                        for filename in os.listdir(SIGNALS_CACHE_DIR):
                            file_path = os.path.join(SIGNALS_CACHE_DIR, filename)
                            os.remove(file_path)

                        print(f"Saving signals to: {signals_filename}")
                        live_signals.to_pickle(signals_filename)
                        print("Signals saved successfully.")
                    # --- END OF NEW CODE ---
                    
                    print("\n--- Latest Trading Signals (Live) ---")
                    if not live_signals.empty:
                        # Filter for the absolute latest interval for each ticker that has a signal
                        latest_signals_by_ticker = live_signals.groupby('Ticker').tail(1)
                        
                        if not latest_signals_by_ticker.empty:
                            for _, row in latest_signals_by_ticker.iterrows():
                                # Re-calculate precision for display
                                display_precision = get_display_precision(pd.Series([row['Close']]))

                                if row['Signal'] == 1:
                                    print(f"📈 BUY {row['Ticker']} at {row.name.strftime('%Y-%m-%d %H:%M:%S')} with {row['Position_Size_Pct']:.2f}% of portfolio")
                                    print(f"  - Confidence: {row['Confidence']:.2%}")
                                    print(f"  - Entry Price: ${row['Close']:.{display_precision}f}")
                                    if not pd.isna(row['Stop_Loss']):
                                        print(f"  - Stop Loss: ${row['Stop_Loss']:.{display_precision}f} ({((row['Stop_Loss']/row['Close'])-1)*100:.2f}%)")
                                    if not pd.isna(row['Take_Profit']):
                                        print(f"  - Take Profit: ${row['Take_Profit']:.{display_precision}f} ({((row['Take_Profit']/row['Close'])-1)*100:.2f}%)")
                                    if not pd.isna(row['Risk_Reward']):
                                        print(f"  - Risk/Reward: {row['Risk_Reward']:.2f}:1")
                                    print("")
                                elif row['Confidence'] >= 0.5: # Show if high confidence but not meeting threshold
                                    print(f"🟡 NEUTRAL/WATCH {row['Ticker']} at {row.name.strftime('%Y-%m-%d %H:%M:%S')} (Confidence: {row['Confidence']:.2%})")
                                    print("")
                                else:
                                    print(f"⚪️ NO SIGNAL {row['Ticker']} at {row.name.strftime('%Y-%m-%d %H:%M:%S')} (Confidence: {row['Confidence']:.2%})")
                                    print("")
                        else:
                            print("No active signals on the latest intervals across all tickers.")
                    else:
                        print("No live signals generated.")
                    
                print("\n" + "="*50)
                print("STAGE 2 COMPLETE: LIVE SIGNAL GENERATION CONCLUDED")
                print("="*50 + "\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging