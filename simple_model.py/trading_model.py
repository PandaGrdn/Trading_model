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
from datetime import datetime
import os
import pickle

# --- Caching Configuration ---
CACHE_DIR = 'data_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Function to fetch and prepare data with caching
def fetch_and_prepare_data(tickers, period='1y', interval='1d', start_date=None, end_date=None, force_download=False):
    """
    Fetch historical data for cryptocurrencies and calculate key features.
    Includes caching to avoid re-downloading data.
    """
    # Sanitize dates for filename
    start_str = start_date.replace('-', '') if start_date else 'none'
    end_str = end_date.replace('-', '') if end_date else 'none'
    
    # Generate a unique filename for the cache
    cache_filename = os.path.join(CACHE_DIR, f"{'_'.join(tickers)}_{start_str}_to_{end_str}_{interval}.pkl")

    # --- Caching Logic ---
    if not force_download and os.path.exists(cache_filename):
        print(f"Loading data from cache: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            all_data = pickle.load(f)
        return all_data

    print("Fetching data from API...")
    all_data = pd.DataFrame()
    
    for ticker in tickers:
        print(f"  -> Fetching {ticker}...")
        crypto = yf.Ticker(ticker)
        data = crypto.history(period=period, interval=interval, start=start_date, end=end_date)
        
        if data.empty:
            print(f"     No data found for {ticker}. Skipping.")
            continue
            
        # Add ticker column
        data['Ticker'] = ticker
        
        # Basic price and volume features
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Price_Volume_Ratio'] = data['Close'] / (data['Volume'] + 1)
        
        # Add to main dataframe
        all_data = pd.concat([all_data, data], axis=0)
    
    # Drop NaNs after calculating basic features
    all_data = all_data.dropna()

    # Save the fetched data to cache
    print(f"Saving data to cache: {cache_filename}")
    with open(cache_filename, 'wb') as f:
        pickle.dump(all_data, f)
    
    return all_data

def calculate_selected_features(data):
    """
    Calculate only the selected top technical indicators from different feature groups
    """
    print("Calculating selected technical indicators...")
    grouped = data.groupby('Ticker')
    result = pd.DataFrame()
    
    for ticker, group in grouped:
        df = group.copy()
        
        # === Moving Averages Group ===
        df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # === Bollinger Bands Group ===
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'], timeperiod=20)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9)
        
        # === Ichimoku Group (1 representative) ===
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
        
        result = pd.concat([result, df], axis=0)
    
    return result

def create_prediction_targets(data, forward_period=3):
    """
    Create prediction targets for a specified forward period (e.g., 3 days)
    """
    data['Future_Return'] = data.groupby('Ticker')['Close'].pct_change(forward_period).shift(-forward_period)
    data['Target'] = (data['Future_Return'] >= 0.0).astype(int)
    data['Return_Magnitude'] = data['Future_Return'].abs()
    return data

def select_features_for_model(data):
    """
    Select the best features while avoiding redundancy
    """
    selected_features = [
        'BB_Upper', 'SMA_20', 'VWAP', 'EMA_12', 'Ichimoku_Conversion', 
        'Parabolic_SAR', 'ATR', 'SMA_50', 'Price_Volume_Ratio', 'High_Low_Range',
        'Selling_Pressure', 'Buying_Pressure', 'OBV', 'RSI', 'MACD',
        'Support_Proximity', 'Resistance_Proximity'
    ]
    
    data['SMA20_Ratio'] = data['Close'] / data['SMA_20']
    data['SMA50_Ratio'] = data['Close'] / data['SMA_50']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-9)
    data['EMA_Ratio'] = data['EMA_12'] / data['EMA_26']
    
    selected_features.extend(['SMA20_Ratio', 'SMA50_Ratio', 'BB_Position', 'EMA_Ratio'])
    return data, selected_features

def train_prediction_model(data, features, test_size=0.2):
    """
    Train an XGBoost model to predict price direction with confidence scores
    """
    model_data = data.dropna(subset=features + ['Target', 'Future_Return'])
    tscv = TimeSeriesSplit(n_splits=5)
    X = model_data[features]
    y = model_data['Target']
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    all_predictions, all_actual = [], []
    
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=1.0, objective='binary:logistic',
            eval_metric='auc', random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_predictions.extend(y_pred_proba)
        all_actual.extend(y_test)
    
    final_model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=1.0, objective='binary:logistic',
        eval_metric='auc', random_state=42
    )
    final_model.fit(X_scaled, y)
    
    auc_score = roc_auc_score(all_actual, all_predictions)
    print(f"Overall ROC AUC: {auc_score:.4f}")
    
    binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
    print("\nClassification Report:")
    print(classification_report(all_actual, binary_predictions))
    
    cm = confusion_matrix(all_actual, binary_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(final_model, max_num_features=20)
    plt.title('Feature Importance')
    plt.show()
    
    return final_model, scaler, features

def calculate_position_size(confidence, max_position_pct=8.0, min_confidence=0.55):
    """
    Calculates an aggressive position size based on model confidence.
    """
    if confidence < min_confidence:
        return 0.0
    scaled_confidence = (confidence - min_confidence) / (1.0 - min_confidence)
    base_position_size = scaled_confidence * max_position_pct
    aggression_multiplier = 1.5
    aggressive_size = base_position_size * aggression_multiplier
    final_position_size = min(aggressive_size, max_position_pct)

    return final_position_size

def calculate_risk_metrics(data, price, confidence, volatility_metric='Normalized_ATR'):
    """
    Calculates highly adaptive and robust risk levels by blending multiple methods
    and using confidence to adjust tolerance for volatility.
    """
    if data.empty or len(data) < 50: # Increased history requirement
        # Fallback for insufficient data
        stop_loss = price * 0.85 # Default 15% stop
        take_profit = price * 1.30 # Default 30% gain
        return {'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward_ratio': 2.0}

    # --- Step 1: Calculate multiple risk indicators ---
    
    # Method A: ATR-based stop distance
    atr_value = data['ATR'].iloc[-1]
    atr_stop_distance = atr_value * 3.0 # Use a wider ATR multiplier
    
    # Method B: Bollinger Band based stop distance
    bb_lower = data['BB_Lower'].iloc[-1]
    bb_stop_distance = price - bb_lower if price > bb_lower else price * 0.15 # Use distance to lower band

    # --- Step 2: Determine the base risk ---
    # Use the larger of the two distances to be more conservative and give the trade more room.
    # This places the stop below both the ATR level and the lower Bollinger Band.
    base_stop_distance = max(atr_stop_distance, bb_stop_distance)
    
    # --- Step 3: Adjust stop based on confidence ---
    # Higher confidence means we have stronger conviction, so we tolerate more volatility (wider stop).
    # This is a key change from typical models.
    # Confidence from 0.5 -> 1.0; Multiplier from 1.0 -> 1.5
    confidence_multiplier = 1.0 + (confidence - 0.5)
    
    adjusted_stop_distance = base_stop_distance * confidence_multiplier

    # --- Step 4: Enforce a maximum risk cap ---
    # Safety brake: Never risk more than 15% of the entry price on a single trade.
    max_risk_pct = 0.15 
    max_risk_distance = price * max_risk_pct
    
    final_stop_distance = min(adjusted_stop_distance, max_risk_distance)

    # --- Step 5: Final SL and TP calculation with a minimum R:R ---
    stop_loss = price - final_stop_distance
    
    min_risk_reward_ratio = 2.0
    required_profit_distance = final_stop_distance * min_risk_reward_ratio
    take_profit = price + required_profit_distance
    
    return {'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward_ratio': min_risk_reward_ratio}

def generate_trading_signals(model, scaler, data, features, min_confidence=0.55):
    """
    Generates trading signals with correctly calculated, ticker-specific risk metrics.
    This version uses a precise assignment method to prevent write errors.
    """
    signals = data.copy()
    X = signals[features].fillna(signals[features].mean())
    X_scaled = scaler.transform(X)
    signals['Confidence'] = model.predict_proba(X_scaled)[:, 1]
    signals['Signal'] = (signals['Confidence'] > min_confidence).astype(int)
    signals['Position_Size_Pct'] = signals['Confidence'].apply(lambda x: calculate_position_size(x, min_confidence=min_confidence))
    
    # Initialize columns to avoid errors later
    signals['Stop_Loss'] = np.nan
    signals['Take_Profit'] = np.nan
    signals['Risk_Reward'] = np.nan

    # --- DEFINITIVE FIX FOR CALCULATION AND ASSIGNMENT ---
    for i in range(len(signals)):
        if signals['Signal'].iloc[i] == 1:
            # 1. Get data for the current row
            current_ticker = signals['Ticker'].iloc[i]
            current_date = signals.index[i]
            
            # 2. Correctly filter for this ticker's history ONLY
            ticker_history = signals[(signals['Ticker'] == current_ticker) & (signals.index <= current_date)]
            
            if not ticker_history.empty:
                # 3. Calculate risk metrics using the clean history
                risk_metrics = calculate_risk_metrics(
                    ticker_history, 
                    signals['Close'].iloc[i], 
                    signals['Confidence'].iloc[i]
                )
                
                # 4. **THE NEW, ROBUST ASSIGNMENT FIX**
                # Use .iloc[row_position, column_position] for precise, unambiguous assignment.
                # This is the safest way to set a value in a DataFrame while iterating.
                signals.iloc[i, signals.columns.get_loc('Stop_Loss')] = risk_metrics['stop_loss']
                signals.iloc[i, signals.columns.get_loc('Take_Profit')] = risk_metrics['take_profit']
                signals.iloc[i, signals.columns.get_loc('Risk_Reward')] = risk_metrics['risk_reward_ratio']

    return signals

def visualize_signals(signals, ticker):
    ticker_signals = signals[signals['Ticker'] == ticker].copy()
    if ticker_signals.empty: return
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

def analyze_performance(signals, forward_period=3):
    actual_signals = signals[signals['Signal'] == 1].copy()
    if actual_signals.empty:
        print("\nNo trades were generated to analyze.")
        return {}
    
    actual_signals['Actual_Return'] = actual_signals['Future_Return']
    win_rate = (actual_signals['Actual_Return'] > 0).mean()
    avg_win = actual_signals[actual_signals['Actual_Return'] > 0]['Actual_Return'].mean()
    avg_loss = actual_signals[actual_signals['Actual_Return'] < 0]['Actual_Return'].mean()
    total_wins = actual_signals[actual_signals['Actual_Return'] > 0]['Actual_Return'].sum()
    total_losses = abs(actual_signals[actual_signals['Actual_Return'] < 0]['Actual_Return'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    print(f"\nPerformance Analysis (Forward Period: {forward_period} days)")
    print(f"Number of Trades: {len(actual_signals)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    plt.figure(figsize=(12, 6)); sns.histplot(actual_signals['Actual_Return'], kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--'); plt.title('Distribution of Trade Returns')
    plt.xlabel('Return'); plt.ylabel('Frequency'); plt.show()
    
    plt.figure(figsize=(10, 6)); plt.scatter(actual_signals['Confidence'], actual_signals['Actual_Return'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--'); plt.xlabel('Model Confidence'); plt.ylabel('Actual Return')
    plt.title('Model Confidence vs Actual Returns'); plt.grid(True); plt.show()
    
    return {'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor}

def main(tickers, train_start_date, train_end_date, test_start_date, test_end_date=None, 
         forward_period=3, min_confidence=0.55, force_download=False):
    """
    Main function to run the entire pipeline.
    Added force_download flag to control caching.
    """
    if test_end_date is None:
        test_end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("--- Preparing Training Data ---")
    train_data = fetch_and_prepare_data(
        tickers, start_date=train_start_date, end_date=train_end_date, interval='1d', force_download=force_download
    )
    
    print("\n--- Preparing Testing Data ---")
    test_data = fetch_and_prepare_data(
        tickers, start_date=test_start_date, end_date=test_end_date, interval='1d', force_download=force_download
    )
    
    if train_data.empty or test_data.empty:
        print("Could not fetch data for training or testing. Exiting.")
        return None, None, None, None

    print("\n--- Feature Engineering ---")
    train_data = calculate_selected_features(train_data)
    test_data = calculate_selected_features(test_data)
    
    print("\n--- Creating Prediction Targets ---")
    train_data = create_prediction_targets(train_data, forward_period)
    test_data = create_prediction_targets(test_data, forward_period)
    
    print("\n--- Selecting Final Features ---")
    train_data, selected_features = select_features_for_model(train_data)
    test_data, _ = select_features_for_model(test_data)
    
    print("\n--- Model Training ---")
    model, scaler, features = train_prediction_model(train_data, selected_features)
    
    print("\n--- Generating Trading Signals ---")
    signals = generate_trading_signals(model, scaler, test_data, features, min_confidence)
    
    print("\n--- Visualizing Signals ---")
    for ticker in tickers:
        visualize_signals(signals, ticker)
    
    print("\n--- Analyzing Performance ---")
    performance = analyze_performance(signals, forward_period)
    
    print("\n--- Process Complete ---")
    return model, scaler, signals, performance

if __name__ == "__main__":
    crypto_tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 
        'AVAX-USD', 'LINK-USD', 'DOGE-USD'
    ]
    
    # --- Configuration ---
    FORCE_API_DOWNLOAD = False # IMPORTANT: Keep this False!
    
    # --- Define Date Ranges ---
    TRAIN_START = '2022-01-01'
    TRAIN_END = '2024-12-31'
    TEST_START = '2025-01-01'
    # Use the current date for the test end date for consistency
    TEST_END = datetime.now().strftime('%Y-%m-%d')


    try:
        model, scaler, signals, performance = main(
            crypto_tickers, 
            train_start_date=TRAIN_START,
            train_end_date=TRAIN_END,
            test_start_date=TEST_START,
            test_end_date=TEST_END,
            forward_period=3,
            min_confidence=0.65,
            force_download=FORCE_API_DOWNLOAD
        )
        
        # --- NEW CODE TO SAVE SIGNALS ---
        if signals is not None and not signals.empty:
            print("\n--- Saving Final Signals ---")
            SIGNALS_CACHE_DIR = 'signals_cache'
            if not os.path.exists(SIGNALS_CACHE_DIR):
                os.makedirs(SIGNALS_CACHE_DIR)
            
            # Create a consistent filename for the signals
            signals_filename = os.path.join(SIGNALS_CACHE_DIR, f"signals_{TEST_START}_to_{TEST_END}.pkl")
            
            if os.path.exists(signals_filename):
                print(f"Signals file already exists: {signals_filename}")
                os.remove(signals_filename)

            print(f"Saving signals to: {signals_filename}")
            signals.to_pickle(signals_filename)
            print("Signals saved successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")