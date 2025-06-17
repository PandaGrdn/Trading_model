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

# Function to fetch and prepare data
def fetch_and_prepare_data(tickers, period='1y', interval='3d', start_date=None, end_date=None):
    """
    Fetch historical data for cryptocurrencies and calculate key features
    """
    all_data = pd.DataFrame()
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        crypto = yf.Ticker(ticker)
        data = crypto.history(period=period, interval=interval)
        
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
        # SMA features (selecting SMA_5, SMA_20 as representatives)
        df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        
        # EMA features (selecting EMA_12, EMA_26 as representatives)
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # === Bollinger Bands Group ===
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'], timeperiod=20)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # === Ichimoku Group (1 representative) ===
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
    
    return result

def create_prediction_targets(data, forward_period=3):
    """
    Create prediction targets for a specified forward period (e.g., 3 days)
    """
    # Calculate future returns
    data['Future_Return'] = data.groupby('Ticker')['Close'].pct_change(forward_period).shift(-forward_period)
    
    # Define binary target (1 if price goes up, 0 if it goes down)
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    
    # Define magnitude of movement (for position sizing later)
    data['Return_Magnitude'] = data['Future_Return'].abs()
    
    return data

def select_features_for_model(data):
    """
    Select the best features while avoiding redundancy
    """
    # Selected features from top 20, choosing representatives from similar feature groups
    selected_features = [
        'BB_Upper',        # Top Bollinger Band feature
        'SMA_20',          # Top SMA feature
        'VWAP',            # Top volume-weighted feature
        'EMA_12',          # Top EMA feature 
        'Ichimoku_Conversion', # Top Ichimoku feature
        'Parabolic_SAR',   # Top trend indicator
        'ATR',             # Top volatility feature
        'SMA_50',          # Additional trend feature
        'Price_Volume_Ratio', # Volume-price relationship
        'High_Low_Range',  # Price range feature
        'Selling_Pressure', # Supply-demand feature
        'Buying_Pressure', # Supply-demand feature
        'OBV',             # On-balance volume
        'RSI',             # Relative Strength Index
        'MACD',            # Moving Average Convergence Divergence
        'Support_Proximity', # Distance from support
        'Resistance_Proximity' # Distance from resistance
    ]
    
    # Calculate price-relative versions of some indicators to reduce collinearity
    data['SMA20_Ratio'] = data['Close'] / data['SMA_20']
    data['SMA50_Ratio'] = data['Close'] / data['SMA_50']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    data['EMA_Ratio'] = data['EMA_12'] / data['EMA_26']
    
    # Add these derived features to our selection
    selected_features.extend([
        'SMA20_Ratio',
        'SMA50_Ratio',
        'BB_Position',
        'EMA_Ratio'
    ])
    
    return data, selected_features

def train_prediction_model(data, features, test_size=0.2):
    """
    Train an XGBoost model to predict price direction with confidence scores
    """
    # Remove any NaN values
    model_data = data.dropna(subset=features + ['Target', 'Future_Return'])
    
    # Create time-series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Feature matrix and target
    X = model_data[features]
    y = model_data['Target']
    
    # Handle remaining NaN values with mean imputation
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with time series cross-validation
    all_predictions = []
    all_actual = []
    
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_predictions.extend(y_pred_proba)
        all_actual.extend(y_test)
    
    # Train final model on all data
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    
    final_model.fit(X_scaled, y)
    
    # Evaluate overall performance
    auc_score = roc_auc_score(all_actual, all_predictions)
    print(f"Overall ROC AUC: {auc_score:.4f}")
    
    # Convert predictions to binary classification for evaluation
    binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_actual, binary_predictions))
    
    # Print confusion matrix
    cm = confusion_matrix(all_actual, binary_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(final_model, max_num_features=20)
    plt.title('Feature Importance')
    plt.show()
    
    return final_model, scaler, features

def calculate_position_size(confidence, max_position_pct=5.0, min_confidence=0.55):
    """
    Calculate position size based on model confidence
    
    Parameters:
    -----------
    confidence : float
        Model's confidence score (0-1)
    max_position_pct : float
        Maximum position size as percentage of portfolio
    min_confidence : float
        Minimum confidence threshold to take a position
        
    Returns:
    --------
    float : Recommended position size as percentage of portfolio
    """
    if confidence < min_confidence:
        return 0.0
    
    # Scale position size based on confidence above threshold
    # Linear scaling: min_confidence -> 0%, 1.0 -> max_position_pct
    scaled_confidence = (confidence - min_confidence) / (1.0 - min_confidence)
    position_size = scaled_confidence * max_position_pct
    
    return min(position_size, max_position_pct)  # Cap at max_position_pct

def calculate_risk_metrics(data, price, confidence, volatility_metric='Normalized_ATR'):
    """
    Calculate stop-loss and take-profit levels based on volatility
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with volatility metrics
    price : float
        Current price
    confidence : float
        Model confidence score
    volatility_metric : str
        Column name for volatility metric
        
    Returns:
    --------
    dict : Dictionary with risk metrics
    """
    # Get latest volatility value
    volatility = data[volatility_metric].iloc[-1]
    
    # Scale stop distance based on confidence (higher confidence = tighter stop)
    confidence_factor = 1.0 - ((confidence - 0.5) * 0.5) if confidence > 0.5 else 1.0
    
    # Calculate stop distance as multiple of ATR
    stop_distance = volatility * price * 2.5 * confidence_factor
    
    # Calculate stop-loss and take-profit levels
    stop_loss = price - stop_distance
    
    # Risk-reward ratio scales with confidence
    risk_reward_ratio = 1.5 + confidence  # 2.0 at 50% confidence, 2.5 at 100% confidence
    take_profit = price + (stop_distance * risk_reward_ratio)
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward_ratio': risk_reward_ratio
    }

def generate_trading_signals(model, scaler, data, features, min_confidence=0.55):
    """
    Generate trading signals based on model predictions
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    scaler : StandardScaler
        Fitted scaler for features
    data : pd.DataFrame
        Data with features
    features : list
        List of feature names
    min_confidence : float
        Minimum confidence threshold for signals
        
    Returns:
    --------
    pd.DataFrame : DataFrame with trading signals
    """
    # Make a copy of data to avoid modifying original
    signals = data.copy()
    
    # Prepare features
    X = signals[features].fillna(signals[features].mean())
    X_scaled = scaler.transform(X)
    
    # Get model predictions and confidence scores
    signals['Confidence'] = model.predict_proba(X_scaled)[:, 1]
    signals['Signal'] = (signals['Confidence'] > min_confidence).astype(int)
    
    # Calculate position size for each signal
    signals['Position_Size_Pct'] = signals['Confidence'].apply(
        lambda x: calculate_position_size(x, max_position_pct=5.0, min_confidence=min_confidence)
    )
    
    # Calculate risk metrics for each signal
    for i in range(len(signals)):
        if signals['Signal'].iloc[i] == 1:
            current_price = signals['Close'].iloc[i]
            confidence = signals['Confidence'].iloc[i]
            
            risk_metrics = calculate_risk_metrics(
                signals.iloc[:i+1], 
                current_price, 
                confidence
            )
            
            signals.loc[signals.index[i], 'Stop_Loss'] = risk_metrics['stop_loss']
            signals.loc[signals.index[i], 'Take_Profit'] = risk_metrics['take_profit']
            signals.loc[signals.index[i], 'Risk_Reward'] = risk_metrics['risk_reward_ratio']
    
    return signals

def visualize_signals(signals, ticker):
    """
    Visualize trading signals on price chart
    """
    # Filter for the specific ticker
    ticker_signals = signals[signals['Ticker'] == ticker].copy()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price chart
    ax1.plot(ticker_signals.index, ticker_signals['Close'], label='Close Price', color='blue')
    
    # Plot Buy signals
    buy_signals = ticker_signals[ticker_signals['Signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
    
    # Add stop loss and take profit markers for buy signals
    for i, row in buy_signals.iterrows():
        # Stop loss
        ax1.plot([i, i], [row['Close'], row['Stop_Loss']], 'r--', alpha=0.5)
        ax1.scatter(i, row['Stop_Loss'], color='red', marker='_', s=100)
        
        # Take profit
        ax1.plot([i, i], [row['Close'], row['Take_Profit']], 'g--', alpha=0.5)
        ax1.scatter(i, row['Take_Profit'], color='green', marker='_', s=100)
    
    # Plot confidence scores
    ax2.bar(ticker_signals.index, ticker_signals['Confidence'], color='purple', alpha=0.7, label='Confidence Score')
    ax2.axhline(y=0.55, color='r', linestyle='--', label='Min Confidence Threshold')
    
    # Customize plot
    ax1.set_title(f'{ticker} Price with Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Confidence Score')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_performance(signals, forward_period=3):
    """
    Analyze performance of trading signals
    """
    # Filter for signals only
    actual_signals = signals[signals['Signal'] == 1].copy()
    
    # Calculate actual returns for the positions
    actual_signals['Actual_Return'] = actual_signals['Future_Return']
    
    # Calculate win rate
    win_rate = (actual_signals['Actual_Return'] > 0).mean()
    
    # Calculate average win and loss
    avg_win = actual_signals[actual_signals['Actual_Return'] > 0]['Actual_Return'].mean()
    avg_loss = actual_signals[actual_signals['Actual_Return'] < 0]['Actual_Return'].mean()
    
    # Calculate profit factor
    total_wins = actual_signals[actual_signals['Actual_Return'] > 0]['Actual_Return'].sum()
    total_losses = abs(actual_signals[actual_signals['Actual_Return'] < 0]['Actual_Return'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Print results
    print(f"\nPerformance Analysis (Forward Period: {forward_period} days)")
    print(f"Number of Trades: {len(actual_signals)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Plot return distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(actual_signals['Actual_Return'], kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Trade Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.show()
    
    # Plot confidence vs actual returns
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_signals['Confidence'], actual_signals['Actual_Return'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Model Confidence')
    plt.ylabel('Actual Return')
    plt.title('Model Confidence vs Actual Returns')
    plt.grid(True)
    plt.show()
    
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }

def main(tickers, train_start_date='2022-01-01', train_end_date='2023-12-31', 
         test_start_date='2024-01-01', test_end_date=None, forward_period=3, min_confidence=0.55):
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
    """
    # Set default test_end_date to current date if not provided
    if test_end_date is None:
        test_end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Fetching training data...")
    train_data = fetch_and_prepare_data(
        tickers, 
        start_date=train_start_date,
        end_date=train_end_date,
        interval='3d'
    )
    
    print("Fetching testing data...")
    test_data = fetch_and_prepare_data(
        tickers,
        start_date=test_start_date,
        end_date=test_end_date,
        interval='3d'
    )
    
    # Calculate features for both datasets
    print("Calculating features for training data...")
    train_data = calculate_selected_features(train_data)
    
    print("Calculating features for testing data...")
    test_data = calculate_selected_features(test_data)
    
    # Create prediction targets
    print("Creating prediction targets...")
    train_data = create_prediction_targets(train_data, forward_period)
    test_data = create_prediction_targets(test_data, forward_period)
    
    # Select features
    print("Selecting features...")
    train_data, selected_features = select_features_for_model(train_data)
    test_data, _ = select_features_for_model(test_data)
    
    # Train model
    print("Training model...")
    model, scaler, features = train_prediction_model(train_data, selected_features)
    
    # Generate trading signals for test period
    print("Generating trading signals...")
    signals = generate_trading_signals(model, scaler, test_data, features, min_confidence)
    
    # Visualize signals for each ticker
    for ticker in tickers:
        visualize_signals(signals, ticker)
    
    # Analyze performance
    performance = analyze_performance(signals, forward_period)
    
    print("\nDone with main")
    
    return model, scaler, signals, performance

if __name__ == "__main__":
    # List of cryptocurrencies to analyze
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 
                      'AVAX-USD', 'LINK-USD', 'DOGE-USD']
    
    # Run the model with specified date ranges
    model, scaler, signals, performance = main(
        crypto_tickers, 
        train_start_date='2024-011-01',  # Training data from 2022
        train_end_date='2024-12-31',    # Training data until end of 2023
        test_start_date='2025-01-01',   # Backtesting from start of 2024
        test_end_date=None,             # Backtesting until current date
        forward_period=3,               # 3-day prediction window
        min_confidence=0.55             # Minimum confidence threshold
    )
    
    # Print latest signals
    print("\nLatest Trading Signals:")
    latest_date = signals.index.max()
    latest_signals = signals[signals.index == latest_date]
    
    for _, row in latest_signals.iterrows():
        if row['Signal'] == 1:
            print(f"BUY {row['Ticker']} with {row['Position_Size_Pct']:.2f}% of portfolio")
            print(f"  - Confidence: {row['Confidence']:.2%}")
            print(f"  - Entry Price: ${row['Close']:.2f}")
            print(f"  - Stop Loss: ${row['Stop_Loss']:.2f} ({((row['Stop_Loss']/row['Close'])-1)*100:.2f}%)")
            print(f"  - Take Profit: ${row['Take_Profit']:.2f} ({((row['Take_Profit']/row['Close'])-1)*100:.2f}%)")
            print(f"  - Risk/Reward: {row['Risk_Reward']:.2f}")
            print()