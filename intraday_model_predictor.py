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
def fetch_and_prepare_data(tickers, start_date=None, end_date=None, interval='30m'):
    """
    Fetch historical data for cryptocurrencies and calculate key features
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    """
    all_data = pd.DataFrame()
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=start_date, end=end_date, interval=interval)
        
        # Add ticker column
        data['Ticker'] = ticker
        
        # Basic price and volume features
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Price_Volume_Ratio'] = data['Close'] / (data['Volume'] + 1)
        
        # Add time-based features
        data = add_time_based_features(data)
        
        # Add to main dataframe
        all_data = pd.concat([all_data, data], axis=0)
    
    # Drop NaNs after calculating basic features
    all_data = all_data.dropna()
    
    return all_data

def add_time_based_features(data):
    """
    Add time-based features for intraday trading pattern detection
    """
    # Convert index to datetime if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Extract time components
    data['Hour'] = data.index.hour
    data['Minute'] = data.index.minute
    data['DayOfWeek'] = data.index.dayofweek  # Monday=0, Sunday=6
    
    # Create cyclical features for time (to handle the circular nature of time)
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    
    # Time-based volatility patterns
    grouped = data.groupby(['Ticker', 'Hour'])
    
    hour_volatility = {}
    for (ticker, hour), group in grouped:
        if (ticker, hour) not in hour_volatility:
            hour_volatility[(ticker, hour)] = group['Log_Return'].std()
    
    # Map hourly volatility back to the dataframe
    for i, row in data.iterrows():
        ticker = row['Ticker']
        hour = row['Hour']
        data.loc[i, 'Hour_Volatility'] = hour_volatility.get((ticker, hour), 0)
    
    return data

def add_mean_reversion_signal(data, lookback=20, z_threshold=2.0):
    """
    Add mean reversion signal based on Z-score of price deviation
    """
    grouped = data.groupby('Ticker')
    result = pd.DataFrame()
    
    for ticker, group in grouped:
        df = group.copy()
        
        # Calculate z-score of close price relative to moving average
        rolling_mean = df['Close'].rolling(window=lookback).mean()
        rolling_std = df['Close'].rolling(window=lookback).std()
        df['Price_Z_Score'] = (df['Close'] - rolling_mean) / rolling_std
        
        # Mean reversion signal (buy when extremely oversold, sell when extremely overbought)
        df['MR_Buy_Signal'] = (df['Price_Z_Score'] < -z_threshold).astype(int)
        
        result = pd.concat([result, df], axis=0)
    
    return result

def detect_market_condition(data, lookback=30):
    """
    Detect current market condition to filter trading decisions
    Returns: 'bullish', 'bearish', or 'sideways'
    """
    # Group by ticker
    grouped = data.groupby('Ticker')
    market_conditions = {}
    
    for ticker, group in grouped:
        # Get recent price data
        recent = group.iloc[-lookback:]
        
        # Calculate short and long term trends
        short_ma = recent['Close'].rolling(5).mean().iloc[-1]
        medium_ma = recent['Close'].rolling(15).mean().iloc[-1]
        long_ma = recent['Close'].rolling(30).mean().iloc[-1]
        
        # Calculate price volatility
        volatility = recent['Close'].pct_change().std()
        
        # Detect trend direction
        if short_ma > medium_ma > long_ma and short_ma/long_ma > 1.02:
            condition = 'bullish'
        elif short_ma < medium_ma < long_ma and long_ma/short_ma > 1.02:
            condition = 'bearish'
        else:
            condition = 'sideways'
            
        # Check if volatility is excessive (might want to avoid trading)
        if volatility > 0.03:  # 3% daily volatility is high
            condition += '_volatile'
            
        market_conditions[ticker] = condition
    
    return market_conditions

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

def train_ensemble_model(data, features, test_size=0.2):
    """
    Train an ensemble of models for more consistent predictions
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
    
    # Initialize models for ensemble
    models = {
        'xgb': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        ),
        'xgb_alt': xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=24
        ),
        'xgb_conservative': xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=84
        )
    }
    
    # Train each model and store
    trained_models = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        trained_models[name] = model
    
    return trained_models, scaler, features

def calculate_position_size(confidence, max_position_pct=3.0, min_confidence=0.60):
    """
    Calculate position size with more conservative approach for consistency
    """
    if confidence < min_confidence:
        return 0.0
    
    # Quadratic scaling to be more conservative with position sizing
    scaled_confidence = ((confidence - min_confidence) / (1.0 - min_confidence)) ** 2
    position_size = scaled_confidence * max_position_pct
    
    # Add portfolio-based position sizing limit
    return min(position_size, max_position_pct)

def calculate_risk_metrics(data, price, confidence, volatility_metric='Normalized_ATR'):
    """
    Calculate stop-loss and take-profit levels with improved risk management
    """
    # Get latest volatility value
    volatility = data[volatility_metric].iloc[-1]
    
    # Use recent volatility trends to adjust stop size (more dynamic)
    recent_volatility_ratio = data[volatility_metric].iloc[-5:].mean() / data[volatility_metric].iloc[-20:].mean()
    
    # Adjust ATR multiplier based on volatility trend
    if recent_volatility_ratio > 1.2:  # Increasing volatility
        atr_multiplier = 3.0  # Wider stop for increasing volatility
    elif recent_volatility_ratio < 0.8:  # Decreasing volatility
        atr_multiplier = 2.0  # Tighter stop for decreasing volatility
    else:
        atr_multiplier = 2.5  # Default
    
    # Scale stop distance based on confidence (higher confidence = tighter stop)
    confidence_factor = 1.0 - ((confidence - 0.5) * 0.3)
    
    # Calculate stop distance as multiple of ATR (adjusted for confidence)
    atr_stop_distance = volatility * price * atr_multiplier * confidence_factor
    
    # Conservative hard stop loss of 5% instead of 8%
    hard_stop_distance = price * 0.05
    
    # Use the tighter of the two stops
    stop_distance = min(atr_stop_distance, hard_stop_distance)
    
    # Calculate stop-loss level
    stop_loss = price - stop_distance
    
    # Risk-reward ratio with more conservative approach
    risk_reward_ratio = 1.5 + (confidence * 0.8)  # 1.9 at 50% confidence, 2.3 at 100%
    
    # Take profit levels
    take_profit = price + (stop_distance * risk_reward_ratio)
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward_ratio': risk_reward_ratio,
        'stop_distance_pct': (stop_distance / price) * 100
    }

def generate_ensemble_signals(models, scaler, data, features, min_confidence=0.55):
    """
    Generate trading signals based on ensemble predictions
    """
    # Make a copy of data to avoid modifying original
    signals = data.copy()
    
    # Prepare features
    X = signals[features].fillna(signals[features].mean())
    X_scaled = scaler.transform(X)
    
    # Get ensemble predictions
    ensemble_probs = np.zeros(len(X))
    for model in models.values():
        ensemble_probs += model.predict_proba(X_scaled)[:, 1]
    
    # Average the probabilities
    signals['Confidence'] = ensemble_probs / len(models)
    
    # Apply minimum confidence threshold
    signals['Signal'] = (signals['Confidence'] > min_confidence).astype(int)
    
    # Calculate position size with more conservative scaling
    signals['Position_Size_Pct'] = signals['Confidence'].apply(
        lambda x: calculate_position_size(x, max_position_pct=3.0, min_confidence=min_confidence)
    )
    
    # Add mean reversion signals
    data = add_mean_reversion_signal(data)
    
    # Combine signals (require both trend and mean-reversion for more consistency)
    signals['Combined_Signal'] = ((signals['Signal'] == 1) & (data['MR_Buy_Signal'] == 1)).astype(int)
    
    # Use the combined signal
    signals['Signal'] = signals['Combined_Signal']
    
    # Apply market condition filter
    market_conditions = detect_market_condition(data)
    
    for i in range(len(signals)):
        ticker = signals['Ticker'].iloc[i]
        market_condition = market_conditions.get(ticker, 'unknown')
        
        # Skip bearish volatile conditions completely
        if market_condition == 'bearish_volatile':
            signals.loc[signals.index[i], 'Signal'] = 0
            continue
            
        # Reduce position size in sideways markets
        if 'sideways' in market_condition:
            signals.loc[signals.index[i], 'Position_Size_Pct'] *= 0.7
    
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

def analyze_performance(signals, forward_period=3, position_size_pct=100):
    """
    Analyze performance of trading signals with improved metrics
    
    Args:
        signals (pd.DataFrame): DataFrame containing signals and predictions
        forward_period (int): Number of days to look forward for returns
        position_size_pct (float): Percentage of capital to use per trade (default 100%)
    """
    # Filter for signals only
    actual_signals = signals[signals['Signal'] == 1].copy()
    
    if actual_signals.empty:
        print("No signals to analyze")
        return {}
    
    # Calculate actual returns for the positions
    actual_signals['Actual_Return'] = actual_signals['Future_Return']
    
    # Calculate position size in dollars (assuming $10,000 initial capital for analysis)
    initial_capital = 10000
    position_size = initial_capital * (position_size_pct / 100)
    
    # Calculate dollar profits/losses
    actual_signals['Dollar_PnL'] = position_size * actual_signals['Actual_Return']
    
    # Calculate win rate
    win_rate = (actual_signals['Actual_Return'] > 0).mean()
    
    # Calculate average win and loss in both percentage and dollar terms
    winning_trades = actual_signals[actual_signals['Actual_Return'] > 0]
    losing_trades = actual_signals[actual_signals['Actual_Return'] < 0]
    
    avg_win_pct = winning_trades['Actual_Return'].mean() if not winning_trades.empty else 0
    avg_loss_pct = losing_trades['Actual_Return'].mean() if not losing_trades.empty else 0
    avg_win_dollar = winning_trades['Dollar_PnL'].mean() if not winning_trades.empty else 0
    avg_loss_dollar = losing_trades['Dollar_PnL'].mean() if not losing_trades.empty else 0
    
    # Calculate profit factor using dollar values
    total_wins = winning_trades['Dollar_PnL'].sum() if not winning_trades.empty else 0
    total_losses = abs(losing_trades['Dollar_PnL'].sum()) if not losing_trades.empty else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Calculate expectancy (average profit per trade)
    expectancy = (win_rate * avg_win_dollar) + ((1 - win_rate) * avg_loss_dollar)
    
    # Calculate risk metrics
    # Calculate portfolio value over time
    portfolio_value = initial_capital
    portfolio_values = []
    
    for _, signal in actual_signals.iterrows():
        portfolio_value += signal['Dollar_PnL']
        portfolio_values.append(portfolio_value)
    
    # Calculate max drawdown using peak-to-trough method
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
    daily_returns = actual_signals['Actual_Return'] / forward_period  # Convert to daily returns
    sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    # Calculate Sortino ratio (downside risk only)
    negative_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (daily_returns.mean() * 252) / (negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
    
    # Print results
    print("\nSignal Performance Analysis:")
    print(f"Number of Signals: {len(actual_signals)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win_pct:.2%} (${avg_win_dollar:.2f})")
    print(f"Average Loss: {avg_loss_pct:.2%} (${avg_loss_dollar:.2f})")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Expectancy: ${expectancy:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    
    # Return detailed metrics
    return {
        'num_signals': len(actual_signals),
        'win_rate': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'avg_win_dollar': avg_win_dollar,
        'avg_loss_dollar': avg_loss_dollar,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'total_wins': total_wins,
        'total_losses': total_losses
    }

def main(tickers, forward_period=3, min_confidence=0.45, initial_portfolio=10000):
    """
    Main function to run the entire prediction and trading signal generation
    
    Args:
        tickers (list): List of ticker symbols
        forward_period (int): Number of days to look forward for predictions
        min_confidence (float): Minimum confidence threshold for signals
        initial_portfolio (float): Initial portfolio value in dollars
    """
    # Set up date ranges
    end_date = datetime.now().strftime('%Y-%m-%d')  # Today
    test_start_date = '2025-01-01'  # Start of 2024
    train_start_date = '2022-01-01'  # Start of 2022 for training
    
    print("Fetching training data...")
    train_data = fetch_and_prepare_data(
        tickers, 
        start_date=train_start_date,
        end_date=test_start_date,
        interval='1d'
    )
    
    print("Fetching testing data...")
    test_data = fetch_and_prepare_data(
        tickers,
        start_date=test_start_date,
        end_date=end_date,
        interval='1d'
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
    models, scaler, features = train_ensemble_model(train_data, selected_features)
    
    # Generate trading signals for test period
    print("Generating trading signals...")
    signals = generate_ensemble_signals(models, scaler, test_data, features, min_confidence)
    
    # Visualize signals for each ticker
    for ticker in tickers:
        visualize_signals(signals, ticker)
    
    # Analyze performance
    performance = analyze_performance(signals, forward_period)
    
    # Calculate portfolio performance
    print("\nPortfolio Performance Analysis:")
    print(f"Initial Portfolio Value: ${initial_portfolio:,.2f}")
    
    # Calculate final portfolio value
    if not signals.empty:
        # Get all trades
        trades = signals[signals['Signal'] == 1].copy()
        
        # Calculate position sizes
        trades['Position_Size'] = initial_portfolio * (trades['Position_Size_Pct'] / 100)
        
        # Calculate P&L for each trade
        trades['PnL'] = trades['Position_Size'] * trades['Future_Return']
        
        # Calculate portfolio metrics
        total_trades = len(trades)
        avg_position_size = trades['Position_Size'].mean()
        
        # Calculate daily portfolio value
        daily_portfolio = pd.DataFrame(index=test_data.index.unique())
        daily_portfolio['Portfolio_Value'] = initial_portfolio
        
        for date in daily_portfolio.index:
            # Get all trades that were open on this date
            active_trades = trades[trades.index <= date]
            
            # Calculate current value of each trade
            current_values = []
            for _, trade in active_trades.iterrows():
                # Get the price on this date
                ticker_data = test_data[(test_data.index == date) & (test_data['Ticker'] == trade['Ticker'])]
                if not ticker_data.empty:
                    current_price = ticker_data['Close'].iloc[0]
                    entry_price = trade['Close']
                    position_value = trade['Position_Size'] * (current_price / entry_price)
                    current_values.append(position_value)
            
            # Update portfolio value
            if current_values:
                daily_portfolio.loc[date, 'Portfolio_Value'] = initial_portfolio + sum(current_values) - sum(active_trades['Position_Size'])
        
        # Get the final portfolio value from the daily tracking
        final_portfolio = daily_portfolio['Portfolio_Value'].iloc[-1]
        total_pnl = final_portfolio - initial_portfolio
        portfolio_return = (final_portfolio / initial_portfolio - 1) * 100
        
        print(f"Final Portfolio Value: ${final_portfolio:,.2f}")
        print(f"Total Return: {portfolio_return:.2f}%")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Number of Trades: {total_trades}")
        print(f"Average Position Size: ${avg_position_size:,.2f}")
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        daily_portfolio['Portfolio_Value'].plot()
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.show()
        
        # Save portfolio history
        daily_portfolio.to_csv('portfolio_history.csv')
        
        # Add portfolio metrics to performance dictionary
        performance.update({
            'initial_portfolio': initial_portfolio,
            'final_portfolio': final_portfolio,
            'total_return': portfolio_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'avg_position_size': avg_position_size
        })
    
    print("\nDone with main")
    
    return models, scaler, signals, performance

if __name__ == "__main__":
    # List of cryptocurrencies to analyze
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
    
    # Run the model with $10,000 initial portfolio
    models, scaler, signals, performance = main(
        crypto_tickers, 
        forward_period=3,  # 3-day prediction window
        min_confidence=0.45,  # Minimum confidence threshold
        initial_portfolio=10000  # Initial portfolio value
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