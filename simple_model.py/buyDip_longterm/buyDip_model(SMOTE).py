import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
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

def calculate_dip_features(data):
    """
    Calculate features specifically designed to identify various types of dips
    """
    print("Calculating dip-specific features...")
    grouped = data.groupby('Ticker')
    result = pd.DataFrame()
    
    for ticker, group in grouped:
        df = group.copy()
        
        # === Dip Detection Features ===
        # Short-term dips (1-3 days)
        df['Drop_1d'] = df['Close'].pct_change(1)
        df['Drop_2d'] = df['Close'].pct_change(2)
        df['Drop_3d'] = df['Close'].pct_change(3)
        df['Drop_5d'] = df['Close'].pct_change(5)
        df['Drop_10d'] = df['Close'].pct_change(10)
        
        # Cumulative drops over different periods
        df['Max_Drop_5d'] = df['Close'].rolling(5).min() / df['Close'].shift(5) - 1
        df['Max_Drop_10d'] = df['Close'].rolling(10).min() / df['Close'].shift(10) - 1
        df['Max_Drop_20d'] = df['Close'].rolling(20).min() / df['Close'].shift(20) - 1
        
        # Distance from recent highs (key dip indicator)
        df['High_5d'] = df['High'].rolling(5).max()
        df['High_10d'] = df['High'].rolling(10).max()
        df['High_20d'] = df['High'].rolling(20).max()
        df['High_50d'] = df['High'].rolling(50).max()
        
        df['Distance_From_5d_High'] = (df['Close'] - df['High_5d']) / df['High_5d']
        df['Distance_From_10d_High'] = (df['Close'] - df['High_10d']) / df['High_10d']
        df['Distance_From_20d_High'] = (df['Close'] - df['High_20d']) / df['High_20d']
        df['Distance_From_50d_High'] = (df['Close'] - df['High_50d']) / df['High_50d']
        
        # Volatility-adjusted dips
        df['Volatility_5d'] = df['Log_Return'].rolling(5).std()
        df['Volatility_20d'] = df['Log_Return'].rolling(20).std()
        df['Normalized_Drop_5d'] = df['Drop_5d'] / (df['Volatility_20d'] + 1e-9)
        df['Normalized_Drop_10d'] = df['Drop_10d'] / (df['Volatility_20d'] + 1e-9)
        
        # Volume during dips (higher volume during dips can indicate capitulation)
        df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
        df['Volume_Spike'] = df['Volume'] / (df['Volume_MA_10'] + 1e-9)
        df['Volume_During_Drop'] = np.where(df['Daily_Return'] < -0.02, df['Volume_Spike'], 0)
        
        result = pd.concat([result, df], axis=0)
    
    return result

def calculate_selected_features(data):
    """
    Calculate technical indicators alongside dip features
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
        
        result = pd.concat([result, df], axis=0)
    
    return result

def create_dip_buying_targets(data, forward_period=5, dip_threshold=-0.03, recovery_threshold=0.05):
    """
    Create targets specifically for dip-buying strategy:
    - Identify when price has dropped (dip)
    - Check if it recovers within the forward period
    - Only create positive targets for dip situations that recover
    """
    print(f"Creating dip-buying targets with {forward_period}d forward period...")
    
    grouped = data.groupby('Ticker')
    result = pd.DataFrame()
    
    for ticker, group in grouped:
        df = group.copy()
        
        # Calculate future returns
        df['Future_Return'] = df['Close'].pct_change(forward_period).shift(-forward_period)
        df['Future_Max_Return'] = df['Close'].rolling(forward_period).max().shift(-forward_period) / df['Close'] - 1
        
        # Identify dip conditions
        # Condition 1: Recent price drop
        df['Is_Recent_Dip'] = (
            (df['Drop_2d'] <= dip_threshold) |  # 2-day drop
            (df['Drop_5d'] <= dip_threshold * 1.5) |  # 5-day drop (less severe)
            (df['Distance_From_10d_High'] <= dip_threshold * 1.2)  # Distance from recent high
        )
        
        # Condition 2: Oversold conditions (additional dip indicators)
        df['Is_Oversold'] = (
            (df['RSI'] <= 35) |  # RSI oversold
            (df['Close'] <= df['BB_Lower']) |  # Below lower Bollinger Band
            (df['Distance_From_20d_High'] <= -0.15)  # Significant drop from 20d high
        )
        
        # Condition 3: Volume confirmation (optional but helpful)
        df['Has_Volume_Confirmation'] = df['Volume_Spike'] >= 1.2
        
        # Combined dip condition
        df['Is_Dip_Opportunity'] = (
            df['Is_Recent_Dip'] & 
            (df['Is_Oversold'] | df['Has_Volume_Confirmation'])
        )
        
        # Create target: Only positive for dips that recover well
        df['Target'] = 0  # Default to no signal
        
        # Positive target: Dip that recovers beyond recovery threshold
        dip_recovery_condition = (
            df['Is_Dip_Opportunity'] & 
            (df['Future_Max_Return'] >= recovery_threshold)
        )
        df.loc[dip_recovery_condition, 'Target'] = 1
        
        # For training balance, also include some negative examples
        # Negative target: Dip that doesn't recover (continues dropping)
        dip_no_recovery_condition = (
            df['Is_Dip_Opportunity'] & 
            (df['Future_Return'] <= -0.02)  # Continues to drop
        )
        df.loc[dip_no_recovery_condition, 'Target'] = 0
        
        result = pd.concat([result, df], axis=0)
    
    # Calculate class distribution
    positive_signals = (result['Target'] == 1).sum()
    total_dip_opportunities = (result['Is_Dip_Opportunity'] == True).sum()
    
    print(f"Dip opportunities identified: {total_dip_opportunities}")
    print(f"Profitable dip opportunities: {positive_signals}")
    print(f"Dip success rate: {positive_signals/total_dip_opportunities*100:.1f}%" if total_dip_opportunities > 0 else "No dips found")
    
    return result

def select_dip_buying_features(data):
    """
    Select features optimized for dip-buying strategy
    """
    # Core dip detection features
    dip_features = [
        'Drop_1d', 'Drop_2d', 'Drop_3d', 'Drop_5d', 'Drop_10d',
        'Distance_From_5d_High', 'Distance_From_10d_High', 'Distance_From_20d_High',
        'Max_Drop_5d', 'Max_Drop_10d', 'Normalized_Drop_5d', 'Normalized_Drop_10d',
        'Volume_Spike', 'Volume_During_Drop'
    ]
    
    # Technical indicators that work well with dip-buying
    technical_features = [
        'RSI', 'BB_Width', 'MACD', 'ATR', 'Normalized_ATR',
        'Support_Proximity', 'Resistance_Proximity', 'Buying_Pressure', 'Selling_Pressure'
    ]
    
    # Create ratio features for better dip identification
    data['RSI_Oversold_Ratio'] = np.maximum(0, 35 - data['RSI']) / 35  # How oversold (0-1)
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-9)
    data['SMA20_Distance'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
    data['SMA50_Distance'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
    data['EMA_Ratio'] = data['EMA_12'] / (data['EMA_26'] + 1e-9)
    
    # Volatility-adjusted features
    data['Volatility_Adjusted_Drop'] = data['Drop_5d'] / (data['Volatility_20'] + 1e-9)
    data['ATR_Normalized_Distance'] = data['Distance_From_10d_High'] / (data['Normalized_ATR'] + 1e-9)
    
    ratio_features = [
        'RSI_Oversold_Ratio', 'BB_Position', 'SMA20_Distance', 'SMA50_Distance', 
        'EMA_Ratio', 'Volatility_Adjusted_Drop', 'ATR_Normalized_Distance'
    ]
    
    selected_features = dip_features + technical_features + ratio_features
    return data, selected_features

def apply_smote_sampling(X, y, sampling_strategy='auto', smote_method='standard'):
    """
    Apply SMOTE or its variants to balance the dataset
    
    Parameters:
    - X: Features
    - y: Target variable
    - sampling_strategy: 'auto', 'minority', dict, or float
    - smote_method: 'standard', 'borderline', 'adasyn', 'smote_tomek', 'smote_enn'
    """
    print(f"Original class distribution: {Counter(y)}")
    
    # Choose SMOTE method
    if smote_method == 'standard':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    elif smote_method == 'borderline':
        sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    elif smote_method == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42, n_neighbors=5)
    elif smote_method == 'smote_tomek':
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
    elif smote_method == 'smote_enn':
        sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    else:
        raise ValueError(f"Unknown SMOTE method: {smote_method}")
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"Resampled class distribution: {Counter(y_resampled)}")
        print(f"SMOTE method used: {smote_method}")
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"SMOTE failed with error: {e}")
        print("Falling back to original data...")
        return X, y

def train_dip_buying_model(data, features, test_size=0.2, use_smote=True, smote_method='standard'):
    """
    Train XGBoost model specifically for dip-buying with SMOTE balancing
    Also trains CNN model for ensemble predictions
    """
    model_data = data.dropna(subset=features + ['Target', 'Future_Return'])
    
    # Filter to only use data where we have dip opportunities for more focused training
    training_data = model_data[model_data['Is_Dip_Opportunity'] == True].copy()
    
    if len(training_data) < 100:
        print("Not enough dip opportunities for focused training, using all data...")
        training_data = model_data.copy()
    
    print(f"Training on {len(training_data)} samples with dip focus")
    
    # Train CNN model
   
    
    # Continue with existing XGBoost training...
    tscv = TimeSeriesSplit(n_splits=5)
    X = training_data[features]
    y = training_data['Target']
    X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    all_predictions, all_actual = [], []
    
    # Calculate class weights for imbalanced data (backup if SMOTE fails)
    pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold_idx + 1}/5")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Apply SMOTE to training data only
        if use_smote and len(np.unique(y_train)) > 1:
            print(f"Applying SMOTE to training fold {fold_idx + 1}...")
            X_train_balanced, y_train_balanced = apply_smote_sampling(
                X_train, y_train, 
                sampling_strategy='auto',  # Balance to majority class
                smote_method=smote_method
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print("SMOTE not applied (insufficient class diversity or disabled)")
        
        # Train model
        if use_smote:
            # When using SMOTE, we don't need scale_pos_weight as classes are balanced
            model = xgb.XGBClassifier(
                n_estimators=200,  # Increased for better performance
                learning_rate=0.02,  # Reduced for more stable learning
                max_depth=6, 
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='auc', 
                random_state=42,
                tree_method='hist'  # Faster training
            )
        else:
            # Use class weights when not using SMOTE
            model = xgb.XGBClassifier(
                n_estimators=150, 
                learning_rate=0.03, 
                max_depth=6, 
                subsample=0.8,
                colsample_bytree=0.8, 
                scale_pos_weight=pos_weight,
                objective='binary:logistic',
                eval_metric='auc', 
                random_state=42
            )
        
        # Fit model
        model.fit(X_train_balanced, y_train_balanced, 
                 eval_set=[(X_test, y_test)], 
                 verbose=False)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_predictions.extend(y_pred_proba)
        all_actual.extend(y_test)
    
    # Train final model on all data with SMOTE
    print("\nTraining final model on all data...")
    if use_smote and len(np.unique(y)) > 1:
        X_final_balanced, y_final_balanced = apply_smote_sampling(
            X_scaled, y, 
            sampling_strategy='auto',
            smote_method=smote_method
        )
    else:
        X_final_balanced, y_final_balanced = X_scaled, y
    
    # Final model configuration
    if use_smote:
        final_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.02,
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='auc', 
            random_state=42,
            tree_method='hist'
        )
    else:
        final_model = xgb.XGBClassifier(
            n_estimators=150, 
            learning_rate=0.03, 
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8, 
            scale_pos_weight=pos_weight,
            objective='binary:logistic',
            eval_metric='auc', 
            random_state=42
        )
    
    final_model.fit(X_final_balanced, y_final_balanced)
    
    # Model evaluation
    auc_score = roc_auc_score(all_actual, all_predictions)
    print(f"\nOverall ROC AUC: {auc_score:.4f}")
    
    # Use different thresholds for better classification
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1 = 0
    best_threshold = 0.5
    
    print("\n=== THRESHOLD ANALYSIS ===")
    for threshold in thresholds:
        binary_predictions = [1 if p > threshold else 0 for p in all_predictions]
        report = classification_report(all_actual, binary_predictions, output_dict=True)
        f1_score = report['1']['f1-score']
        precision = report['1']['precision']
        recall = report['1']['recall']
        
        print(f"Threshold {threshold:.1f}: F1={f1_score:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.1f} (F1-score: {best_f1:.3f})")
    
    # Final classification report with best threshold
    best_binary_predictions = [1 if p > best_threshold else 0 for p in all_predictions]
    print(f"\n=== FINAL CLASSIFICATION REPORT (Threshold: {best_threshold}) ===")
    print(classification_report(all_actual, best_binary_predictions))
    
    # Confusion Matrix
    cm = confusion_matrix(all_actual, best_binary_predictions)
    plt.figure(figsize=(10, 8))
    
    # Create subplots for confusion matrix and feature importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'Dip-Buying Model Confusion Matrix\n(Threshold: {best_threshold}, SMOTE: {use_smote})')
    
    # Feature Importance
    xgb.plot_importance(final_model, max_num_features=20, ax=ax2)
    ax2.set_title('Dip-Buying Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    return final_model, scaler, features, best_threshold


def calculate_position_size(confidence, max_position_pct=8.0, min_confidence=0.60):
    """
    Calculate position size with higher minimum confidence for dip-buying
    """
    if confidence < min_confidence:
        return 0.0
    
    scaled_confidence = (confidence - min_confidence) / (1.0 - min_confidence)
    base_position_size = scaled_confidence * max_position_pct
    
    # More conservative for dip-buying
    conservative_multiplier = 1.2
    conservative_size = base_position_size * conservative_multiplier
    final_position_size = min(conservative_size, max_position_pct)

    return final_position_size

def calculate_risk_metrics(data_slice, price, confidence, min_risk_reward_ratio=1.5): # min_risk_reward_ratio is now a default
    """
    Calculates a CONSERVATIVE risk profile based on fixed rules.
    - Stop-Loss is strictly capped at a maximum of 2% loss.
    - Take-Profit is fixed at a 3% gain.
    """
    # --- The original dynamic calculation is kept as a potential input ---
    # This helps determine if the stop should be even TIGHTER than 2%, but never wider.
    atr_value = data_slice['ATR'].iloc[-1] if 'ATR' in data_slice.columns and not data_slice['ATR'].empty else price * 0.02
    atr_stop_distance = atr_value * 2.5

    bb_lower = data_slice['BB_Lower'].iloc[-1] if 'BB_Lower' in data_slice.columns and not data_slice['BB_Lower'].empty else price * 0.98
    bb_stop_distance = max(price - bb_lower, price * 0.01)
    
    # Let's use the dynamic calculation to find an *initial* stop distance
    base_stop_distance = max(atr_stop_distance, bb_stop_distance)
    
    confidence_multiplier = 0.9 + (confidence - 0.5) * 0.4 # Slightly adjusted range: 0.9 to 1.1
    adjusted_stop_distance = base_stop_distance * confidence_multiplier

    # --- START OF KEY MODIFICATIONS ---

    # 1. Enforce a STRICT maximum risk limit of 2%
    max_risk_pct = 0.02  # CHANGED: This is your maximum 2% risk
    max_risk_distance = price * max_risk_pct
    
    # The final stop distance is the SMALLER of the dynamic one or your hard 2% limit.
    # This means if volatility is low, your stop might be even tighter than 2%.
    final_stop_distance = min(adjusted_stop_distance, max_risk_distance)

    # Let's also add a small absolute minimum stop to avoid getting stopped out instantly by the spread
    min_stop_distance = price * 0.005 # Minimum 0.5% stop loss
    final_stop_distance = max(final_stop_distance, min_stop_distance)

    # Calculate the final stop-loss price based on the determined distance
    stop_loss = price - final_stop_distance

    stop_loss = price - (price * 0.8)
    
    # 2. Set a FIXED Take-Profit at exactly 3% gain
    take_profit = price * 1.03 # CHANGED: This sets the take profit to entry price + 3%
    
    # 3. Calculate the actual Risk:Reward ratio for accurate reporting
    profit_distance = take_profit - price
    loss_distance = price - stop_loss # This is equal to final_stop_distance
    
    # Avoid division by zero if loss_distance is somehow zero
    actual_risk_reward_ratio = profit_distance / loss_distance if loss_distance > 0 else float('inf')

    # --- END OF MODIFICATIONS ---
    
    return {
        'stop_loss': stop_loss, 
        'take_profit': take_profit, 
        'risk_reward_ratio': actual_risk_reward_ratio
    }


def generate_trading_signals(model, scaler, data, features, min_confidence=0.60):
    """
    Generate dip-buying signals with strict risk management
    """
    signals = data.copy()
    X = signals[features].fillna(signals[features].mean())
    X_scaled = scaler.transform(X)
    signals['Confidence'] = model.predict_proba(X_scaled)[:, 1]
    
    # Only signal on actual dip opportunities with high confidence
    signals['Signal'] = (
        (signals['Confidence'] > min_confidence) & 
        (signals.get('Is_Dip_Opportunity', True))  # Use dip filter if available
    ).astype(int)
    
    signals['Position_Size_Pct'] = signals['Confidence'].apply(
        lambda x: calculate_position_size(x, min_confidence=min_confidence)
    )
    
    # Initialize risk columns
    signals['Stop_Loss'] = np.nan
    signals['Take_Profit'] = np.nan
    signals['Risk_Reward'] = np.nan

    # Calculate risk metrics for each signal
    for i in range(len(signals)):
        if signals['Signal'].iloc[i] == 1:
            current_ticker = signals['Ticker'].iloc[i]
            current_date = signals.index[i]
            
            # Get ticker history
            ticker_history = signals[
                (signals['Ticker'] == current_ticker) & 
                (signals.index <= current_date)
            ].tail(50)  # Use last 50 periods for risk calculation
            
            if not ticker_history.empty:
                risk_metrics = calculate_risk_metrics(
                    ticker_history, 
                    signals['Close'].iloc[i], 
                    signals['Confidence'].iloc[i],
                    min_risk_reward_ratio=2.0  # Enforce 2:1 minimum
                )
                
                # Precise assignment to avoid pandas warnings
                signals.iloc[i, signals.columns.get_loc('Stop_Loss')] = risk_metrics['stop_loss']
                signals.iloc[i, signals.columns.get_loc('Take_Profit')] = risk_metrics['take_profit']
                signals.iloc[i, signals.columns.get_loc('Risk_Reward')] = risk_metrics['risk_reward_ratio']

    return signals

def visualize_signals(signals, ticker):
    """
    Enhanced visualization showing dip-buying signals
    """
    ticker_signals = signals[signals['Ticker'] == ticker].copy()
    if ticker_signals.empty: 
        print(f"No data for {ticker}")
        return
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price chart with signals
    ax1.plot(ticker_signals.index, ticker_signals['Close'], label='Close Price', color='blue', linewidth=1)
    
    # Add moving averages for context
    if 'SMA_20' in ticker_signals.columns:
        ax1.plot(ticker_signals.index, ticker_signals['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
    if 'BB_Lower' in ticker_signals.columns:
        ax1.plot(ticker_signals.index, ticker_signals['BB_Lower'], label='BB Lower', color='red', alpha=0.5, linestyle='--')
    
    # Buy signals
    buy_signals = ticker_signals[ticker_signals['Signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Dip Buy Signal', zorder=5)
    
    # Show stop loss and take profit levels
    for i, row in buy_signals.iterrows():
        if not pd.isna(row['Stop_Loss']) and not pd.isna(row['Take_Profit']):
            ax1.plot([i, i], [row['Close'], row['Stop_Loss']], 'r--', alpha=0.7, linewidth=2)
            ax1.plot([i, i], [row['Close'], row['Take_Profit']], 'g--', alpha=0.7, linewidth=2)
            ax1.scatter(i, row['Stop_Loss'], color='red', marker='_', s=100, zorder=4)
            ax1.scatter(i, row['Take_Profit'], color='green', marker='_', s=100, zorder=4)
    
    ax1.set_title(f'{ticker} - Dip-Buying Strategy Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confidence scores
    ax2.bar(ticker_signals.index, ticker_signals['Confidence'], color='purple', alpha=0.7, label='Confidence Score')
    ax2.axhline(y=0.60, color='r', linestyle='--', label='Min Confidence Threshold')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # RSI for dip context
    if 'RSI' in ticker_signals.columns:
        ax3.plot(ticker_signals.index, ticker_signals['RSI'], color='orange', label='RSI')
        ax3.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Oversold')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Date')
    plt.tight_layout()
    plt.show()

def analyze_performance(signals, forward_period=5):
    """
    Analyze performance with focus on dip-buying metrics
    """
    actual_signals = signals[signals['Signal'] == 1].copy()
    if actual_signals.empty:
        print("\nNo dip-buying trades were generated to analyze.")
        return {}
    
    actual_signals['Actual_Return'] = actual_signals['Future_Return']
    
    # Basic performance metrics
    win_rate = (actual_signals['Actual_Return'] > 0).mean()
    avg_win = actual_signals[actual_signals['Actual_Return'] > 0]['Actual_Return'].mean()
    avg_loss = actual_signals[actual_signals['Actual_Return'] < 0]['Actual_Return'].mean()
    
    # Risk-adjusted metrics
    avg_risk_reward = actual_signals['Risk_Reward'].mean()
    min_risk_reward = actual_signals['Risk_Reward'].min()
    
    # Calculate profit factor
    total_wins = actual_signals[actual_signals['Actual_Return'] > 0]['Actual_Return'].sum()
    total_losses = abs(actual_signals[actual_signals['Actual_Return'] < 0]['Actual_Return'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Dip-specific metrics
    successful_dips = (actual_signals['Actual_Return'] > 0.05).sum()  # 5%+ gains
    total_dips = len(actual_signals)
    big_winner_rate = successful_dips / total_dips if total_dips > 0 else 0
    
    print(f"\n=== DIP-BUYING PERFORMANCE ANALYSIS ===")
    print(f"Analysis Period: {forward_period} days forward")
    print(f"Total Dip-Buy Signals: {len(actual_signals)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Risk:Reward Ratio: {avg_risk_reward:.2f}")
    print(f"Minimum Risk:Reward Ratio: {min_risk_reward:.2f}")
    print(f"Big Winner Rate (>5%): {big_winner_rate:.2%}")
    
    # Visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Return distribution
    ax1.hist(actual_signals['Actual_Return'], bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='r', linestyle='--', label='Break-even')
    ax1.axvline(x=actual_signals['Actual_Return'].mean(), color='g', linestyle='--', label='Average Return')
    ax1.set_title('Distribution of Dip-Buy Returns')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confidence vs Returns
    ax2.scatter(actual_signals['Confidence'], actual_signals['Actual_Return'], alpha=0.6, color='blue')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Model Confidence')
    ax2.set_ylabel('Actual Return')
    ax2.set_title('Confidence vs Actual Returns')
    ax2.grid(True, alpha=0.3)
    
    # Risk-Reward Analysis
    ax3.scatter(actual_signals['Risk_Reward'], actual_signals['Actual_Return'], alpha=0.6, color='green')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.axvline(x=2.0, color='orange', linestyle='--', alpha=0.5, label='Min R:R Target')
    ax3.set_xlabel('Risk:Reward Ratio')
    ax3.set_ylabel('Actual Return')
    ax3.set_title('Risk:Reward vs Actual Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Cumulative returns over time
    actual_signals_sorted = actual_signals.sort_index()
    cumulative_returns = (1 + actual_signals_sorted['Actual_Return']).cumprod()
    ax4.plot(actual_signals_sorted.index, cumulative_returns, color='purple', linewidth=2)
    ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Return Multiple')
    ax4.set_title('Cumulative Performance of Dip-Buying Strategy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'win_rate': win_rate, 
        'avg_win': avg_win, 
        'avg_loss': avg_loss, 
        'profit_factor': profit_factor,
        'avg_risk_reward': avg_risk_reward,
        'min_risk_reward': min_risk_reward,
        'big_winner_rate': big_winner_rate
    }

def main(tickers, train_start_date, train_end_date, test_start_date, test_end_date=None, 
         forward_period=5, min_confidence=0.60, force_download=False):
    """
    Main function to run the enhanced dip-buying pipeline
    """
    if test_end_date is None:
        test_end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=== ENHANCED DIP-BUYING TRADING MODEL ===")
    print("--- Preparing Training Data ---")
    train_data = fetch_and_prepare_data(
        tickers, start_date=train_start_date, end_date=train_end_date, 
        interval='1d', force_download=force_download
    )
    
    print("\n--- Preparing Testing Data ---")
    test_data = fetch_and_prepare_data(
        tickers, start_date=test_start_date, end_date=test_end_date, 
        interval='1d', force_download=force_download
    )
    
    if train_data.empty or test_data.empty:
        print("Could not fetch data for training or testing. Exiting.")
        return None, None, None, None

    print("\n--- Calculating Dip-Specific Features ---")
    train_data = calculate_dip_features(train_data)
    test_data = calculate_dip_features(test_data)
    
    print("\n--- Calculating Technical Features ---")
    train_data = calculate_selected_features(train_data)
    test_data = calculate_selected_features(test_data)
    
    print("\n--- Creating Dip-Buying Targets ---")
    train_data = create_dip_buying_targets(train_data, forward_period)
    test_data = create_dip_buying_targets(test_data, forward_period)
    
    print("\n--- Selecting Dip-Buying Features ---")
    train_data, selected_features = select_dip_buying_features(train_data)
    test_data, _ = select_dip_buying_features(test_data)
    
    print(f"\n--- Training Dip-Buying Model with {len(selected_features)} features ---")
    model, scaler, features, best_threshold = train_dip_buying_model(train_data, selected_features)
    
    print("\n--- Generating Dip-Buying Signals ---")
    signals = generate_trading_signals(model, scaler, test_data, features, best_threshold)
    
    print("\n--- Visualizing Dip-Buying Signals ---")
    for ticker in tickers:
        visualize_signals(signals, ticker)
    
    print("\n--- Analyzing Dip-Buying Performance ---")
    performance = analyze_performance(signals, forward_period)
    
    print("\n--- Process Complete ---")
    return model, scaler, signals, performance

if __name__ == "__main__":
    crypto_tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 
        'AVAX-USD', 'LINK-USD', 'DOGE-USD'
    ]
    
    # --- Configuration ---
    FORCE_API_DOWNLOAD = False  # Keep this False for caching
    
    # --- Define Date Ranges ---
    TRAIN_START = '2022-01-01'
    TRAIN_END = '2024-12-31'
    TEST_START = '2025-01-01'
    TEST_END = '2025-05-18'

    try:
        model, scaler, signals, performance = main(
            crypto_tickers, 
            train_start_date=TRAIN_START,
            train_end_date=TRAIN_END,
            test_start_date=TEST_START,
            test_end_date=TEST_END,
            forward_period=5,
            min_confidence=0.8,  # Higher confidence threshold for dip-buying
            force_download=FORCE_API_DOWNLOAD
        )
        
        # --- Save Enhanced Signals (maintains original format) ---
        if signals is not None and not signals.empty:
            print("\n--- Saving Enhanced Dip-Buying Signals ---")
            SIGNALS_CACHE_DIR = 'signals_cache'
            if not os.path.exists(SIGNALS_CACHE_DIR):
                os.makedirs(SIGNALS_CACHE_DIR)
            
            # Create filename consistent with original format
            signals_filename = os.path.join(SIGNALS_CACHE_DIR, f"signals_{TEST_START}_to_{TEST_END}.pkl")
            
            for filename in os.listdir(SIGNALS_CACHE_DIR):
                file_path = os.path.join(SIGNALS_CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)

            print(f"Saving enhanced dip-buying signals to: {signals_filename}")
            signals.to_pickle(signals_filename)
            print("Enhanced signals saved successfully.")
            
            # Print summary of generated signals
            total_signals = (signals['Signal'] == 1).sum()
            avg_confidence = signals[signals['Signal'] == 1]['Confidence'].mean() if total_signals > 0 else 0
            avg_risk_reward = signals[signals['Signal'] == 1]['Risk_Reward'].mean() if total_signals > 0 else 0
            
            print(f"\n=== FINAL SIGNAL SUMMARY ===")
            print(f"Total Dip-Buying Signals Generated: {total_signals}")
            print(f"Average Signal Confidence: {avg_confidence:.3f}")
            print(f"Average Risk:Reward Ratio: {avg_risk_reward:.2f}")
            print(f"Minimum Risk:Reward Ratio Enforced: 2.0")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()