import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

# For reproducible results
tf.random.set_seed(42)
np.random.seed(42)

import os
import pickle
import warnings
from datetime import datetime, timedelta

# Suppress specific future warnings from pandas or other libraries
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Caching Configuration ---
CACHE_DIR = 'data_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- Helper Function for Display Precision (not directly used in model logic) ---
def get_display_precision(price_series):
    """
    Determines the appropriate decimal precision for display based on price magnitude.
    Handles pandas Series to find minimum non-zero price for robust precision.
    """
    if price_series.empty:
        return 2 # Default for empty series

    non_zero_prices = price_series[price_series.notna() & (price_series != 0)]
    if non_zero_prices.empty:
        return 2 # Default if all are zero or NaN

    min_price = non_zero_prices.min()

    if min_price >= 100:
        return 2
    elif min_price >= 10:
        return 3
    elif min_price >= 1:
        return 4
    elif min_price >= 0.1:
        return 5
    elif min_price >= 0.01:
        return 6
    else:
        return int(np.ceil(-np.log10(min_price))) + 2

# Function to fetch and prepare data with caching
def fetch_and_prepare_data(tickers, start_date, end_date, interval='1d', force_download=False):
    """
    Fetch historical data for cryptocurrencies/stocks and calculate basic features.
    Includes caching to avoid re-downloading data.
    """
    start_str = start_date.replace('-', '')
    end_str = end_date.replace('-', '')
    cache_filename = os.path.join(CACHE_DIR, f"{'_'.join(sorted(tickers))}_{start_str}_to_{end_str}_{interval}.pkl")

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
        data = crypto.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            print(f"    No data found for {ticker} in the specified period/interval. Skipping.")
            continue

        data['Ticker'] = ticker
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1)).replace([np.inf, -np.inf], np.nan)
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Price_Volume_Ratio'] = data['Close'] / (data['Volume'].replace(0, np.nan) + 1e-9)

        all_data = pd.concat([all_data, data], axis=0)

    # Only sort if all_data is not empty, otherwise Key Error occurs
    if not all_data.empty:
        all_data = all_data.sort_index().sort_values(by='Ticker')
        all_data = all_data.dropna(subset=['Log_Return', 'Daily_Return', 'Volume_Change', 'Price_Volume_Ratio'])
        print(f"Saving data to cache: {cache_filename}")
        with open(cache_filename, 'wb') as f:
            pickle.dump(all_data, f)
    else:
        print(f"No data to save for {', '.join(tickers)} after initial processing.")

    return all_data

# Simplified calculate_selected_features for a basic LSTM
def calculate_selected_features(df):
    """
    Calculates a basic set of technical indicators for the LSTM model.
    """
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], _, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9) # Only MACD line
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Normalized_ATR'] = df['ATR'] / (df['Close'] + 1e-9) # Important for target creation

    # Clean up NaNs created by feature calculations
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.fillna(0) # Catch any remaining NaNs

    return df

# Create prediction targets
def create_prediction_targets(data, forward_period, volatility_feature_name, profit_pct, stop_loss_pct):
    """
    Create prediction targets based on the 'first touch' of a dynamically calculated profit or stop-loss threshold
    within a specified forward period. Also stores the calculated profit and stop-loss prices.
    """
    print(f"Creating prediction targets based on dynamic volatility-adjusted thresholds within {forward_period} intervals...")
    data['Target'] = np.nan
    data['Future_Return'] = np.nan
    data['Return_Magnitude'] = np.nan
    data['Calculated_Take_Profit'] = np.nan # New column to store calculated TP
    data['Calculated_Stop_Loss'] = np.nan  # New column to store calculated SL

    # Process each ticker group separately
    grouped = data.groupby('Ticker')
    processed_data_list = []

    for ticker, group in grouped:
        temp_group = group.copy()
        
        for i in range(len(temp_group) - forward_period):
            current_close = temp_group['Close'].iloc[i]
            current_index = temp_group.index[i]
            current_volatility = temp_group[volatility_feature_name].iloc[i]

            # Handle cases where volatility is NaN or zero
            if pd.isna(current_volatility) or current_volatility <= 1e-9:
                temp_group.loc[current_index, 'Target'] = 0
                temp_group.loc[current_index, 'Future_Return'] = 0.0
                temp_group.loc[current_index, 'Return_Magnitude'] = 0.0
                temp_group.loc[current_index, 'Calculated_Take_Profit'] = current_close # Default
                temp_group.loc[current_index, 'Calculated_Stop_Loss'] = current_close # Default
                continue


            profit_price = current_close * (1 + profit_pct)
            stop_loss_price = current_close * (1 - stop_loss_pct)

            # Store the calculated levels for this point
            temp_group.loc[current_index, 'Calculated_Take_Profit'] = profit_price
            temp_group.loc[current_index, 'Calculated_Stop_Loss'] = stop_loss_price

            future_prices = temp_group['Close'].iloc[i+1 : i+1+forward_period]

            if future_prices.empty:
                continue

            target_set = False
            for j, future_price in enumerate(future_prices):
                if future_price >= profit_price:
                    temp_group.loc[current_index, 'Target'] = 1
                    temp_group.loc[current_index, 'Future_Return'] = (future_price - current_close) / current_close
                    target_set = True
                    break
                if future_price <= stop_loss_price:
                    temp_group.loc[current_index, 'Target'] = 0
                    temp_group.loc[current_index, 'Future_Return'] = (future_price - current_close) / current_close
                    target_set = True
                    break

            if not target_set:
                # If neither profit nor stop loss reached, assign a default target (e.g., 0 for no profit)
                # and the return at the end of the period
                temp_group.loc[current_index, 'Target'] = 0
                temp_group.loc[current_index, 'Future_Return'] = (future_prices.iloc[-1] - current_close) / current_close
            
            temp_group.loc[current_index, 'Return_Magnitude'] = abs(temp_group.loc[current_index, 'Future_Return'])
        
        processed_data_list.append(temp_group)
            
    data = pd.concat(processed_data_list).sort_index()
    # Drop any rows where target could not be determined (e.g., at the end of the series or due to NaNs)
    data = data.dropna(subset=['Target', 'Calculated_Take_Profit', 'Calculated_Stop_Loss'])
    data['Target'] = data['Target'].astype(int) # Ensure target is integer

    print("Target creation complete.")
    return data

# Function to create sequences for LSTM
def create_sequences(features_df, targets_series, lookback_window):
    """
    Creates sequences of data for LSTM input and returns corresponding indices.
    """
    X, y, indices = [], [], []
    # Align features and targets by index and drop NaNs introduced by shifts/alignments
    aligned_data = pd.concat([features_df, targets_series], axis=1).dropna()
    features = aligned_data[features_df.columns].values
    targets = aligned_data[targets_series.name].values
    # Get the index (DateTime) of the aligned data
    data_indices = aligned_data.index

    for i in range(lookback_window, len(features)):
        X.append(features[i-lookback_window : i])
        y.append(targets[i])
        indices.append(data_indices[i]) # Store the index of the predicted point
    return np.array(X), np.array(y), np.array(indices)


# Main execution block
if __name__ == '__main__':
    # --- Configuration ---
    # Tickers and Date Range (example: Bitcoin for the last year, hourly data)
    tickers = ['BTC-USD', 'ETH-USD'] # Added ETH-USD for multi-ticker testing
    # Adjust date range to be within the last 730 days for 1h interval data from Yahoo Finance
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d') # Roughly 1 year ago
    interval = '15m'

    # LSTM Specific Parameters
    LOOKBACK_WINDOW = 60 # Number of past intervals to consider for each prediction
    FORWARD_PERIOD = 8   # Number of future intervals to look for profit/stop loss
    PROFIT_ATR_MULTIPLE = 1.0 # 1.0 * ATR for profit target
    STOP_LOSS_ATR_MULTIPLE = 1.5 # 1.5 * ATR for stop loss

    # Model Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LSTM_UNITS = 50
    DROPOUT_RATE = 0.2

    print("--- STAGE 1: Data Fetching and Feature Engineering ---")
    data = fetch_and_prepare_data(tickers, start_date, end_date, interval, force_download=False)
    
    if data.empty:
        print("No data to process. Exiting.")
        exit()
    else:
        print(f"Raw data fetched: {data.shape[0]} rows across {len(data['Ticker'].unique())} tickers.")
        
        # Calculate a simplified set of features
        data = data.groupby('Ticker', group_keys=False).apply(calculate_selected_features)
        print(f"Features calculated. Data shape: {data.shape}")

        # Create prediction targets (now also stores Calculated_Take_Profit and Calculated_Stop_Loss)
        data = data.groupby('Ticker', group_keys=False).apply(
            lambda x: create_prediction_targets(x, FORWARD_PERIOD, volatility_feature_name='Normalized_ATR',
                                                profit_pct=0.03,
                                                stop_loss_pct=0.02)
        )
        
        # Drop rows with NaN targets (from create_prediction_targets or initial feature calculation)
        data.dropna(subset=['Target', 'Calculated_Take_Profit', 'Calculated_Stop_Loss'], inplace=True)
        if data.empty:
            print("No data with valid targets after processing. Exiting.")
            exit()
        print(f"Targets created. Data shape after dropping NaNs: {data.shape}")

        # Define feature columns after all processing
        feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Target', 'Future_Return', 'Return_Magnitude', 'Calculated_Take_Profit', 'Calculated_Stop_Loss']]
        
        if not feature_columns:
            print("No suitable features found after initial processing. Exiting.")
            exit()

        print(f"Selected features for LSTM: {len(feature_columns)}")

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[feature_columns])
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_columns, index=data.index)
        data[feature_columns] = scaled_features_df

        # Prepare data for LSTM: Create sequences
        X_all, y_all, indices_all = [], [], [] # Add indices_all
        # Group by ticker to ensure sequences don't cross ticker boundaries
        for ticker_sym in data['Ticker'].unique(): # Use ticker_sym to avoid conflict with `tickers` list
            ticker_data = data[data['Ticker'] == ticker_sym]
            # Ensure enough data points for lookback window
            if len(ticker_data) >= LOOKBACK_WINDOW:
                X_ticker, y_ticker, indices_ticker = create_sequences(ticker_data[feature_columns], ticker_data['Target'], LOOKBACK_WINDOW)
                if X_ticker.shape[0] > 0: # Ensure sequences were actually created
                    X_all.append(X_ticker)
                    y_all.append(y_ticker)
                    indices_all.append(indices_ticker) # Append indices
        
        if not X_all:
            print("Not enough data to create LSTM sequences. Exiting.")
            exit()
        
        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        all_indices = np.concatenate(indices_all) # Combine all indices

        print(f"LSTM sequences created. X shape: {X.shape}, y shape: {y.shape}, Indices shape: {all_indices.shape}")

        # Split data into training and testing sets using TimeSeriesSplit
        # For simplicity, we'll use the last fold as test set from TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        # Get the indices for the last split
        train_index, test_index = list(tscv.split(X))[-1] 

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        test_dates = all_indices[test_index] # Get the dates corresponding to the test set

        # --- IMPORTANT FIX: Ensure original_close_prices_and_tickers_test is perfectly aligned ---
        # Get all relevant original data for ALL sequences first
        # This DataFrame will be aligned by index with X, y, all_indices
        original_info_all_sequences = data.loc[all_indices, [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Future_Return',
            'Calculated_Take_Profit', 'Calculated_Stop_Loss' # Include these columns for signals
        ]].copy()
        
        # Then, slice it using test_index to get the specific test set information
        # This guarantees alignment with X_test, y_test, and test_dates
        original_close_prices_and_tickers_test = original_info_all_sequences.iloc[test_index].copy()


        print(f"Data split into training/testing. X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        # --- STAGE 2: LSTM Model Definition and Training ---
        print("\n--- STAGE 2: LSTM Model Definition and Training ---")

        model = Sequential([
            LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(DROPOUT_RATE),
            LSTM(units=LSTM_UNITS),
            Dropout(DROPOUT_RATE),
            Dense(units=1, activation='sigmoid') # For binary classification (0 or 1 target)
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
        model.summary()

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        print("Training LSTM model...")
        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_split=0.2, # Use a portion of training data for validation
                            callbacks=[early_stopping, reduce_lr],
                            verbose=1)

        print("\n--- STAGE 3: Model Evaluation ---")
        loss, accuracy, auc_score = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc_score:.4f}")

        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # --- STAGE 4: Plotting Signals (Ticker-specific plots) ---
        print("\n--- STAGE 4: Plotting Signals (Ticker-specific plots) ---")
        
        unique_test_tickers = original_close_prices_and_tickers_test['Ticker'].unique()

        for current_ticker in unique_test_tickers:
            print(f"\nGenerating plots for ticker: {current_ticker}")
            
            # Create a boolean mask directly on the aligned original_close_prices_and_tickers_test
            ticker_mask = (original_close_prices_and_tickers_test['Ticker'] == current_ticker).values 

            # Filter all relevant arrays using this unified numpy boolean mask
            # Make sure test_dates_ticker is defined correctly first.
            # It should already be correctly sliced above, but let's be explicit:
            test_dates_ticker = test_dates[ticker_mask] # This is crucial: get the correct dates for the ticker

            # Now, construct the plot_df_ticker directly using the filtered Close prices and the correct dates
            # We will use the original 'Close' price from the aligned data.
            close_prices_ticker = original_close_prices_and_tickers_test.loc[ticker_mask, 'Close'].values
            
            plot_df_ticker = pd.DataFrame({
                'Close': close_prices_ticker
            }, index=pd.to_datetime(test_dates_ticker)) # Use the correct test_dates_ticker here

            y_test_ticker = y_test[ticker_mask]
            y_pred_ticker = y_pred[ticker_mask]

            # Plot 1: Close Price with Actual Targets and Predicted Buy Signals
            plt.figure(figsize=(16, 8))
            plt.plot(plot_df_ticker.index, plot_df_ticker['Close'], label='Close Price', color='blue', alpha=0.8)

            # Ensure y_test_series_aligned and y_pred_series_aligned have the *exact same index* as plot_df_ticker
            y_test_series_aligned = pd.Series(y_test_ticker, index=plot_df_ticker.index)
            y_pred_series_aligned = pd.Series(y_pred_ticker.flatten(), index=plot_df_ticker.index)

            # Filter plot_df_ticker based on actual and predicted signals
            actual_buy_signals_df = plot_df_ticker[y_test_series_aligned == 1]
            predicted_buy_signals_df = plot_df_ticker[y_pred_series_aligned == 1]

            # Plot actual buy targets
            plt.scatter(actual_buy_signals_df.index, actual_buy_signals_df['Close'],
                        marker='o', color='gold', s=100, label='Actual Buy Target (1)', zorder=4, edgecolor='black')

            # Plot predicted buy signals
            plt.scatter(predicted_buy_signals_df.index, predicted_buy_signals_df['Close'],
                        marker='^', color='green', s=120, label='Predicted Buy Signal (1)', zorder=5)

            plt.title(f'{current_ticker} Close Price with Actual Targets and Predicted Buy Signals (Test Set)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot 2: Actual vs. Predicted Targets Over Time
            # This plot should now also look correct as test_dates_ticker is correctly aligned.
            plt.figure(figsize=(16, 6))
            plt.plot(pd.to_datetime(test_dates_ticker), y_test_ticker, label='Actual Target', color='darkorange', marker='.', linestyle='None', alpha=0.6)
            plt.plot(pd.to_datetime(test_dates_ticker), y_pred_ticker, label='Predicted Target', color='purple', marker='x', linestyle='None', alpha=0.6)
            
            plt.title(f'{current_ticker} Actual vs. Predicted Targets (Test Set)')
            plt.xlabel('Date')
            plt.ylabel('Target Value (0 or 1)')
            plt.yticks([0, 1])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            
            # Plot 2: Actual vs. Predicted Targets Over Time
            # For this plot, using test_dates_ticker for x-axis is fine as y_test_ticker and y_pred_ticker are aligned with it.
            plt.figure(figsize=(16, 6))
            plt.plot(pd.to_datetime(test_dates_ticker), y_test_ticker, label='Actual Target', color='darkorange', marker='.', linestyle='None', alpha=0.6)
            plt.plot(pd.to_datetime(test_dates_ticker), y_pred_ticker, label='Predicted Target', color='purple', marker='x', linestyle='None', alpha=0.6)
            
            plt.title(f'{current_ticker} Actual vs. Predicted Targets (Test Set)')
            plt.xlabel('Date')
            plt.ylabel('Target Value (0 or 1)')
            plt.yticks([0, 1])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # --- STAGE 5: Overall Analysis and Plotting (Aggregated) ---
        print("\n--- STAGE 5: Overall Analysis and Plotting (Aggregated) ---")

        # 1. Trading Strategy Simulation and Performance
        # Create a DataFrame for test set results
        test_results = pd.DataFrame({
            'Close': original_close_prices_and_tickers_test['Close'].values,
            'Ticker': original_close_prices_and_tickers_test['Ticker'].values,
            'Actual_Target': y_test,
            'Predicted_Target': y_pred.flatten(),
            'Future_Return': original_close_prices_and_tickers_test['Future_Return'].values
        }, index=pd.to_datetime(test_dates))

        # Initialize cooldown tracking
        last_trade_time = {} # To store the last entry time for each ticker

        # Simulate trades with cooldown:
        test_results['Trade_Return_With_Cooldown'] = 0.0
        test_results['Trade_Initiated'] = False # New column to mark if a trade was initiated

        COOLDOWN_PERIOD = 5
        for i, row in test_results.iterrows():
            ticker = row['Ticker']
            current_time = i # The index is already a datetime object

            # Check if predicted target is 1 (buy signal)
            if row['Predicted_Target'] == 1:
                # Check cooldown for this ticker
                if ticker not in last_trade_time or (current_time - last_trade_time[ticker]).total_seconds() >= COOLDOWN_PERIOD * 60 * 60: # Cooldown in seconds, convert intervals to seconds. If interval is '15m', then 20*15*60 seconds.
                    # If not in cooldown, initiate trade
                    test_results.loc[i, 'Trade_Return_With_Cooldown'] = row['Future_Return']
                    test_results.loc[i, 'Trade_Initiated'] = True
                    last_trade_time[ticker] = current_time # Update last trade time for this ticker
                # else: Trade signal ignored due to cooldown
            # else: No predicted buy signal, no trade

        num_trades_cooldown = (test_results['Trade_Initiated'] == True).sum()
        total_trade_return_cooldown = test_results['Trade_Return_With_Cooldown'].sum()
        average_trade_return_cooldown = total_trade_return_cooldown / num_trades_cooldown if num_trades_cooldown > 0 else 0

        print(f"\nTrading Simulation Results (Overall Test Set) with Cooldown:")
        print(f"Total Predicted Trades (with Cooldown): {num_trades_cooldown}")
        print(f"Total Return from Predicted Trades (with Cooldown): {total_trade_return_cooldown:.4f}")
        print(f"Average Return per Predicted Trade (with Cooldown): {average_trade_return_cooldown:.4f}")

        # Calculate Cumulative Returns (Equity Curve) with Cooldown
        if num_trades_cooldown > 0:
            test_results['Cumulative_Return_Cooldown'] = (1 + test_results['Trade_Return_With_Cooldown']).cumprod() - 1
        else:
            test_results['Cumulative_Return_Cooldown'] = 0 # No trades, no return

        # Plot Equity Curve with Cooldown
        plt.figure(figsize=(16, 8))
        plt.plot(test_results.index, test_results['Cumulative_Return_Cooldown'], label='Cumulative Return (Simulated Trades with Cooldown)', color='darkorange')
        plt.title(f'Overall Simulated Trading Strategy Cumulative Returns (Test Set) with Cooldown')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2. Confusion Matrix Plot (Overall)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0 (No Buy)', 'Predicted 1 (Buy)'],
                    yticklabels=['Actual 0 (No Buy)', 'Actual 1 (Buy)'])
        plt.title(f'Overall Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        # 3. Feature Correlation Heatmap (Overall)
        # Calculate correlation matrix on the scaled features from the entire dataset
        feature_corr = scaled_features_df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(feature_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=".5")
        plt.title(f'Overall Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

        # --- STAGE 6: Saving Signals for Backtesting ---
        print("\n--- STAGE 6: Saving Signals for Backtesting ---")

        # Construct the signals DataFrame for backtesting
        # Get all relevant original data for ALL sequences first
        # This DataFrame will be aligned by index with X, y, all_indices
        test_data_for_signals = original_info_all_sequences.iloc[test_index].copy() # CORRECTED LINE

        MIN_CONFIDENCE_FOR_SIGNAL=0.60
        # Add the predicted signal and confidence
        test_data_for_signals['Signal'] = y_pred.flatten()
        test_data_for_signals['Confidence'] = y_pred_proba.flatten()

        # Dynamic Position Sizing based on Confidence (from daytrade.py)
        # Apply a sigmoid-like function to scale position size based on confidence
        # This makes position size increase more sharply above MIN_CONFIDENCE_FOR_SIGNAL
        k = 10 # Controls the steepness of the sigmoid
        x0 = MIN_CONFIDENCE_FOR_SIGNAL # Midpoint of the sigmoid (confidence at which size is 50% of max)

        # Ensure confidence is a Series for element-wise operation if not already
        confidence_series = test_data_for_signals['Confidence']
        
        # Calculate position_size_pct
        test_data_for_signals['Position_Size_Pct'] = 1 / (1 + np.exp(-k * (confidence_series - x0)))

        # Select columns required by the backtester, including the newly added calculated levels
        signals_to_save_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker',
            'Signal', 'Confidence', 'Calculated_Take_Profit', 'Calculated_Stop_Loss', 'Position_Size_Pct' # Add Position_Size_Pct here
        ]
        
        # Rename the calculated columns to match backtester's expectation
        final_signals_df = test_data_for_signals[signals_to_save_columns].rename(columns={
            'Calculated_Take_Profit': 'Take_Profit',
            'Calculated_Stop_Loss': 'Stop_Loss'
        })
        
        # Ensure the index is named correctly and is datetime
        final_signals_df.index.name = 'DateTime'
        
        # Define signal cache directory
        SIGNALS_CACHE_DIR = 'signals_cache'
        if not os.path.exists(SIGNALS_CACHE_DIR):
            os.makedirs(SIGNALS_CACHE_DIR)

        # Define signal filename (matching backtester's expected format)
        test_start_date_str = test_dates.min().strftime('%Y-%m-%d')
        test_end_date_str = test_dates.max().strftime('%Y-%m-%d')
        signals_filename = os.path.join(SIGNALS_CACHE_DIR, f"signals_{test_start_date_str}_to_{test_end_date_str}.pkl")

        print(f"Saving signals for backtesting to: {signals_filename}")
        final_signals_df.to_pickle(signals_filename)

        print("\nModel training, evaluation, and plotting complete. Signals have been saved for backtesting.")