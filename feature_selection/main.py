import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import yfinance as yf
import talib
from datetime import datetime, timedelta
import statsmodels.api as sm
import requests
from pytrends.request import TrendReq
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def fetch_crypto_data(tickers, period='1y', interval='1d'):
    """
    Fetch historical data for cryptocurrencies
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
        data['Price_Volume_Ratio'] = data['Close'] / (data['Volume'] + 1)  # Adding 1 to avoid division by zero
        
        # Add to main dataframe
        all_data = pd.concat([all_data, data], axis=0)
    
    # Drop NaNs after calculating basic features
    all_data = all_data.dropna()
    
    return all_data

def enrich_with_technical_indicators(data):
    """
    Add technical indicators using TALib
    """
    print("Calculating technical indicators...")
    grouped = data.groupby('Ticker')
    result = pd.DataFrame()
    
    for ticker, group in grouped:
        df = group.copy()
        
        # Price action indicators
        df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        
        # Moving average ratios and crosses
        df['SMA_5_20_Ratio'] = df['SMA_5'] / df['SMA_20']
        df['SMA_20_50_Ratio'] = df['SMA_20'] / df['SMA_50']
        df['SMA_50_200_Ratio'] = df['SMA_50'] / df['SMA_200']
        df['Price_SMA_20_Ratio'] = df['Close'] / df['SMA_20']
        
        # Exponential moving averages
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # Momentum indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_EMA'] = talib.EMA(df['RSI'], timeperiod=9)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Volatility indicators
        df['Volatility_5'] = df['Log_Return'].rolling(window=5).std()
        df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Normalized_ATR'] = df['ATR'] / df['Close']
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'], timeperiod=20)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['Percent_B'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['OBV_EMA'] = talib.EMA(df['OBV'], timeperiod=20)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        
        # Trend indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Plus_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Minus_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Aroon_Up'], df['Aroon_Down'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
        df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']
        df['Parabolic_SAR'] = talib.SAR(df['High'], df['Low'])
        df['SAR_Ratio'] = df['Close'] / df['Parabolic_SAR']
        
        # Price patterns and oscillators
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['Stoch_RSI_K'], df['Stoch_RSI_D'] = talib.STOCHRSI(df['Close'], timeperiod=14)
        df['Ultimate_Oscillator'] = talib.ULTOSC(df['High'], df['Low'], df['Close'])
        df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
        
        # Candle pattern recognition (binary indicators)
        df['Doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['Engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        df['Hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['Morning_Star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Market cycle indicators
        df['Ichimoku_Conversion'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Ichimoku_Base'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Ichimoku_A'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        df['Ichimoku_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        
        # Custom indicators
        df['High_Low_Range'] = df['High'] - df['Low']
        df['HL_Range_Ratio'] = df['High_Low_Range'] / df['Close']
        df['Gap_Up'] = (df['Low'] > df['High'].shift(1)).astype(int)
        df['Gap_Down'] = (df['High'] < df['Low'].shift(1)).astype(int)
        df['Price_Change_Rate'] = (df['Close'] - df['Open']) / df['Open']
        
        # Statistical indicators
        df['Z_Score_20'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
        df['Skewness_20'] = df['Log_Return'].rolling(window=20).apply(lambda x: stats.skew(x))
        df['Kurtosis_20'] = df['Log_Return'].rolling(window=20).apply(lambda x: stats.kurtosis(x))
        
        result = pd.concat([result, df], axis=0)
    
    return result

def add_market_indicators(data):
    """
    Add market-wide indicators and calendar features
    """
    print("Adding market indicators and calendar features...")
    
    # Calendar features
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter
    data['Year'] = data.index.year
    data['Day_of_Month'] = data.index.day
    data['Week_of_Year'] = data.index.isocalendar().week
    data['Is_Month_End'] = data.index.is_month_end.astype(int)
    data['Is_Month_Start'] = data.index.is_month_start.astype(int)
    data['Is_Quarter_End'] = data.index.is_quarter_end.astype(int)
    data['Is_Quarter_Start'] = data.index.is_quarter_start.astype(int)
    data['Is_Year_End'] = data.index.is_year_end.astype(int)
    data['Is_Year_Start'] = data.index.is_year_start.astype(int)
    
    # Bitcoin dominance (proxy for market direction)
    btc_data = data[data['Ticker'] == 'BTC-USD'].copy()
    if not btc_data.empty:
        btc_dates = btc_data.index
        total_market = data[data.index.isin(btc_dates)].groupby(data.index)['Close'].sum()
        btc_close = btc_data['Close']
        btc_dominance = pd.Series(index=btc_dates)
        for date in btc_dates:
            if date in total_market.index:
                btc_dominance[date] = btc_close[date] / total_market[date] if total_market[date] > 0 else np.nan
        
        # Merge back with original data
        btc_dominance_df = pd.DataFrame(btc_dominance, columns=['BTC_Dominance'])
        data = pd.merge(data, btc_dominance_df, left_index=True, right_index=True, how='left')
        
        # Propagate BTC dominance value to all coins on the same date
        data['BTC_Dominance'] = data.groupby(data.index)['BTC_Dominance'].transform('first')
    
    return data

def add_relative_strength_indicators(data):
    """
    Add relative strength indicators between coins
    """
    print("Adding relative strength indicators...")
    tickers = data['Ticker'].unique()
    dates = data.index.unique()
    
    # Create a MultiIndex DataFrame to store relative performance
    relative_strength = pd.DataFrame(index=pd.MultiIndex.from_product([dates, tickers], names=['Date', 'Ticker']))
    
    # Calculate rolling returns for each coin
    for ticker in tickers:
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data['Rolling_Return_5d'] = ticker_data['Close'].pct_change(5)
        ticker_data['Rolling_Return_20d'] = ticker_data['Close'].pct_change(20)
        
        for date in ticker_data.index:
            if date in relative_strength.index.get_level_values('Date'):
                relative_strength.loc[(date, ticker), 'Rolling_Return_5d'] = ticker_data.loc[date, 'Rolling_Return_5d']
                relative_strength.loc[(date, ticker), 'Rolling_Return_20d'] = ticker_data.loc[date, 'Rolling_Return_20d']
    
    # Calculate relative strength against the group
    for date in dates:
        date_data = relative_strength.loc[date]
        avg_5d = date_data['Rolling_Return_5d'].mean()
        avg_20d = date_data['Rolling_Return_20d'].mean()
        
        for ticker in tickers:
            if (date, ticker) in relative_strength.index:
                relative_strength.loc[(date, ticker), 'RS_5d'] = relative_strength.loc[(date, ticker), 'Rolling_Return_5d'] - avg_5d
                relative_strength.loc[(date, ticker), 'RS_20d'] = relative_strength.loc[(date, ticker), 'Rolling_Return_20d'] - avg_20d
    
    # Merge relative strength metrics back to main dataframe
    relative_strength = relative_strength.reset_index()
    data = data.reset_index()
    data = pd.merge(data, relative_strength[['Date', 'Ticker', 'RS_5d', 'RS_20d']], 
                    on=['Date', 'Ticker'], how='left')
    data = data.set_index('Date')
    
    return data

def fetch_market_sentiment():
    """
    Attempt to fetch market sentiment data from Fear & Greed Index
    (This is a simplified version and may need API credentials in real use)
    """
    try:
        # This is a placeholder - in real implementation, you would use a proper API
        response = requests.get('https://api.alternative.me/fng/?limit=365')
        if response.status_code == 200:
            sentiment_data = response.json()
            # Process the response into a DataFrame
            sentiment_df = pd.DataFrame(sentiment_data['data'])
            # Convert to timezone-naive datetime to match main dataframe
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], unit='s').dt.tz_localize(None)
            sentiment_df = sentiment_df.set_index('timestamp')
            sentiment_df['value'] = sentiment_df['value'].astype(float)
            return sentiment_df
        else:
            print("Could not fetch sentiment data, using random values")
            return None
    except:
        print("Error in fetching sentiment data, using random values")
        return None

def add_google_trends_data(data, keywords):
    """
    Add Google Trends data for specified keywords
    """
    try:
        print("Fetching Google Trends data...")
        pytrends = TrendReq(hl='en-US', tz=360)
        trend_data = pd.DataFrame()
        
        # Get date range from data
        start_date = data.index.min().strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')
        
        for keyword in keywords:
            pytrends.build_payload([keyword], cat=0, timeframe=f'{start_date} {end_date}', geo='', gprop='')
            trend = pytrends.interest_over_time()
            if not trend.empty:
                trend_data[f'Trend_{keyword}'] = trend[keyword]
        
        if not trend_data.empty:
            # Reindex to match the original data
            trend_data = trend_data.reindex(data.index, method='ffill')
            
            # Merge with main data
            for col in trend_data.columns:
                data[col] = trend_data[col].reindex(data.index)
        
        return data
    except Exception as e:
        print(f"Error fetching Google Trends data: {e}")
        return data

def add_synthetic_features(data):
    """
    Add more complex synthetic features and interactions
    """
    print("Generating synthetic features...")
    
    # Price momentum and volatility interactions
    data['RSI_Volatility'] = data['RSI'] * data['Volatility_20']
    data['MOM_Volatility'] = data['MOM'] * data['Volatility_20']
    
    # Volume-weighted indicators
    data['Volume_RSI'] = data['RSI'] * (data['Volume'] / data['Volume'].rolling(window=20).mean())
    data['Volume_Weighted_Return'] = data['Daily_Return'] * (data['Volume'] / data['Volume'].rolling(window=20).mean())
    
    # Trend strength indicators
    data['Trend_Strength'] = data['ADX'] * np.where(data['Plus_DI'] > data['Minus_DI'], 1, -1)
    data['RSI_Trend'] = np.where(data['RSI'] > 50, 1, -1) * data['ADX']
    
    # Oscillator convergence/divergence
    data['MACD_RSI_Convergence'] = np.sign(data['MACD']) * np.sign(data['RSI'] - 50)
    
    # Support/resistance proximity
    data['Support_Proximity'] = (data['Close'] - data['BB_Lower']) / data['Close']
    data['Resistance_Proximity'] = (data['BB_Upper'] - data['Close']) / data['Close']
    
    # Cross indicators as binary signals
    data['SMA_5_20_Cross'] = np.where(data['SMA_5'] > data['SMA_20'], 1, -1)
    data['Golden_Cross'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1)
    data['MACD_Signal_Cross'] = np.where(data['MACD'] > data['MACD_Signal'], 1, -1)
    
    # Volatility regime
    data['Volatility_Regime'] = np.where(data['Volatility_20'] > data['Volatility_20'].rolling(window=50).mean(), 1, 0)
    
    # Complex indicators
    data['Buying_Pressure'] = ((data['Close'] - data['Low']) / (data['High'] - data['Low'])) * data['Volume']
    data['Selling_Pressure'] = ((data['High'] - data['Close']) / (data['High'] - data['Low'])) * data['Volume']
    
    # Mean reversion potential
    data['Mean_Reversion_Potential'] = data['Z_Score_20'] * data['RSI']
    
    # Probability of trend continuation
    data['Trend_Continuation'] = data['ADX'] * data['MOM'] * (1 if data['MOM'].iloc[-1] > 0 else -1)
    
    # Overbought/oversold composite
    data['Overbought_Composite'] = (
        (data['RSI'] > 70).astype(int) + 
        (data['Stoch_K'] > 80).astype(int) + 
        (data['MFI'] > 80).astype(int)
    )
    data['Oversold_Composite'] = (
        (data['RSI'] < 30).astype(int) + 
        (data['Stoch_K'] < 20).astype(int) + 
        (data['MFI'] < 20).astype(int)
    )
    
    return data

def prepare_for_clustering(data):
    """
    Prepare data for clustering by handling missing values and outliers
    """
    print("Preparing data for clustering...")
    
    # Get all numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove any unwanted columns for clustering
    exclude_columns = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in numeric_cols if col not in exclude_columns]
    
    # Create a copy of data with only numeric features
    X = data[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Handle outliers - cap values at 3 standard deviations
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        X[col] = X[col].clip(lower=mean - 3*std, upper=mean + 3*std)
    
    return X, feature_cols

def find_optimal_clusters(data, max_k=10):
    """
    Find optimal number of clusters using silhouette score
    """
    silhouette_scores = []
    k_values = range(2, max_k+1)
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        silhouette_avg = silhouette_score(data_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.4f}")
    
    # Find the best k
    optimal_k = k_values[np.argmax(silhouette_scores)]
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method For Optimal k')
    plt.axvline(x=optimal_k, color='r', linestyle='--')
    plt.grid(True)
    plt.show()
    
    return optimal_k

def analyze_feature_importance(data, feature_cols, n_clusters):
    """
    Analyze feature importance after clustering
    """
    # Get data for chosen features
    X = data[feature_cols].copy()
    
    # Handle missing values more aggressively
    X = X.fillna(X.mean())
    
    # Also handle any infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Double-check for any remaining NaNs
    if X.isna().any().any():
        # If there are still NaNs, drop those columns
        problematic_cols = X.columns[X.isna().any()].tolist()
        print(f"Warning: Dropping columns with persistent NaN values: {problematic_cols}")
        X = X.drop(columns=problematic_cols)
        feature_cols = [col for col in feature_cols if col not in problematic_cols]
    
    # Scale the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Final sanity check for NaNs in scaled data
    if np.isnan(data_scaled).any():
        print("Warning: NaNs detected after scaling. Applying additional cleaning...")
        data_scaled = np.nan_to_num(data_scaled, nan=0.0)
    
    clusters = kmeans.fit_predict(data_scaled)
    data['Cluster'] = clusters
    
    # Calculate feature importance by comparing cluster centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                  columns=X.columns)
    
    # Calculate feature importance based on cluster separation
    importance_scores = []
    
    for feature in X.columns:
        # Calculate the variance of cluster centers for this feature
        center_variance = cluster_centers[feature].var()
        
        # Calculate the overall variance of this feature in the dataset
        overall_variance = X[feature].var()
        
        # Feature importance as ratio of between-cluster variance to total variance
        if overall_variance > 0:
            importance = center_variance / overall_variance
        else:
            importance = 0
            
        importance_scores.append({
            'Feature': feature,
            'Importance_Score': importance
        })
    
    # Convert to DataFrame and sort by importance
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('Importance_Score', ascending=False)
    
    return importance_df, data, cluster_centers

def visualize_results(data, feature_importance, cluster_centers):
    """
    Visualize clustering results and feature importance
    """
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(20)
    sns.barplot(x='Importance_Score', y='Feature', data=top_features)
    plt.title('Top 20 Feature Importance for Clustering')
    plt.tight_layout()
    plt.show()
    
    # Visualize clusters in 2D using PCA
    try:
        # Remove non-numeric columns and drop Ticker and Cluster for PCA
        features = data.drop(['Cluster', 'Ticker'], axis=1, errors='ignore')
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Handle NaN and infinite values
        numeric_features = numeric_features.fillna(numeric_features.mean())
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        # Additional check for any remaining NaNs
        if numeric_features.isna().any().any():
            # Drop problematic columns
            problematic_cols = numeric_features.columns[numeric_features.isna().any()].tolist()
            print(f"Warning: Dropping columns with NaN values for PCA: {problematic_cols}")
            numeric_features = numeric_features.drop(columns=problematic_cols)
        
        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        
        # Final check for NaNs in scaled data
        if np.isnan(scaled_features).any():
            print("Warning: NaNs detected after scaling in PCA. Replacing with zeros...")
            scaled_features = np.nan_to_num(scaled_features, nan=0.0)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], 
                       hue=data['Cluster'], palette='viridis', s=100, alpha=0.7)
        plt.title('Clusters Visualization using PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
    except Exception as e:
        print(f"Error in PCA visualization: {e}")
        print("Skipping PCA visualization")
    
    # Plot cluster centers for top features
    try:
        top_features = feature_importance.head(10)['Feature'].tolist()
        
        # Only keep features that exist in cluster_centers
        top_features = [f for f in top_features if f in cluster_centers.columns]
        
        if top_features:
            fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 3*len(top_features)))
            
            # Handle case when there's only one feature (axes is not array)
            if len(top_features) == 1:
                axes = [axes]
                
            for i, feature in enumerate(top_features):
                centers = cluster_centers[feature].values
                axes[i].bar(range(len(centers)), centers)
                axes[i].set_title(f'Cluster Centers for {feature}')
                axes[i].set_xlabel('Cluster')
                axes[i].set_ylabel(feature)
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error in cluster centers visualization: {e}")
        print("Skipping cluster centers visualization")
    
    # Feature distribution by cluster for top 5 features
    try:
        for feature in top_features[:5]:
            if feature in data.columns:
                plt.figure(figsize=(12, 6))
                for cluster in range(len(cluster_centers)):
                    cluster_data = data[data['Cluster'] == cluster][feature].dropna()
                    if not cluster_data.empty and len(cluster_data) > 1:  # Need at least 2 points for KDE
                        sns.kdeplot(cluster_data, label=f'Cluster {cluster}')
                plt.title(f'Distribution of {feature} by Cluster')
                plt.legend()
                plt.show()
    except Exception as e:
        print(f"Error in distribution visualization: {e}")
        print("Skipping distribution visualization")

def comprehensive_crypto_analysis(tickers, period='1y', interval='1d', max_k=10):
    """
    Main function to run the entire analysis
    """
    # Step 1: Fetch basic crypto data
    data = fetch_crypto_data(tickers, period, interval)
    print(f"Fetched data shape: {data.shape}")
    
    # Step 2: Add technical indicators using TALib
    data = enrich_with_technical_indicators(data)
    print(f"After adding technical indicators, data shape: {data.shape}")
    
    # Step 3: Add market-wide indicators and calendar features
    data = add_market_indicators(data)
    print(f"After adding market indicators, data shape: {data.shape}")
    
    # Step 4: Add relative strength between coins
    data = add_relative_strength_indicators(data)
    print(f"After adding relative strength, data shape: {data.shape}")
    
    # Step 5: Try to add market sentiment (Fear & Greed Index)
    sentiment_df = fetch_market_sentiment()
    if sentiment_df is not None:
        try:
            # Ensure both DataFrames have timezone-naive DatetimeIndex
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            if sentiment_df.index.tz is not None:
                sentiment_df.index = sentiment_df.index.tz_localize(None)
            
            # Merge sentiment data with main dataframe
            data = pd.merge(data, sentiment_df, left_index=True, right_index=True, how='left')
            data['Fear_Greed_Index'] = data['value'].fillna(method='ffill')
            data = data.drop('value', axis=1, errors='ignore')
            print("Successfully added Fear & Greed Index")
        except Exception as e:
            print(f"Error adding sentiment data: {e}")
            print("Continuing without sentiment data")
    
    # Step 6: Try to add Google Trends data
    try:
        data = add_google_trends_data(data, ['bitcoin', 'cryptocurrency', 'crypto crash', 'crypto bull'])
    except Exception as e:
        print(f"Error adding Google Trends data: {e}")
        print("Continuing without Google Trends data")
    
    # Step 7: Add synthetic features and interactions
    data = add_synthetic_features(data)
    print(f"After adding synthetic features, data shape: {data.shape}")
    
    # Step 8: Prepare data for clustering
    X, feature_cols = prepare_for_clustering(data)
    print(f"Number of features for clustering: {len(feature_cols)}")
    
    # Step 9: Find optimal number of clusters
    print("\nFinding optimal number of clusters...")
    optimal_k = find_optimal_clusters(X, max_k)
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Step 10: Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_df, clustered_data, cluster_centers = analyze_feature_importance(
        data, feature_cols, optimal_k)
    
    print("\nTop 50 Feature Importance Ranking:")
    print(importance_df.head(50))
    
    # Step 11: Visualize results
    print("\nGenerating visualizations...")
    visualize_results(clustered_data, importance_df, cluster_centers)
    
    return importance_df, clustered_data, cluster_centers, feature_cols

# Example usage
if __name__ == "__main__":
    # List of cryptocurrencies to analyze
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 
                      'DOT-USD', 'AVAX-USD', 'LINK-USD', 'DOGE-USD', 'SHIB-USD']
    
    # Run the comprehensive analysis
    importance, clustered_data, centers, features = comprehensive_crypto_analysis(
        crypto_tickers, 
        period='1y',
        interval='1d',
        max_k=8
    )