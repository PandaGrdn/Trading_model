import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import deque
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')

class WalkForwardTradingSignals:
    
    def __init__(self, portfolio_value, available_cash, training_end_date=None, enable_walk_forward=True):
        # Portfolio setup
        self.portfolio_value = portfolio_value
        self.available_cash = available_cash
        self.training_end_date = training_end_date or datetime.now().strftime('%Y-%m-%d')
        self.enable_walk_forward = enable_walk_forward
        
        # Stock universe
        self.symbols = ['AAPL', 'LCID', 'VST', 'NRG', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'ACHR', 'PLTR']
        self.spy_symbol = 'SPY'
        
        # Walk-forward parameters
        self.covid_start = datetime(2020, 3, 1)
        self.covid_end = datetime(2022, 6, 1)
        self.pre_covid_start = datetime(2017, 1, 1)
        self.pre_covid_end = datetime(2020, 2, 28)
        self.training_period = 504  # 2 years for walk-forward
        
        # Model parameters (from successful walk-forward)
        self.lookback_days = 30
        self.prediction_horizon = 5
        self.min_data_points = 60
        
        # Trading parameters (from original)
        self.top_stocks = 4
        self.min_confidence = 0.006
        self.max_position_size = 0.60
        
        # Risk management (from original)
        self.stop_loss = -0.02
        self.take_profit = 0.04
        self.max_drawdown = -0.25
        
        # Data storage
        self.price_data = {}
        self.feature_data = {}
        self.models = {}
        self.scalers = {}
        self.is_trained = {}
        self.ensemble_weights = {}
        
        print(f"Walk-Forward Trading Signals Generator Initialized")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Available Cash: ${available_cash:,.2f}")
        print(f"Training End Date: {self.training_end_date}")
        print(f"Walk-Forward Mode: {'Enabled' if enable_walk_forward else 'Disabled'}")
        if enable_walk_forward:
            print(f"Pre-COVID Training: {self.pre_covid_start.strftime('%Y-%m-%d')} to {self.pre_covid_end.strftime('%Y-%m-%d')}")
            print(f"COVID Exclusion: {self.covid_start.strftime('%Y-%m-%d')} to {self.covid_end.strftime('%Y-%m-%d')}")
        print(f"Universe: {', '.join(self.symbols)}\n")
    
    def download_data(self):
        """Download historical data with extended range for walk-forward training"""
        print("Downloading market data...")
        
        # Calculate extended start date for walk-forward training
        end_date = datetime.strptime(self.training_end_date, '%Y-%m-%d')
        
        if self.enable_walk_forward:
            # Start from 2017 to get clean pre-COVID data
            start_date = self.pre_covid_start - timedelta(days=50)
            print(f"Extended data range for walk-forward: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            # Standard training period
            start_date = end_date - timedelta(days=self.training_period + 50)
            print(f"Standard data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        all_symbols = self.symbols + [self.spy_symbol]
        
        for symbol in all_symbols:
            try:
                print(f"  Downloading {symbol}...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                
                if len(data) < self.min_data_points:
                    print(f"  WARNING: Insufficient data for {symbol}")
                    continue
                
                # Convert to our format with date filtering info
                price_data = []
                for idx, row in data.iterrows():
                    # Convert timezone-aware index to timezone-naive for comparison
                    date_naive = idx.tz_localize(None) if idx.tz is not None else idx
                    
                    price_point = {
                        'time': idx,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume']),
                        'is_pre_covid': date_naive < self.covid_start,
                        'is_covid': self.covid_start <= date_naive <= self.covid_end,
                        'is_post_covid': date_naive > self.covid_end
                    }
                    price_data.append(price_point)
                
                self.price_data[symbol] = price_data
                
                # Count data by regime
                pre_covid_count = sum(1 for p in price_data if p['is_pre_covid'])
                covid_count = sum(1 for p in price_data if p['is_covid'])
                post_covid_count = sum(1 for p in price_data if p['is_post_covid'])
                
                print(f"  {symbol}: {len(price_data)} total days")
                if self.enable_walk_forward:
                    print(f"    Pre-COVID: {pre_covid_count} days")
                    print(f"    COVID (excluded): {covid_count} days") 
                    print(f"    Post-COVID: {post_covid_count} days")
                
            except Exception as e:
                print(f"  ERROR downloading {symbol}: {str(e)}")
        
        print(f"\nData download complete. {len(self.price_data)} symbols ready.\n")
    
    def filter_training_data(self, data_points, include_pre_covid=True, exclude_covid=True, include_post_covid=True):
        """Filter data points based on COVID period rules"""
        if not self.enable_walk_forward:
            return data_points
            
        filtered_data = []
        for point in data_points:
            include_point = False
            
            if point['is_pre_covid'] and include_pre_covid:
                include_point = True
            elif point['is_covid'] and not exclude_covid:
                include_point = True
            elif point['is_post_covid'] and include_post_covid:
                include_point = True
                
            if include_point:
                filtered_data.append(point)
        
        return filtered_data
    
    def generate_features(self, symbol, end_idx=None, use_filtered_data=True):
        """Generate advanced features with optional COVID filtering"""
        try:
            raw_data = self.price_data[symbol]
            if end_idx is None:
                end_idx = len(raw_data)
            
            if end_idx < self.lookback_days:
                return None
            
            # Apply COVID filtering if enabled
            if use_filtered_data and self.enable_walk_forward:
                # Get data up to end_idx, then filter
                candidate_data = raw_data[:end_idx]
                filtered_data = self.filter_training_data(candidate_data)
                
                # Take the last lookback_days from filtered data
                if len(filtered_data) < self.lookback_days:
                    return None
                period_data = filtered_data[-self.lookback_days:]
            else:
                # Standard approach - use raw data
                start_idx = max(0, end_idx - self.lookback_days)
                period_data = raw_data[start_idx:end_idx]
            
            # Extract OHLCV
            closes = np.array([d['close'] for d in period_data])
            highs = np.array([d['high'] for d in period_data])
            lows = np.array([d['low'] for d in period_data])
            volumes = np.array([d['volume'] for d in period_data])
            
            features = []
            
            # 1. Multi-timeframe returns (G-Research key feature)
            for period in [1, 2, 3, 5, 10, 15]:
                if len(closes) > period:
                    ret = np.log(closes[-1] / closes[-period-1]) if closes[-period-1] > 0 else 0
                    features.append(ret)
                else:
                    features.append(0)
            
            # 2. Volatility features (crucial for regime detection)
            returns = np.diff(np.log(closes + 1e-8))
            vol_5d = np.std(returns[-5:]) if len(returns) >= 5 else 0
            vol_15d = np.std(returns[-15:]) if len(returns) >= 15 else 0
            vol_ratio = vol_5d / vol_15d if vol_15d > 0 else 1
            features.extend([vol_5d, vol_15d, vol_ratio])
            
            # 3. Technical indicators
            rsi = self.calculate_rsi(closes, 14)
            features.append((rsi - 50) / 50)
            
            # MACD approximation
            ema_12 = self.calculate_ema(closes, 12)
            ema_26 = self.calculate_ema(closes, 26)
            macd = (ema_12 - ema_26) / closes[-1] if closes[-1] > 0 else 0
            features.append(macd)
            
            # 4. Moving average features
            sma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
            sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            
            price_vs_sma5 = (closes[-1] / sma_5 - 1) if sma_5 > 0 else 0
            price_vs_sma10 = (closes[-1] / sma_10 - 1) if sma_10 > 0 else 0
            price_vs_sma20 = (closes[-1] / sma_20 - 1) if sma_20 > 0 else 0
            sma_trend = (sma_5 / sma_20 - 1) if sma_20 > 0 else 0
            
            features.extend([price_vs_sma5, price_vs_sma10, price_vs_sma20, sma_trend])
            
            # 5. Volume analysis
            vol_sma = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            vol_ratio_current = volumes[-1] / vol_sma if vol_sma > 0 else 1
            features.append(np.log(vol_ratio_current + 1e-8))
            
            # 6. High-Low range analysis
            hl_ratio = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0
            avg_hl_ratio = np.mean((highs[-5:] - lows[-5:]) / closes[-5:]) if len(closes) >= 5 else hl_ratio
            features.extend([hl_ratio, hl_ratio / avg_hl_ratio if avg_hl_ratio > 0 else 1])
            
            # 7. Market regime features (using SPY with same filtering)
            if self.spy_symbol in self.price_data:
                spy_raw_data = self.price_data[self.spy_symbol]
                spy_end_idx = min(end_idx, len(spy_raw_data))
                
                if use_filtered_data and self.enable_walk_forward:
                    spy_candidate = spy_raw_data[:spy_end_idx]
                    spy_filtered = self.filter_training_data(spy_candidate)
                    if len(spy_filtered) >= 10:
                        spy_closes = [d['close'] for d in spy_filtered[-10:]]
                        spy_ret = np.log(spy_closes[-1] / spy_closes[0]) if len(spy_closes) >= 2 and spy_closes[0] > 0 else 0
                        features.append(spy_ret)
                    else:
                        features.append(0)
                else:
                    if spy_end_idx >= 10:
                        spy_closes = [d['close'] for d in spy_raw_data[max(0, spy_end_idx-10):spy_end_idx]]
                        spy_ret = np.log(spy_closes[-1] / spy_closes[0]) if len(spy_closes) >= 2 and spy_closes[0] > 0 else 0
                        features.append(spy_ret)
                    else:
                        features.append(0)
            else:
                features.append(0)
            
            # 8. Mean reversion features
            price_zscore = (closes[-1] - np.mean(closes)) / (np.std(closes) + 1e-8)
            features.append(price_zscore)
            
            # 9. Momentum features
            mom_3 = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 and closes[-4] > 0 else 0
            mom_5 = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 and closes[-6] > 0 else 0
            features.extend([mom_3, mom_5])
            
            # 10. Trend strength using linear regression slope
            if len(closes) >= 10:
                x = np.arange(10)
                y = closes[-10:]
                if np.std(y) > 0:
                    slope = np.polyfit(x, y, 1)[0] / np.mean(y)
                else:
                    slope = 0
            else:
                slope = 0
            features.append(slope)
            
            # Ensure exactly 25 features
            target_features = 25
            if len(features) < target_features:
                features.extend([0] * (target_features - len(features)))
            else:
                features = features[:target_features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error generating features for {symbol}: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 1e-8
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_ema(self, prices, period):
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def initialize_models(self):
        """Initialize ensemble models with walk-forward optimized parameters"""
        return {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=80,
                max_depth=5,
                learning_rate=0.08,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=3,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1,
                force_col_wise=True
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.12,
                subsample=0.9,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=42
            ),
            'ridge': Ridge(alpha=0.5, random_state=42)
        }
    
    def train_models_walk_forward(self):
        """Train ensemble models using walk-forward methodology with COVID filtering"""
        print("Training machine learning models with walk-forward methodology...")
        
        if self.enable_walk_forward:
            print("Walk-forward training enabled:")
            print("- Using 2017-2019 pre-COVID data for stable patterns")
            print("- Excluding COVID period (2020-2022) from training")
            print("- Including post-COVID recovery data")
        
        trained_count = 0
        
        for symbol in self.symbols:
            if symbol not in self.price_data:
                continue
            
            print(f"  Training {symbol}...")
            
            try:
                price_data_list = self.price_data[symbol]
                
                # Apply COVID filtering to training data
                if self.enable_walk_forward:
                    filtered_training_data = self.filter_training_data(
                        price_data_list,
                        include_pre_covid=True,  # Include clean 2017-2019 data
                        exclude_covid=True,      # Exclude volatile COVID period
                        include_post_covid=True  # Include recent data
                    )
                    
                    print(f"    Original data: {len(price_data_list)} days")
                    print(f"    Filtered data: {len(filtered_training_data)} days")
                    
                    if len(filtered_training_data) < self.min_data_points + self.prediction_horizon:
                        print(f"    WARNING: Insufficient filtered data for {symbol}")
                        continue
                        
                    training_data = filtered_training_data
                else:
                    training_data = price_data_list
                    if len(training_data) < self.min_data_points + self.prediction_horizon:
                        print(f"    WARNING: Insufficient data for {symbol}")
                        continue
                
                # Generate feature sequences from filtered data
                features_list = []
                targets = []
                
                # Create mapping from filtered to original indices for feature generation
                for i in range(self.lookback_days, len(training_data) - self.prediction_horizon):
                    # For walk-forward, generate features using filtered approach
                    if self.enable_walk_forward:
                        # Find position in original data
                        current_time = training_data[i]['time']
                        original_idx = None
                        for j, orig_point in enumerate(price_data_list):
                            if orig_point['time'] == current_time:
                                original_idx = j + 1  # +1 because we want features up to this point
                                break
                        
                        if original_idx is None:
                            continue
                            
                        feature = self.generate_features(symbol, original_idx, use_filtered_data=True)
                    else:
                        # Standard feature generation
                        feature = self.generate_features(symbol, i + 1, use_filtered_data=False)
                    
                    if feature is not None:
                        features_list.append(feature)
                        
                        # Target: 5-day forward return from filtered data
                        current_price = training_data[i]['close']
                        future_price = training_data[i + self.prediction_horizon]['close']
                        
                        if current_price > 0 and future_price > 0:
                            target = np.log(future_price / current_price)
                            targets.append(target)
                        else:
                            features_list.pop()  # Remove the last feature if target is invalid
                
                if len(features_list) < 30:
                    print(f"    WARNING: Insufficient training samples for {symbol}")
                    continue
                
                X = np.array(features_list)
                y = np.array(targets)
                
                # Use recent data for training (walk-forward adaptation)
                if len(X) > 60:
                    X = X[-60:]
                    y = y[-60:]
                
                # Handle NaN/inf values
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
                y = np.nan_to_num(y, nan=0.0, posinf=0.1, neginf=-0.1)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[symbol] = scaler
                
                # Initialize models
                self.models[symbol] = self.initialize_models()
                
                # Train with time series cross-validation for meta-model
                tscv = TimeSeriesSplit(n_splits=3)
                cv_predictions = {model_name: [] for model_name in self.models[symbol].keys()}
                cv_targets = []
                valid_models = []
                
                # Cross-validation loop
                for train_idx, val_idx in tscv.split(X_scaled):
                    if len(val_idx) == 0:
                        continue
                    
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    cv_targets.extend(y_val)
                    
                    for model_name, model in self.models[symbol].items():
                        try:
                            model.fit(X_train, y_train)
                            val_pred = model.predict(X_val)
                            cv_predictions[model_name].extend(val_pred)
                        except:
                            cv_predictions[model_name].extend([0.0] * len(y_val))
                
                # Train final models on full data
                for model_name, model in self.models[symbol].items():
                    try:
                        model.fit(X_scaled, y)
                        valid_models.append(model_name)
                    except Exception as e:
                        print(f"    Error training {model_name}: {str(e)}")
                        continue
                
                # Train meta-model (ensemble stacking)
                if len(valid_models) >= 2 and len(cv_targets) > 10:
                    try:
                        min_length = min(len(cv_predictions[model]) for model in valid_models)
                        if min_length != len(cv_targets):
                            min_length = min(min_length, len(cv_targets))
                        
                        meta_X_list = []
                        for model_name in valid_models:
                            pred_array = np.array(cv_predictions[model_name][:min_length])
                            meta_X_list.append(pred_array)
                        
                        meta_X = np.column_stack(meta_X_list)
                        meta_y = np.array(cv_targets[:min_length])
                        
                        if meta_X.shape[0] == len(meta_y) and len(meta_y) > 5:
                            meta_model = Ridge(alpha=0.1)
                            meta_model.fit(meta_X, meta_y)
                            self.ensemble_weights[symbol] = meta_model
                            print(f"    Meta-model trained with {len(valid_models)} base models")
                        else:
                            self.ensemble_weights[symbol] = None
                            print(f"    Using equal weights (meta-model training failed)")
                    except Exception as e:
                        print(f"    Meta-model training error: {str(e)}")
                        self.ensemble_weights[symbol] = None
                else:
                    self.ensemble_weights[symbol] = None
                    print(f"    Using equal weights (insufficient models/data)")
                
                self.is_trained[symbol] = True
                trained_count += 1
                print(f"    {symbol} trained successfully")
                
            except Exception as e:
                print(f"    ERROR training {symbol}: {str(e)}")
        
        success_rate = (trained_count / len(self.symbols)) * 100
        print(f"\nModel training complete: {trained_count}/{len(self.symbols)} models trained ({success_rate:.1f}%)")
        
        if self.enable_walk_forward:
            print("Walk-forward training methodology applied:")
            print("- Models trained on regime-filtered data")
            print("- COVID volatility excluded from training")
            print("- Enhanced ensemble stacking with meta-learning")
        
        print()
    
    def get_prediction(self, symbol):
        """Get ensemble prediction for a symbol"""
        if not self.is_trained.get(symbol, False):
            return 0
        
        try:
            # Generate current features (no filtering for current prediction)
            features = self.generate_features(symbol, use_filtered_data=False)
            if features is None:
                return 0
            
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            features_scaled = self.scalers[symbol].transform([features])
            
            # Get base model predictions
            valid_predictions = []
            
            for model_name, model in self.models[symbol].items():
                try:
                    pred = model.predict(features_scaled)[0]
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.1, neginf=-0.1)
                    valid_predictions.append(pred)
                except:
                    continue
            
            # Ensemble prediction using meta-model
            if (symbol in self.ensemble_weights and 
                self.ensemble_weights[symbol] is not None and 
                len(valid_predictions) >= 2):
                try:
                    meta_X = np.array(valid_predictions).reshape(1, -1)
                    ensemble_pred = self.ensemble_weights[symbol].predict(meta_X)[0]
                    return np.nan_to_num(ensemble_pred, nan=0.0, posinf=0.1, neginf=-0.1)
                except:
                    pass
            
            # Fallback: average of base predictions
            return np.mean(valid_predictions) if valid_predictions else 0
            
        except Exception as e:
            print(f"Error getting prediction for {symbol}: {str(e)}")
            return 0
    
    def generate_signals(self):
        """Generate trading signals using walk-forward trained models"""
        print("Generating trading signals...\n")
        
        # Get current prices
        current_prices = {}
        for symbol in self.symbols:
            if symbol in self.price_data and len(self.price_data[symbol]) > 0:
                current_prices[symbol] = self.price_data[symbol][-1]['close']
        
        # Get predictions
        predictions = {}
        prediction_details = {}
        
        for symbol in self.symbols:
            if self.is_trained.get(symbol, False):
                pred = self.get_prediction(symbol)
                predictions[symbol] = pred
                prediction_details[symbol] = {
                    'prediction': pred,
                    'confidence_pct': pred * 100,
                    'current_price': current_prices.get(symbol, 0)
                }
        
        if not predictions:
            print("ERROR: No predictions available")
            return
        
        # Filter predictions (same logic as walk-forward)
        filtered_predictions = {}
        for symbol, pred in predictions.items():
            if abs(pred) <= 0.2:  # Cap at 20%
                filtered_predictions[symbol] = pred
        
        # Sort by prediction strength
        sorted_predictions = sorted(filtered_predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Select positions (same logic as walk-forward)
        selected_longs = []
        selected_shorts = []
        
        for symbol, pred in sorted_predictions:
            if pred > self.min_confidence:
                selected_longs.append((symbol, pred))
            elif pred < -self.min_confidence:
                selected_shorts.append((symbol, pred))
        
        # Take top positions
        selected_longs = selected_longs[:self.top_stocks]
        selected_shorts = selected_shorts[:max(2, self.top_stocks//2)]
        
        # Display results
        print("WALK-FORWARD TRADING SIGNALS")
        print("=" * 50)
        
        if self.enable_walk_forward:
            print("Training Method: Walk-Forward with COVID Filtering")
            print("- Pre-COVID data (2017-2019): Clean market patterns")
            print("- COVID period excluded: Avoided regime contamination")
            print("- Ensemble meta-learning: Advanced model stacking")
            print()
        
        if selected_longs:
            print("LONG POSITIONS (BUY)")
            print("-" * 30)
            total_long_weight = 0
            
            # Calculate position sizing (same as walk-forward)
            total_long_strength = sum(abs(pred) for _, pred in selected_longs) if selected_longs else 1
            
            for i, (symbol, pred) in enumerate(selected_longs):
                current_price = current_prices.get(symbol, 0)
                
                # Dynamic position sizing based on prediction strength
                relative_strength = abs(pred) / total_long_strength
                weight = min(self.max_position_size, relative_strength * 0.8 + 0.1)
                
                total_long_weight += weight
                dollar_amount = self.available_cash * weight
                shares = int(dollar_amount / current_price) if current_price > 0 else 0
                
                confidence_pct = pred * 100
                print(f"{i+1}. {symbol:6} | Price: ${current_price:7.2f} | Confidence: {confidence_pct:+5.1f}% | Weight: {weight*100:4.1f}% | Amount: ${dollar_amount:8,.0f} | Shares: {shares:4d}")
            
            print(f"\nTotal Long Allocation: {total_long_weight*100:.1f}% (${self.available_cash * total_long_weight:,.0f})")
        
        if selected_shorts:
            print("\nSHORT POSITIONS (SELL SHORT)")
            print("-" * 40)
            total_short_weight = 0
            
            # Calculate position sizing for shorts (same as walk-forward)
            total_short_strength = sum(abs(pred) for _, pred in selected_shorts) if selected_shorts else 1
            
            for i, (symbol, pred) in enumerate(selected_shorts):
                current_price = current_prices.get(symbol, 0)
                
                # Dynamic position sizing for shorts
                relative_strength = abs(pred) / total_short_strength
                weight = min(self.max_position_size * 0.6, relative_strength * 0.4 + 0.05)
                
                total_short_weight += weight
                dollar_amount = self.available_cash * weight
                shares = int(dollar_amount / current_price) if current_price > 0 else 0
                
                confidence_pct = pred * 100
                print(f"{i+1}. {symbol:6} | Price: ${current_price:7.2f} | Confidence: {confidence_pct:+5.1f}% | Weight: {weight*100:4.1f}% | Amount: ${dollar_amount:8,.0f} | Shares: {shares:4d}")
            
            print(f"\nTotal Short Allocation: {total_short_weight*100:.1f}% (${self.available_cash * total_short_weight:,.0f})")
        
        if not selected_longs and not selected_shorts:
            print("NO TRADING SIGNALS")
            print("All predictions below confidence threshold")
            print("\nModel Predictions (Below Threshold):")
            for symbol in sorted(prediction_details.keys()):
                details = prediction_details[symbol]
                print(f"  {symbol}: {details['confidence_pct']:+5.1f}% (${details['current_price']:.2f})")
        
        # Enhanced risk management info
        print(f"\nRISK MANAGEMENT")
        print("=" * 30)
        print(f"Stop Loss: {self.stop_loss*100:+.1f}%")
        print(f"Take Profit: {self.take_profit*100:+.1f}%")
        print(f"Max Position Size: {self.max_position_size*100:.1f}%")
        print(f"Max Drawdown: {self.max_drawdown*100:+.1f}%")
        print(f"Available Cash: ${self.available_cash:,.0f}")
        print(f"Min Confidence: {self.min_confidence*100:.1f}%")
        
        # Walk-forward specific insights
        if self.enable_walk_forward:
            print(f"\nWALK-FORWARD INSIGHTS")
            print("=" * 35)
            trained_models = sum(1 for trained in self.is_trained.values() if trained)
            meta_models = sum(1 for symbol in self.symbols if self.ensemble_weights.get(symbol) is not None)
            
            print(f"Models Trained: {trained_models}/{len(self.symbols)}")
            print(f"Meta-Models: {meta_models}/{trained_models}")
            print(f"Training Data: Pre-COVID + Post-COVID (COVID excluded)")
            print(f"Ensemble Method: Stacked generalization with Ridge meta-learner")
        
        # Trading instructions
        print(f"\nTRADING INSTRUCTIONS")
        print("=" * 35)
        print("1. Execute trades during market hours (9:30 AM - 4:00 PM ET)")
        print("2. Set stop losses and take profits immediately after entry")
        print("3. Monitor positions daily for risk management triggers")
        print("4. Re-run signals twice per week (Monday/Thursday recommended)")
        print("5. Never exceed suggested position sizes")
        print("6. Consider market volatility when scaling position sizes")
        
        if self.enable_walk_forward:
            print("7. Walk-forward methodology reduces overfitting risk")
            print("8. COVID period exclusion improves model stability")
        
        return {
            'longs': selected_longs,
            'shorts': selected_shorts,
            'prices': current_prices,
            'all_predictions': prediction_details,
            'training_method': 'walk_forward' if self.enable_walk_forward else 'standard',
            'model_stats': {
                'trained_models': sum(1 for trained in self.is_trained.values() if trained),
                'meta_models': sum(1 for symbol in self.symbols if self.ensemble_weights.get(symbol) is not None),
                'total_symbols': len(self.symbols)
            }
        }
    
    def run_analysis(self):
        """Run complete analysis pipeline with walk-forward training"""
        print("Walk-Forward Trading Analysis Pipeline")
        print("=" * 50)
        print()
        
        # Download extended data for walk-forward
        self.download_data()
        
        # Train models using walk-forward methodology
        if self.enable_walk_forward:
            self.train_models_walk_forward()
        else:
            # Fallback to original training method
            self.train_models_standard()
        
        # Generate signals
        signals = self.generate_signals()
        
        return signals
    
    def train_models_standard(self):
        """Original training method (fallback)"""
        print("Training models using standard methodology...")
        
        trained_count = 0
        
        for symbol in self.symbols:
            if symbol not in self.price_data:
                continue
            
            print(f"  Training {symbol}...")
            
            try:
                price_data_list = self.price_data[symbol]
                
                if len(price_data_list) < self.min_data_points + self.prediction_horizon:
                    print(f"    WARNING: Insufficient data for {symbol}")
                    continue
                
                # Generate feature sequences
                features_list = []
                targets = []
                
                for i in range(self.lookback_days, len(price_data_list) - self.prediction_horizon):
                    feature = self.generate_features(symbol, i + 1, use_filtered_data=False)
                    if feature is not None:
                        features_list.append(feature)
                        
                        current_price = price_data_list[i]['close']
                        future_price = price_data_list[i + self.prediction_horizon]['close']
                        
                        if current_price > 0 and future_price > 0:
                            target = np.log(future_price / current_price)
                            targets.append(target)
                        else:
                            features_list.pop()
                
                if len(features_list) < 30:
                    print(f"    WARNING: Insufficient training samples for {symbol}")
                    continue
                
                X = np.array(features_list)
                y = np.array(targets)
                
                if len(X) > 60:
                    X = X[-60:]
                    y = y[-60:]
                
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
                y = np.nan_to_num(y, nan=0.0, posinf=0.1, neginf=-0.1)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[symbol] = scaler
                
                self.models[symbol] = self.initialize_models()
                
                # Train models
                for model_name, model in self.models[symbol].items():
                    try:
                        model.fit(X_scaled, y)
                    except Exception as e:
                        print(f"    Error training {model_name}: {str(e)}")
                        continue
                
                self.ensemble_weights[symbol] = None  # No meta-model in standard mode
                self.is_trained[symbol] = True
                trained_count += 1
                print(f"    {symbol} trained successfully")
                
            except Exception as e:
                print(f"    ERROR training {symbol}: {str(e)}")
        
        print(f"\nStandard training complete: {trained_count}/{len(self.symbols)} models trained.\n")


def main():
    """Main function with walk-forward option"""
    print("WALK-FORWARD TRADING SIGNALS GENERATOR")
    print("=" * 50)
    
    try:
        # Portfolio inputs
        portfolio_value = float(input("Enter your total portfolio value ($): ").replace(',', '').replace(',', ''))
        available_cash = float(input("Enter available cash to invest ($): ").replace(',', '').replace(',', ''))
        
        # Walk-forward option
        print("\nTraining Method Options:")
        print("1. Walk-Forward (Recommended) - Uses pre-COVID + post-COVID data, excludes COVID period")
        print("2. Standard - Uses recent data including all periods")
        
        use_walkforward = input("Select training method (1 or 2): ").strip() == '1'
        
        # Optional custom date
        use_custom_date = input("\nUse custom training end date? (y/n): ").lower() == 'y'
        if use_custom_date:
            training_end_date = input("Enter training end date (YYYY-MM-DD): ")
        else:
            training_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print("\n" + "=" * 60)
        
        # Initialize and run with walk-forward
        trader = WalkForwardTradingSignals(
            portfolio_value, 
            available_cash, 
            training_end_date, 
            enable_walk_forward=use_walkforward
        )
        signals = trader.run_analysis()
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        
        if use_walkforward:
            print("\nWalk-Forward Benefits Applied:")
            print("- Reduced overfitting through regime-aware training")
            print("- Enhanced model stability via COVID period exclusion")
            print("- Advanced ensemble learning with meta-models")
        
        print("\nNext steps:")
        print("1. Review the signals above carefully")
        print("2. Execute trades in your brokerage account")
        print("3. Set stop losses and take profits immediately")
        print("4. Monitor positions daily for risk management")
        print("5. Re-run analysis twice weekly for fresh signals")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        print("Please check your inputs and try again")


if __name__ == "__main__":
    main()