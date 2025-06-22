import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

# --- Configuration ---
CACHE_DIR = 'data_cache'
TICKERS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD',
    'AVAX-USD', 'LINK-USD', 'DOGE-USD'
]
TEST_START_DATE = '2025-01-01'
TEST_END_DATE = '2025-06-18'

class CryptoBacktester:
    """
    A professional, event-driven backtester for crypto trading strategies.

    This class simulates portfolio performance based on pre-generated trading
    signals, providing detailed performance metrics and visualizations.

    Key features:
    - Manages a portfolio with a fixed initial capital.
    - Correctly handles cash flow: cash decreases on buys and increases on sells.
    - Executes trades based on signals, stop-loss, and take-profit levels.
    - Assumes trades can be executed within a daily High-Low range.
    - Accounts for transaction fees.
    - Generates a comprehensive trade log and daily portfolio performance history.
    """

    def __init__(self, signals_df, initial_capital=100000, fee_rate=0.001):
        """
        Initializes the backtester.

        Args:
            signals_df (pd.DataFrame): DataFrame containing price data and trading signals.
                                       Must include 'Ticker', 'Close', 'High', 'Low', 'Signal',
                                       'Position_Size_Pct', 'Stop_Loss', 'Take_Profit'.
            initial_capital (float): The starting capital for the portfolio.
            fee_rate (float): The transaction fee rate (e.g., 0.001 for 0.1%).
        """
        self.signals = signals_df.sort_index()
        self.initial_capital = float(initial_capital)
        self.fee_rate = float(fee_rate)

        # --- State Variables ---
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.holdings = {}  # Stores active positions, e.g., {'BTC-USD': {'quantity': 1.5, 'entry_price': 50000, ...}}
        
        # --- Logging ---
        self.trade_log = []
        self.portfolio_history = []

    def run(self):
        """
        Runs the entire backtesting simulation from start to end date.
        """
        print("--- Starting Backtest Simulation ---")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Unique dates in the dataset, ensuring chronological order
        unique_dates = self.signals.index.unique().sort_values()

        for date in unique_dates:
            # Data for the current day
            day_data = self.signals.loc[date]
            
            # 1. Update portfolio value with current market prices and check for exits
            self._update_portfolio_and_check_exits(date, day_data)
            
            # 2. Check for new trade entries
            self._enter_new_trades(date, day_data)
            
            # 3. Record daily portfolio status
            self._record_daily_performance(date)

        print("--- Backtest Simulation Complete ---")
        return self._generate_results()

    def _update_portfolio_and_check_exits(self, date, day_data):
        """
        Updates market value of holdings and processes exits (SL/TP).
        """
        # --- NEW ROBUSTNESS CHECK ---
        # If only one row of data exists for this timestamp, day_data will be a Series.
        # We must convert it to a DataFrame to ensure consistent processing.
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T

        holdings_value = 0.0
        # Iterate over a copy of keys since we may modify the dictionary
        for ticker in list(self.holdings.keys()):
            position = self.holdings[ticker]
            
            # Get the current day's price data for the specific ticker
            # This check will now work correctly because day_data is guaranteed to be a DataFrame
            if ticker in day_data['Ticker'].values:
                ticker_day_data = day_data[day_data['Ticker'] == ticker].iloc[0]
                current_low = ticker_day_data['Low']
                current_high = ticker_day_data['High']
                current_close = ticker_day_data['Close']
            else:
                # If no data for the ticker on this day, use last known price for valuation
                # and skip exit checks. This handles cases where one asset has a holiday/no data.
                current_close = position['entry_price'] # Fallback
                holdings_value += position['quantity'] * current_close
                continue

            # Check for exit conditions
            exit_price = None
            exit_reason = None
            
            # Use .loc to access stop_loss and take_profit to avoid potential warnings
            stop_loss_level = position.get('stop_loss', np.nan)
            take_profit_level = position.get('take_profit', np.nan)

            if pd.notna(stop_loss_level) and current_low <= stop_loss_level:
                exit_price = stop_loss_level
                exit_reason = 'Stop-Loss Hit'
            elif pd.notna(take_profit_level) and current_high >= take_profit_level:
                exit_price = take_profit_level
                exit_reason = 'Take-Profit Hit'
                
            if exit_price:
                self._execute_sell(date, ticker, exit_price, exit_reason)
            else:
                # If not exited, add its current value to the total holdings value
                holdings_value += position['quantity'] * current_close
                
        # Update portfolio value with cash + value of remaining holdings
        self.portfolio_value = self.cash + holdings_value

    def _enter_new_trades(self, date, day_data):
        """
        Checks for and executes new buy signals for the current day.
        """
        # --- NEW ROBUSTNESS CHECK ---
        # If only one row of data exists for this timestamp, day_data will be a Series.
        # We must convert it back to a DataFrame to ensure consistent processing.
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T

        # Filter for rows with a buy signal (this will now work correctly)
        signals_today = day_data[day_data['Signal'] == 1]
        
        for _, signal_row in signals_today.iterrows():
            ticker = signal_row['Ticker']
            
            # Condition: Do not open a new position if one already exists for this ticker
            if ticker not in self.holdings:
                self._execute_buy(date, signal_row)
                
    def _execute_buy(self, date, signal_row):
        """
        Executes a buy order and updates portfolio state.
        THIS IS WHERE THE USER'S KEY REQUIREMENT IS HANDLED.
        """
        ticker = signal_row['Ticker']
        position_size_pct = signal_row['Position_Size_Pct'] / 100.0
        
        # **CRITICAL**: Position size is based on AVAILABLE CASH, not total portfolio value.
        # This prevents the exponential growth fallacy.
        trade_value = self.cash * position_size_pct
        
        if trade_value > 10: # Minimum trade size
            entry_price = signal_row['Close']
            fee = trade_value * self.fee_rate
            
            # Check if we have enough cash for the trade + fee
            if self.cash >= trade_value + fee:
                self.cash -= (trade_value + fee)
                quantity = trade_value / entry_price
                
                self.holdings[ticker] = {
                    'entry_date': date,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': signal_row['Stop_Loss'],
                    'take_profit': signal_row['Take_Profit'],
                }
                
                # Log the buy trade
                self.trade_log.append({
                    'Ticker': ticker,
                    'Direction': 'BUY',
                    'EntryDate': date,
                    'EntryPrice': entry_price,
                    'Quantity': quantity,
                    'Value': trade_value,
                    'Fee': fee
                })

    def _execute_sell(self, date, ticker, exit_price, reason):
        """
        Executes a sell order and updates portfolio state.
        """
        position = self.holdings.pop(ticker) # Remove from holdings
        
        trade_value = position['quantity'] * exit_price
        fee = trade_value * self.fee_rate
        self.cash += (trade_value - fee)
        
        pnl = (exit_price - position['entry_price']) * position['quantity'] - fee
        
        # Find the corresponding buy trade to log the full trade lifecycle
        for trade in reversed(self.trade_log):
            if trade['Ticker'] == ticker and 'ExitDate' not in trade:
                trade['ExitDate'] = date
                trade['ExitPrice'] = exit_price
                trade['PnL'] = pnl
                trade['Reason'] = reason
                break

    def _record_daily_performance(self, date):
        """
        Appends the current portfolio status to the history log.
        """
        self.portfolio_history.append({
            'Date': date,
            'PortfolioValue': self.portfolio_value,
            'Cash': self.cash,
            'HoldingsValue': self.portfolio_value - self.cash
        })
        
    def _generate_results(self):
        """
        Calculates and returns final performance metrics after the simulation.
        """
        if not self.portfolio_history:
            print("No portfolio history was recorded. Cannot generate results.")
            return None

        self.history_df = pd.DataFrame(self.portfolio_history).set_index('Date')
        self.trades_df = pd.DataFrame(self.trade_log)

        # --- Key Performance Indicators (KPIs) ---
        total_return = (self.history_df['PortfolioValue'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Drawdown Calculation
        rolling_max = self.history_df['PortfolioValue'].cummax()
        daily_drawdown = self.history_df['PortfolioValue'] / rolling_max - 1.0
        max_drawdown = daily_drawdown.min() * 100
        
        # Sharpe Ratio (assuming risk-free rate is 0)
        daily_returns = self.history_df['PortfolioValue'].pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) # Annualized
        else:
            sharpe_ratio = 0.0

        # Win/Loss Analysis from trade log
        if not self.trades_df.empty and 'PnL' in self.trades_df.columns:
            completed_trades = self.trades_df.dropna(subset=['PnL'])
            if not completed_trades.empty:
                win_rate = (completed_trades['PnL'] > 0).mean() * 100
                profit_factor = completed_trades[completed_trades['PnL'] > 0]['PnL'].sum() / \
                                abs(completed_trades[completed_trades['PnL'] < 0]['PnL'].sum())
            else:
                win_rate = 0.0
                profit_factor = float('inf')
        else:
            win_rate = 0.0
            profit_factor = float('inf')

        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.history_df['PortfolioValue'].iloc[-1],
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(completed_trades) if 'completed_trades' in locals() else 0,
        }
        return results

    def display_summary(self, results):
        """Prints a professional summary of the backtest results."""
        print("\n--- Backtest Performance Summary ---")
        print(f"Period: {self.history_df.index.min().date()} to {self.history_df.index.max().date()}")
        print("-" * 35)
        print(f"Initial Portfolio Value: ${results['initial_capital']:>12,.2f}")
        print(f"Final Portfolio Value:   ${results['final_portfolio_value']:>12,.2f}")
        print(f"Total Return:            {results['total_return_pct']:>12.2f}%")
        print("-" * 35)
        print(f"Max Drawdown:            {results['max_drawdown_pct']:>12.2f}%")
        print(f"Sharpe Ratio (Annualized): {results['sharpe_ratio']:>9.2f}")
        print("-" * 35)
        print(f"Total Completed Trades:  {results['total_trades']:>12}")
        print(f"Win Rate:                {results['win_rate_pct']:>12.2f}%")
        print(f"Profit Factor:           {results['profit_factor']:>12.2f}")
        print("-" * 35)

    def plot_results(self):
        """Generates plots for equity curve and drawdown."""
        if self.history_df.empty:
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Plot 1: Equity Curve
        ax1.plot(self.history_df.index, self.history_df['PortfolioValue'], label='Portfolio Value', color='navy')
        ax1.set_title('Portfolio Equity Curve', fontsize=16)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        
        # Plot 2: Drawdown
        daily_drawdown = (self.history_df['PortfolioValue'] / self.history_df['PortfolioValue'].cummax() - 1) * 100
        ax2.fill_between(daily_drawdown.index, daily_drawdown, 0, color='red', alpha=0.3)
        ax2.plot(daily_drawdown.index, daily_drawdown, color='red', linewidth=1)
        ax2.set_title('Portfolio Drawdown', fontsize=16)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # --- Configuration ---
    SIGNALS_CACHE_DIR = 'signals_cache'
    TEST_START_DATE = '2025-06-12'
    # Use the current date to match the filename from the generator script
    TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
    INITIAL_CAPITAL = 100000.0
    FEE_RATE = 0.001 # Example: 0.1% fee

    # --- Load Pre-Computed Signals ---
    signals_filename = os.path.join(SIGNALS_CACHE_DIR, f"signals_{TEST_START_DATE}_to_{TEST_END_DATE}.pkl")

    if not os.path.exists(signals_filename):
        print(f"--- ERROR: Signal file not found at '{signals_filename}' ---")
        print("Please run the `trading_model.py` script first to generate and save the signals.")
    else:
        print(f"Loading pre-computed signals from: {signals_filename}")
        final_signals = pd.read_pickle(signals_filename)

        # 1. Run the Backtester
        backtester = CryptoBacktester(
            signals_df=final_signals, 
            initial_capital=INITIAL_CAPITAL, 
            fee_rate=FEE_RATE
        )
        results = backtester.run()
        
        # 2. Display Results
        if results:
            backtester.display_summary(results)
            backtester.plot_results()
            
            if not backtester.trades_df.empty and 'PnL' in backtester.trades_df.columns:
                print("\n--- Final Trade Log ---")
                completed_trades = backtester.trades_df.dropna(subset=['PnL'])
                print(completed_trades[['Ticker', 'EntryDate', 'ExitDate', 'PnL', 'EntryPrice', 'ExitPrice', 'Reason']].round(2))