# Stock Price Prediction System

An advanced stock price prediction system that uses machine learning to generate trading signals with enhanced risk management and position sizing.

## Features

- Multi-factor stock data analysis
- XGBoost-based prediction model
- Dynamic position sizing based on confidence and risk
- Enhanced risk management with multiple calculation methods
- Realistic trade simulation with stop-loss and take-profit levels
- Performance analysis with risk-adjusted metrics
- Data caching system for efficient API usage
- Support for multiple stocks and time periods

## Requirements

- Python 3.8+
- Required packages (see requirements.txt)
- Alpha Vantage API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Alpha Vantage API key:
   - Get an API key from [Alpha Vantage](https://www.alphavantage.co/)
   - Add your API key to the configuration

## Usage

Run the main script:
```bash
python simple_model.py/gemini_realistic.py
```

## Configuration

The system can be configured through the following parameters:
- `train_start_date`: Start date for training data
- `train_end_date`: End date for training data
- `test_start_date`: Start date for testing data
- `test_end_date`: End date for testing data
- `forward_period`: Number of days to predict ahead
- `min_confidence`: Minimum confidence threshold for signals
- `use_cache`: Whether to use cached data
- `force_refresh`: Whether to force refresh data

## Project Structure

- `simple_model.py/`: Main code directory
  - `gemini_realistic.py`: Main prediction system
  - `cache/`: Directory for cached data
  - `models/`: Directory for saved models
  - `data/`: Directory for processed data

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 