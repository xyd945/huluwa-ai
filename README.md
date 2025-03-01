# AI Trading Agent

An advanced cryptocurrency trading system powered by machine learning and real-time market data.

## Project Overview

This AI Trading Agent collects data from multiple cryptocurrency exchanges, processes it, applies machine learning models to generate trading signals, and can execute trades based on those signals. Key features include:

- **Multi-Exchange Support**: Connect to Binance, BitMEX, and Deribit
- **Comprehensive Data Collection**: 
  - Liquidation data
  - Funding rates
  - Open interest
  - New token launches
  - Real-time market transactions
- **Advanced Feature Engineering**: Technical indicators and custom features
- **AI-Powered Trading Signals**: LSTM and other ML models
- **Risk Management**: Position sizing, stop-loss, and take-profit controls
- **Configurable Parameters**: Easily adjust all components through a YAML config file

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-trading-agent.git
   cd ai-trading-agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create required directories**:
   ```bash
   mkdir -p logs data models
   ```

## Configuration

The system is configured through `config/settings.yaml`. Before running, you need to:

1. Set your exchange API keys (if using live data):
   ```yaml
   exchanges:
     binance:
       api_key: "YOUR_API_KEY"
       api_secret: "YOUR_API_SECRET"
       testnet: true  # Set to false for production
   ```

2. Configure which data collectors to enable:
   ```yaml
   data_collection:
     liquidation:
       enabled: true
       interval: 60  # seconds
       exchanges: ["binance", "bitmex"]
     # ... other collectors
   ```

3. Configure AI model parameters:
   ```yaml
   ai_module:
     model_type: "lstm"  # Options: lstm, xgboost, transformer
     training:
       batch_size: 64
       epochs: 100
       test_size: 0.2
   ```

4. Enable/disable live trading:
   ```yaml
   execution:
     enabled: false  # Set to true for live trading
     max_position_size: 1000  # in USD
     stop_loss_pct: 2.0
     take_profit_pct: 5.0
   ```

## Running the Application

### Start the Trading Agent

Run the main application:

```bash
python main.py
```

This will:
1. Initialize all configured data collectors
2. Start collecting real-time data
3. Process the data into features
4. Train or load ML models (if configured)
5. Generate trading signals
6. Execute trades (if enabled in config)

### Common Command Line Options

```bash
# Use a custom config file
python main.py --config custom_config.yaml

# Run in backtesting mode (if implemented)
python main.py --backtest --start-date 2023-01-01 --end-date 2023-12-31

# Run in debug mode with verbose logging
python main.py --debug
```

## Project Structure

- `config/`: Configuration files
- `data_ingestion/`: Data collection modules
- `data_processing/`: Data cleaning and feature engineering
- `ai_module/`: Machine learning model training and prediction
- `execution/`: Order management and risk control
- `utils/`: Utility functions (logging, database)
- `main.py`: Application entry point

## Logging

Logs are stored in the `logs/` directory. You can adjust the logging level in the config file:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "logs/trading.log"
  rotation: "1 day"
  retention: "30 days"
```

## Database

By default, the application uses SQLite for data storage. You can configure other database types in the settings:

```yaml
database:
  type: "sqlite"  # Options: sqlite, postgresql, mongodb
  path: "data/trading.db"  # For sqlite
  # Additional settings for other DB types
```

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves significant risk and you should never trade with money you cannot afford to lose. Always do your own research before making any investment decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# File Structure for the AI Trading Agent

I've analyzed the codebase and created a complete file structure for your README. This structure provides a clear overview of the project organization:

```README.md
ai-trading-agent/
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
├── config/
│   ├── settings.yaml         # Main configuration (gitignored)
│   └── settings.yaml.example # Example configuration template
├── data/                     # Data storage directory
│   └── trading.db            # SQLite database (generated)
├── logs/                     # Log files directory
│   └── trading.log           # Main log file (generated)
├── models/                   # ML model storage
│   └── lstm_model.keras      # Trained LSTM model (generated)
├── data_ingestion/           # Data collection modules
│   ├── __init__.py
│   ├── exchange_connector.py # Base connector for exchanges
│   ├── liquidation_collector.py
│   ├── funding_rate_collector.py
│   ├── open_interest_collector.py
│   ├── token_launches_collector.py
│   └── transactions_collector.py
├── data_processing/          # Data processing modules
│   ├── __init__.py
│   ├── data_cleaner.py       # Data cleaning utilities
│   └── feature_engineering.py # Feature creation from raw data
├── ai_module/                # AI and ML components
│   ├── __init__.py
│   ├── model_trainer.py      # Trains ML models on processed data
│   └── prediction_engine.py  # Generates predictions/signals
├── execution/                # Trade execution modules
│   ├── __init__.py
│   └── order_manager.py      # Handles order creation and management
└── utils/                    # Utility modules
    ├── __init__.py
    ├── logger.py             # Logging configuration
    └── db_handler.py         # Database operations
```

## Data Flow

The application follows this data flow:

1. **Data Collection**: Modules in `data_ingestion/` collect real-time data from exchanges
2. **Data Processing**: Raw data is cleaned and transformed into features
3. **AI Processing**: ML models analyze processed data to generate trading signals
4. **Execution**: If enabled, trades are executed based on signals

## Key Components

- **AITradingAgent**: Central orchestrator that initializes and manages all components
- **DatabaseHandler**: Stores collected data in SQLite (or other configured DB)
- **Data Collectors**: Separate modules for different data types (liquidations, funding rates, etc.)
- **Model Trainer**: Trains LSTM or other ML models on processed data
- **Prediction Engine**: Uses trained models to generate trading signals
- **Order Manager**: Executes trades based on signals and risk parameters

The modular design allows each component to be developed, tested, and enhanced independently.

