# config.py - Shared configuration constants for the degen_package

# File paths
DATABASE_FILE = 'bets_database.db'
MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'
LOG_FILE = 'degen_package.log'

# API keys (replace with your actual keys)
ODDS_API_KEY = '268d0312628e0408905fc746b0d1c815'  # Replace with your Odds API key
API_TIMEOUT = 10  # Seconds for API request timeout

# Package versioning
PACKAGE_VERSION = '1.0.0'

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE_MODE = 'a'  # Append mode

# Sports betting constants
MIN_BETS_FOR_TRAINING = 10  # Minimum bets required to train the model
SLEEP_BETWEEN_CHECKS = 1800  # 30 minutes in seconds
SLEEP_DURING_WAIT = 3600  # 1 hour in seconds
WAIT_AFTER_BET_HOURS = 2  # Hours to wait after bet commencement

# XRP Accumulator constants
CHECK_INTERVAL = 60  # 1 minute
MIN_USD_BALANCE = 1.00
TREND_WINDOW_SHORT = 5  # 5 minutes
TREND_WINDOW_LONG = 60  # 1 hour
SMA_WINDOW = 5  # 5 periods
RSI_PERIOD = 14  # 14 periods
BUY_RSI_THRESHOLD = 30  # Buy when RSI < 30 (oversold)
SELL_RSI_THRESHOLD = 70  # Sell when RSI > 70 (overbought)
PROFIT_MARGIN = 0.05  # 5% profit target
XRPL_RPC_URL = "https://s1.ripple.com:51234/"  # Public XRP Ledger server

# Required Python packages
REQUIRED_PACKAGES = [
    'requests', 'sqlite3', 'pandas', 'numpy', 'sklearn', 'colorama', 'joblib',
    'xrpl-py', 'coinbase-advanced-py'
]