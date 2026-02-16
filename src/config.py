"""Configuration management for Moneyball system.

Loads configuration from config.yaml and environment variables.
Environment variables take precedence over config file values.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for Moneyball system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. Defaults to project root.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Determine project root and config path
        if config_path is None:
            self.project_root = Path(__file__).parent.parent
            config_path = self.project_root / "config.yaml"
        else:
            config_path = Path(config_path)
            self.project_root = config_path.parent
            
        # Load YAML configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Validate required configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate that required configuration values are present."""
        required_sections = ['data', 'logging', 'data_fetching', 'features', 
                           'model', 'backtest', 'agents', 'dashboard']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
                
        # Validate API keys (warn if missing, don't fail)
        api_keys = {
            'COINGECKO_API_KEY': 'CoinGecko',
            'DUNE_API_KEY': 'Dune Analytics',
            'REDDIT_CLIENT_ID': 'Reddit',
            'REDDIT_CLIENT_SECRET': 'Reddit',
            'GITHUB_TOKEN': 'GitHub',
            'GEMINI_API_KEY': 'Google Gemini'
        }
        
        missing_keys = []
        for key, service in api_keys.items():
            if not os.getenv(key):
                missing_keys.append(f"{service} ({key})")
                
        if missing_keys:
            print(f"Warning: Missing API keys for: {', '.join(missing_keys)}")
            print("   Some features may not work. Add keys to .env file.")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., 'data.raw_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('model.breakout_threshold')
            0.15
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    # Data paths
    @property
    def raw_data_dir(self) -> Path:
        """Path to raw data directory."""
        return self.project_root / self.get('data.raw_dir', 'data/raw')
    
    @property
    def features_dir(self) -> Path:
        """Path to features directory."""
        return self.project_root / self.get('data.features_dir', 'data/features')
    
    @property
    def models_dir(self) -> Path:
        """Path to models directory."""
        path = self.project_root / self.get('data.models_dir', 'models')
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def log_dir(self) -> Path:
        """Path to logs directory."""
        return self.project_root / self.get('logging.log_dir', 'logs')
    
    # API Keys
    @property
    def coingecko_api_key(self) -> Optional[str]:
        """CoinGecko API key."""
        return os.getenv('COINGECKO_API_KEY')
    
    @property
    def dune_api_key(self) -> Optional[str]:
        """Dune Analytics API key."""
        return os.getenv('DUNE_API_KEY')
    
    @property
    def reddit_client_id(self) -> Optional[str]:
        """Reddit client ID."""
        return os.getenv('REDDIT_CLIENT_ID')
    
    @property
    def reddit_client_secret(self) -> Optional[str]:
        """Reddit client secret."""
        return os.getenv('REDDIT_CLIENT_SECRET')
    
    @property
    def github_token(self) -> Optional[str]:
        """GitHub personal access token."""
        return os.getenv('GITHUB_TOKEN')
    
    @property
    def gemini_api_key(self) -> Optional[str]:
        """Google Gemini API key."""
        return os.getenv('GEMINI_API_KEY')
    
    # Data fetching configuration
    @property
    def top_n_coins(self) -> int:
        """Number of top coins to fetch."""
        return self.get('data_fetching.crypto.top_n_coins', 100)
    
    @property
    def top_n_stocks(self) -> int:
        """Number of top stocks to fetch."""
        return self.get('data_fetching.stocks.top_n_stocks', 200)
    
    @property
    def stock_market(self) -> str:
        """Stock market to use (US, IN, or auto)."""
        return self.get('data_fetching.stocks.market', 'auto')
    
    @property
    def incremental_fetching(self) -> bool:
        """Whether to use incremental fetching."""
        return self.get('data_fetching.incremental', True)
    
    @property
    def lookback_days(self) -> int:
        """Number of days to fetch on first run."""
        return self.get('data_fetching.lookback_days', 180)
    
    # Feature engineering
    @property
    def forward_fill_days(self) -> int:
        """Maximum days to forward-fill missing data."""
        return self.get('features.forward_fill_days', 7)
    
    @property
    def winsorize_percentiles(self) -> tuple[int, int]:
        """Percentiles for winsorization."""
        percentiles = self.get('features.winsorize_percentiles', [1, 99])
        return tuple(percentiles)
    
    @property
    def zscore_window_days(self) -> int:
        """Window size for z-score normalization."""
        return self.get('features.zscore_window_days', 90)
    
    @property
    def regime_n_states(self) -> int:
        """Number of HMM states for regime detection."""
        return self.get('features.regime.n_states', 3)
    
    # Model configuration
    @property
    def breakout_threshold(self) -> float:
        """Breakout threshold (e.g., 0.15 for 15%)."""
        return self.get('model.breakout_threshold', 0.15)
    
    @property
    def breakout_horizon_days(self) -> int:
        """Breakout prediction horizon in days."""
        return self.get('model.breakout_horizon_days', 30)
    
    @property
    def train_months(self) -> int:
        """Training window size in months."""
        return self.get('model.train_months', 6)
    
    @property
    def test_months(self) -> int:
        """Test window size in months."""
        return self.get('model.test_months', 1)
    
    @property
    def min_roc_auc(self) -> float:
        """Minimum acceptable ROC-AUC."""
        return self.get('model.min_roc_auc', 0.65)
    
    @property
    def optuna_trials(self) -> int:
        """Number of Optuna hyperparameter tuning trials."""
        return self.get('model.optuna_trials', 50)
    
    # Agent configuration
    @property
    def llm_provider(self) -> str:
        """LLM provider (gemini or langchain)."""
        return self.get('agents.llm.provider', 'gemini')
    
    @property
    def llm_model(self) -> str:
        """LLM model name."""
        return self.get('agents.llm.model', 'gemini-1.5-flash')
    
    @property
    def llm_temperature(self) -> float:
        """LLM temperature."""
        return self.get('agents.llm.temperature', 0.3)
    
    @property
    def agent_timeout_seconds(self) -> int:
        """Agent timeout in seconds."""
        return self.get('agents.timeout_seconds', 15)
    
    @property
    def cache_ttl_hours(self) -> int:
        """Cache TTL in hours."""
        return self.get('agents.cache_ttl_hours', 1)
    
    # Dashboard configuration
    @property
    def dashboard_title(self) -> str:
        """Dashboard title."""
        return self.get('dashboard.title', 'Moneyball - AI Investment Discovery')
    
    @property
    def default_universe(self) -> list[str]:
        """Default asset universe."""
        return self.get('dashboard.default_universe', ['Crypto'])
    
    @property
    def default_top_n(self) -> int:
        """Default number of top assets to display."""
        return self.get('dashboard.default_top_n', 20)
    
    # Validation thresholds
    @property
    def min_history_months(self) -> int:
        """Minimum history required in months."""
        return self.get('validation.min_history_months', 6)
    
    @property
    def min_crypto_assets(self) -> int:
        """Minimum number of crypto assets required."""
        return self.get('validation.min_crypto_assets', 50)
    
    @property
    def min_stock_assets(self) -> int:
        """Minimum number of stock assets required."""
        return self.get('validation.min_stock_assets', 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return self.config.copy()


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance.
    
    Args:
        config_path: Path to config.yaml file (only used on first call)
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        New configuration instance
    """
    global _config
    _config = Config(config_path)
    return _config
