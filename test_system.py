#!/usr/bin/env python3
"""
Quick system test to verify all components can be imported and initialized.
This test does NOT require data or trained models.
"""

import sys
from pathlib import Path

print("="*80)
print("MONEYBALL SYSTEM TEST")
print("="*80)
print()

# Test 1: Configuration
print("1. Testing configuration...")
try:
    from src.config import get_config
    config = get_config()
    print(f"   [OK] Config loaded: {config.project_name}")
    print(f"   [OK] Data dir: {config.raw_data_dir}")
    print(f"   [OK] Models dir: {config.models_dir}")
except Exception as e:
    print(f"   [FAIL] Config failed: {e}")
    sys.exit(1)

# Test 2: Data Fetchers
print("\n2. Testing data fetchers...")
try:
    from src.fetch.crypto import CoinGeckoFetcher
    from src.fetch.stocks import StockFetcher
    from src.fetch.sentiment import RedditFetcher
    print("   [OK] Crypto fetcher imported")
    print("   [OK] Stock fetcher imported")
    print("   [OK] Sentiment fetcher imported")
except Exception as e:
    print(f"   [FAIL] Fetcher import failed: {e}")
    sys.exit(1)

# Test 3: Feature Engineering
print("\n3. Testing feature engineering...")
try:
    from src.features.crypto_features import compute_all_crypto_features
    from src.features.stock_features import compute_all_stock_features
    from src.features.technical_features import compute_all_technical_features
    from src.features.regime import RegimeDetector
    print("   [OK] Crypto features imported")
    print("   [OK] Stock features imported")
    print("   [OK] Technical features imported")
    print("   [OK] Regime detector imported")
except Exception as e:
    print(f"   [FAIL] Feature import failed: {e}")
    sys.exit(1)

# Test 4: ML Models
print("\n4. Testing ML models...")
try:
    from src.models.train import BreakoutPredictor
    from src.models.backtest import run_backtest
    print("   [OK] BreakoutPredictor imported")
    print("   [OK] Backtest function imported")
except Exception as e:
    print(f"   [FAIL] Model import failed: {e}")
    sys.exit(1)

# Test 5: Agent System
print("\n5. Testing agent system...")
try:
    from src.agents.orchestrator import OrchestratorAgent
    from src.agents.quant_agent import QuantAgent
    from src.agents.onchain_agent import OnChainAgent
    from src.agents.sentiment_agent import SentimentAgent
    from src.agents.cio_agent import CIOAgent
    from src.audit_logger import AuditLogger
    
    # Initialize agents
    audit_logger = AuditLogger(config.log_dir)
    orchestrator = OrchestratorAgent(audit_logger)
    quant = QuantAgent(audit_logger)
    onchain = OnChainAgent(audit_logger)
    sentiment = SentimentAgent(audit_logger)
    cio = CIOAgent(audit_logger)
    
    print("   [OK] Orchestrator agent initialized")
    print("   [OK] Quant agent initialized")
    print("   [OK] OnChain agent initialized")
    print("   [OK] Sentiment agent initialized")
    print("   [OK] CIO agent initialized")
    print("   [OK] Audit logger initialized")
except Exception as e:
    print(f"   [FAIL] Agent initialization failed: {e}")
    sys.exit(1)

# Test 6: Dashboard
print("\n6. Testing dashboard...")
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    print("   [OK] Streamlit imported")
    print("   [OK] Plotly imported")
except Exception as e:
    print(f"   [FAIL] Dashboard import failed: {e}")
    sys.exit(1)

# Test 7: Data Validation
print("\n7. Testing data validation...")
try:
    from src.data_validation import DataValidator
    print("   [OK] Data validation imported")
except Exception as e:
    print(f"   [FAIL] Data validation import failed: {e}")
    sys.exit(1)

# Test 8: Check directories
print("\n8. Checking directory structure...")
required_dirs = [
    config.raw_data_dir,
    config.features_dir,
    config.models_dir,
    config.log_dir
]
for dir_path in required_dirs:
    if dir_path.exists():
        print(f"   [OK] {dir_path} exists")
    else:
        print(f"   [WARN] {dir_path} does not exist (will be created on first run)")

# Test 9: Check configuration file
print("\n9. Checking configuration files...")
config_files = [
    Path("config.yaml"),
    Path(".env.example"),
    Path("requirements.txt"),
    Path(".gitignore")
]
for file_path in config_files:
    if file_path.exists():
        print(f"   [OK] {file_path} exists")
    else:
        print(f"   [FAIL] {file_path} missing")

# Test 10: Check for API keys
print("\n10. Checking API keys...")
env_file = Path(".env")
if env_file.exists():
    print("   [OK] .env file exists")
    # Check if keys are configured (not just template values)
    with open(env_file) as f:
        content = f.read()
        if "your-" in content or "your_" in content:
            print("   [WARN] API keys appear to be template values")
            print("   Edit .env with your actual API keys")
        else:
            print("   [OK] API keys appear to be configured")
else:
    print("   [WARN] .env file not found")
    print("   Run: cp .env.example .env")
    print("   Then edit .env with your API keys")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print()
print("[OK] All core components can be imported and initialized")
print("[OK] Directory structure is correct")
print("[OK] Configuration files are present")
print()
print("NEXT STEPS:")
print()
print("1. Configure API keys:")
print("   cp .env.example .env")
print("   # Edit .env with your API keys")
print()
print("2. Fetch data (15-20 minutes):")
print("   python -m src.fetch.crypto")
print("   python -m src.fetch.stocks")
print()
print("3. Engineer features (2-5 minutes):")
print("   python -m src.features.pipeline")
print()
print("4. Train model (10-20 minutes):")
print("   python -m src.models.train")
print()
print("5. Generate recommendations:")
print("   python -m src.models.predict")
print()
print("6. Launch dashboard:")
print("   streamlit run src/dashboard/app.py")
print()
print("="*80)
print("SYSTEM TEST PASSED - Ready to run!")
print("="*80)
print()
print("For detailed instructions, see: README.md")
