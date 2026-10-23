# Moneyball - AI Investment Discovery System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


Finding undervalued assets before they break out, using the Moneyball philosophy applied to crypto and small-cap stocks.

## Overview

In the 2003 film *Moneyball*, the Oakland A's revolutionized baseball by identifying undervalued players using non-obvious metrics. This project applies the same philosophy to investing: traditional metrics (P/E, momentum, TVL) are lagging indicators. The real alpha lies in hidden signals:

- Whale accumulation patterns before price moves
- Exchange netflow divergence (accumulation not yet reflected in price)
- Developer activity surges in crypto projects
- Insider buying in small-cap stocks
- Fundamental anomalies (high profitability, low asset growth)

## Features

- **Data Pipeline**: CoinGecko (crypto), yfinance (stocks), Dune Analytics, Reddit, GitHub
- **Feature Engineering**: 20+ hidden signals, technical indicators, regime detection (HMM)
- **ML Models**: XGBoost + LightGBM ensemble, walk-forward validation, SHAP explainability
- **Multi-Agent AI**: Orchestrator, Quant, On-Chain, Sentiment, CIO agents
- **Streamlit Dashboard**: Scout, Backtest Simulator, Audit Logs

## Requirements

- Python 3.11+
- API Keys: CoinGecko (required), Gemini (required for AI memos), Dune/Reddit/GitHub (optional)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/moneyball.git
cd moneyball

python -m venv venv
# Windows: venv\Scripts\activate

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Linux/Mac: source venv/bin/activate

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


pip install -r requirements.txt
```

**Note:** ta-lib may require system installation (macOS: `brew install ta-lib`, Ubuntu: `sudo apt-get install ta-lib`).

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`:
   ```
   COINGECKO_API_KEY=your_key
   GEMINI_API_KEY=your_key
   DUNE_API_KEY=your_key          # optional
   REDDIT_CLIENT_ID=your_id       # optional
   REDDIT_CLIENT_SECRET=your_key  # optional
   GITHUB_TOKEN=your_token        # optional
   ```

3. Edit `config.yaml` for preferences (asset universe, model params, etc.)

## Quick Start

```bash
# 1. Fetch data (15-20 min first run)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python -m src.fetch.crypto
python -m src.fetch.stocks

# 2. Engineer features

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python -m src.features.pipeline

# 3. Train model

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python -m src.models.train

# 4. Generate recommendations

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python -m src.models.predict

# 5. Launch dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

streamlit run src/dashboard/app.py
```

Open http://localhost:8501 (or 8502 if 8501 is in use).

## Project Structure

```
moneyball/
├── src/
│   ├── fetch/          # Data acquisition (crypto, stocks, sentiment)
│   ├── features/       # Feature engineering & regime detection
│   ├── models/         # Training, prediction, backtesting
│   ├── agents/         # Multi-agent AI system
│   ├── dashboard/      # Streamlit UI
│   ├── audit_logger.py
│   └── config.py
├── data/raw/           # Raw data (Parquet)
├── data/features/      # Engineered features
├── models/             # Trained models
├── logs/               # Audit logs
├── config.yaml
└── requirements.txt
```

## Run from Any Directory

The dashboard adds the project root to `sys.path` automatically. To run other modules from any directory:

```bash
# Windows PowerShell

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

$env:PYTHONPATH = "s:\path\to\moneyball"
python -m src.models.predict

# Linux/Mac

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-blue?style=flat)](https://github.com/Amrit-04/moneyball)
[![Machine Learning](https://img.shields.io/badge/ML-Signal%20Screening-orange?style=flat)](https://github.com/Amrit-04/moneyball)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

export PYTHONPATH=/path/to/moneyball
python -m src.models.predict
```

Or run from the project directory:
```bash
cd moneyball
python -m src.models.predict
```

## System Test

Verify setup without data or trained models:

```bash
python test_system.py
```

## Performance Targets

- Model ROC-AUC: > 0.65
- Backtest Sharpe: > 1.0
- Memo generation: < 15 seconds

## License

MIT

## Disclaimer

For educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Do your own research and consult a qualified advisor before investing.
