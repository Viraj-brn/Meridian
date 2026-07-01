# Meridian — Market-Cap Weighted Index Tracker

A production-quality equity index tracker that constructs a custom market-cap weighted index from 50 large-cap US stocks, computes daily index levels using the **divisor method**, performs quarterly rebalancing, and layers on comprehensive risk analytics including **PCA-based factor decomposition**.

Built to demonstrate the full spectrum of skills relevant to quantitative finance and financial data engineering roles.

![Dashboard](web/static/charts/performance.png)

---

## What It Does

### 1. Index Construction (MSCI-Style)
- **Universe**: 50 stocks across all 11 GICS sectors
- **Weighting**: Free-float adjusted market capitalization (Strategy pattern enables equal-weight and capped variants)
- **Divisor Method**: Index level = `Σ(P_i × Q_i × FFF_i) / Divisor`, ensuring continuity across rebalances
- **Quarterly Rebalancing**: Every 63 trading days, weights are recalculated and the divisor adjusted

### 2. Risk Analytics
- **Volatility**: Rolling (20-day, 60-day) and EWMA annualized volatility
- **Value at Risk**: Historical and parametric VaR at 95% confidence
- **Conditional VaR**: Expected Shortfall — the average loss in the worst 5% of days
- **Maximum Drawdown**: Largest peak-to-trough decline with dates
- **Covariance & Correlation**: Full N×N matrices with eigendecomposition

### 3. Factor Analysis (Simplified Barra Model)
- **PCA Decomposition**: Extract principal risk factors from the return covariance matrix
- **Variance Attribution**: "Factor 1 (Market) explains X% of total portfolio risk"
- **Factor Exposures**: OLS regression to compute each stock's beta to each factor

### 4. Performance Metrics
| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted return: `(R - Rf) / σ` |
| **Sortino Ratio** | Like Sharpe, but only penalizes downside volatility |
| **Tracking Error** | How closely the index tracks its benchmark (SPY) |
| **Information Ratio** | Active return per unit of active risk |
| **Calmar Ratio** | Annualized return / maximum drawdown |

### 5. SQL Analytics
All data stored in SQLite with MSCI-style queries:
- Top N constituents by weight
- Sector breakdown
- Rolling return rankings (window functions)
- Data quality gap detection (`LAG` + date arithmetic)
- Price anomaly detection (potential corporate actions)
- Rebalance candidate identification (`PERCENT_RANK`)

### 6. Web Dashboard (Flask)
- **Dashboard**: Key metrics, performance vs benchmark, sector allocation
- **Analytics**: Volatility, VaR, factor decomposition, correlation heatmap
- **Constituents**: Full holdings table, anomaly detection, data gaps
- **Methodology**: How the index is constructed and analyzed

---

## Architecture

```
meridian/
├── data/                    # Data layer
│   ├── fetcher.py           # yfinance download with caching
│   ├── cleaner.py           # Validation, forward-fill, outlier detection
│   └── corporate_actions.py # Split detection and adjustment
│
├── models/                  # Core domain models
│   ├── security.py          # Security dataclass
│   ├── weighting.py         # Strategy pattern: MarketCap, Equal, Capped
│   ├── index.py             # EquityIndex with divisor method
│   └── portfolio.py         # Portfolio with return computation
│
├── analytics/               # Quantitative analytics engine
│   ├── returns.py           # Simple, log, cumulative returns
│   ├── risk.py              # Volatility, VaR, CVaR, drawdown, covariance
│   ├── factor_analysis.py   # PCA decomposition, factor exposures
│   └── performance.py       # Sharpe, Sortino, tracking error, IR
│
├── database/                # Persistence layer
│   ├── schema.sql           # SQLite table definitions
│   ├── connection.py        # Repository pattern with context manager
│   └── queries.py           # Named SQL analytics queries
│
├── visualization/           # Chart generation
│   └── charts.py            # matplotlib dark-themed professional charts
│
├── web/                     # Flask web dashboard
│   ├── app.py               # Flask routes
│   ├── templates/           # Jinja2 HTML templates
│   └── static/css/          # Professional dark theme CSS
│
├── tests/                   # Unit tests (pytest)
│   ├── test_returns.py      # Return calculations
│   ├── test_risk.py         # Risk metrics
│   ├── test_index.py        # Index construction
│   └── test_data_quality.py # Data validation
│
├── main.py                  # CLI orchestration
├── config.py                # All constants and parameters
└── requirements.txt         # Dependencies
```

---

## Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `weighting.py` | Swap weighting schemes without modifying `EquityIndex` |
| **Repository** | `connection.py` | Abstract database operations behind a clean interface |
| **Factory** | `main.py` | Construct `Security` objects from database records |
| **Pipeline** | `main.py` | Fetch → Clean → Build → Analyze → Visualize |
| **Context Manager** | `DatabaseManager` | Safe connection handling with auto-commit/rollback |

---

## How to Run

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/yourusername/Meridian.git
cd Meridian
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Fetch data, build index, run analytics, generate charts
python main.py --all

# Then launch the web dashboard
python main.py --serve
```

The dashboard will be available at `http://localhost:5000`.

### Individual Steps

```bash
python main.py --fetch          # Download market data
python main.py --build-index    # Construct the index
python main.py --analytics      # Run analytics + generate charts
python main.py --serve          # Launch web dashboard
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Key Concepts Demonstrated

### The Divisor Method
```
Index(t) = Σ(P_i(t) × Q_i × FFF_i) / Divisor

Initial:   Divisor = Σ(P_i(0) × Q_i × FFF_i) / 1000
Rebalance: D_new = D_old × (New_Numerator / Old_Numerator)
```

The divisor absorbs all structural changes (additions, removals, rebalances) so the index level reflects **only price movements**, never artificial jumps.

### Free-Float Adjustment
```
Weight_i = (Price_i × Shares_i × FFF_i) / Σ(Price_j × Shares_j × FFF_j)
```

Free-float factor excludes shares not available for public trading (government, insider, strategic holdings). This is the standard used by MSCI, S&P, and FTSE.

### Factor Model
```
R_stock = α + β₁·F₁ + β₂·F₂ + ... + βₖ·Fₖ + ε
```

PCA extracts the dominant risk factors from the return covariance matrix. The first factor typically explains 40-60% of variance (the "market factor").

### Portfolio Risk
```
Portfolio Variance = w^T · Σ · w
```

Where `w` is the weight vector and `Σ` is the covariance matrix. This is the fundamental equation of Modern Portfolio Theory.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Core** | Python 3.10+ | Application logic |
| **Numerical** | NumPy, Pandas, SciPy | Matrix ops, time series |
| **ML** | scikit-learn | PCA for factor analysis |
| **Database** | SQLite | Persistent storage, SQL analytics |
| **Charts** | matplotlib, seaborn | Professional visualizations |
| **Web** | Flask | Dashboard (no JS frameworks) |
| **Data** | yfinance | Market data source |
| **Tests** | pytest | Unit testing |

---

## License

MIT
