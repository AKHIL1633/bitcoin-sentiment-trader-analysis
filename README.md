# Trader Performance vs. Market Sentiment — Analysis

Explores the relationship between Hyperliquid trader performance and the Bitcoin
Fear & Greed Index: regime-level performance metrics, robustness checks (era-split,
lagged correlation, bootstrap CIs), a crowd-bias correctness backtest, and a
predictive model with feature importance.

**Start here → [REPORT.md](REPORT.md)** for the full write-up: methodology, charts, and strategy recommendations.

## Headline result

| Sentiment | Win Rate | Profit Factor | PnL per $ traded |
|---|---:|---:|---:|
| Extreme Fear | 78.4% | 2.8x | 92 bps |
| Fear | 86.7% | 5.5x | 59 bps |
| Neutral | 80.1% | 4.9x | 68 bps |
| Greed | 78.5% | 3.2x | 91 bps |
| **Extreme Greed** | **88.9%** | **10.3x** | **206 bps** |

Extreme Greed is this cohort's best-performing regime by every efficiency metric — the opposite of the "buy fear, sell greed" folk wisdom. Full detail, caveats, and a backtest of what that implies for position sizing are in [REPORT.md](REPORT.md).

## Pipeline

```mermaid
flowchart TD
    A1[("fear_greed_index.csv")]
    A2[("historical_data.csv")]

    A1 --> EXP["01-02: explore raw files"]
    A2 --> EXP

    A1 --> CLEAN["03: clean_merge.py\nfix timestamp bug, join on date"]
    A2 --> CLEAN
    CLEAN --> PARQ[("outputs/data/merged_trades.parquet")]

    PARQ --> METRICS["04: metrics.py\nregime performance, correlations"]
    METRICS --> DATA[("outputs/data/*.csv")]

    PARQ --> DEEP["06: deeper_patterns.py\naccounts, coins, risk"]
    DATA --> DEEP

    PARQ --> ADV["08: advanced_stats.py\nera-split, lag, bootstrap, sentiment beta"]
    DATA --> ADV
    ADV --> DATA

    PARQ --> BT["09: backtest.py\ncrowd-bias backtest"]

    PARQ --> ML["10: predictive_model.py\nrandom forest / logistic regression"]
    ML --> DATA

    PARQ --> PLOTS["05 / 07 / 11: plots*.py"]
    DATA --> PLOTS
    PLOTS --> FIGS[("outputs/figures/*.png")]

    DATA --> REPORT["REPORT.md"]
    FIGS --> REPORT
    BT --> REPORT
```

## Sample output

| | |
|---|---|
| ![Win rate by sentiment](outputs/figures/win_rate_by_sentiment.png) | ![Capital efficiency by sentiment](outputs/figures/pnl_efficiency_by_sentiment.png) |
| ![Was the crowd's bias correct](outputs/figures/crowd_bias_correctness.png) | ![Daily PnL vs sentiment over time](outputs/figures/daily_pnl_vs_sentiment_timeseries.png) |

All 14 charts are in [`outputs/figures/`](outputs/figures/), referenced inline throughout [REPORT.md](REPORT.md).

## Project layout

```
├── fear_greed_index.csv         raw sentiment data (given)
├── historical_data.csv          raw Hyperliquid trade data (given)
├── requirements.txt             Python dependencies
├── REPORT.md                    full findings + strategy recommendations
├── README.md                    this file
├── notebooks/                   analysis scripts, run in order
│   ├── 01_explore.py            initial look at both raw files
│   ├── 02_explore_time.py       diagnose the timestamp columns
│   ├── 03_clean_merge.py        clean, align timezones, merge -> outputs/data/merged_trades.parquet
│   ├── 04_metrics.py            core performance metrics by sentiment regime
│   ├── 05_plots.py              primary charts
│   ├── 06_deeper_patterns.py    per-account, per-coin, concentration & risk checks
│   ├── 07_plots_extra.py        remaining charts
│   ├── 08_advanced_stats.py     era split, lagged correlation, Kruskal-Wallis, bootstrap CIs, sentiment beta
│   ├── 09_backtest.py           crowd-bias correctness check + simple sizing/skip backtests
│   ├── 10_predictive_model.py   random forest / logistic regression, feature importance for win prediction
│   └── 11_plots_advanced.py     charts for 08-10
└── outputs/
    ├── data/                    cached CSV/parquet outputs from the scripts above
    └── figures/                 all PNG charts referenced in REPORT.md
```

## Reproducing

Run from the project root (the scripts read `fear_greed_index.csv` / `historical_data.csv` and write to `outputs/` using relative paths):

```bash
pip install -r requirements.txt
python notebooks/01_explore.py
python notebooks/02_explore_time.py
python notebooks/03_clean_merge.py
python notebooks/04_metrics.py
python notebooks/05_plots.py
python notebooks/06_deeper_patterns.py
python notebooks/07_plots_extra.py
python notebooks/08_advanced_stats.py
python notebooks/09_backtest.py
python notebooks/10_predictive_model.py
python notebooks/11_plots_advanced.py
```

Each script prints its findings to stdout and/or writes to `outputs/`.
