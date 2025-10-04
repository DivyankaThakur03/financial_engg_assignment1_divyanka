# MSAI 451 – Programming Assignment 1: Predicting AAPL Daily and Weekly Returns Using Technical Indicators

## Overview

This project builds a simple, time-aware machine-learning pipeline to predict short-term price direction for Apple Inc. (AAPL) using historical daily OHLCV data. Two horizons are evaluated:
*   **t+1**: next-day direction
*   **t+5**: next-week direction (5 trading days ahead)

The goal is to demonstrate careful preprocessing, leakage-free splitting, basic feature engineering, simple baseline models, and honest evaluation and plots.

## Dataset

*   **Source**: Yahoo Finance (provided CSV: `msds_getdata_yfinance_aapl.csv`).
*   **Date range in the file**: 2000–2025.
*   **Columns expected in CSV**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, ( `Dividends`, `Stock Splits` are optional).
*   The script auto-creates `Adj Close = Close` if `Adj Close` is missing.
*   Dates are cleaned and parsed; rows with unparseable dates are dropped before sorting.

## Project Files (repo root)

*   `assignment1_divya.py` — main script: data load, feature build, time-based split, training, evaluation, plots
*   `msds_getdata_yfinance_aapl.csv` — AAPL OHLCV input data
*   `equity_curve_test.png` — equity curve for the test set (t+1 and t+5 runs)
*   `feature_importances.png` — top-10 Random Forest importances (from t+1 run)
*   `report.pdf` — written report 

## Feature Engineering

**Momentum/trend:**
*   `MA5`, `MA20` (simple moving averages)
*   `EMA5`, `EMA12`, `EMA26` and `MACD` (EMA12 − EMA26)

**Volatility and bands:**
*   `Vol10` (10-day rolling std of returns)
*   Bollinger Band width (based on 20-day MA and std)

**Candlestick and lags:**
*   `HML` = High − Low
*   `OMC` = Open − Close
*   Lags for `Adj Close`, `HML`, `OMC`, `Volume` (1–3 days)

**Calendar:**
*   Day of week; `IsFriday` flag

**Targets:**
*   **t+1**: 1 if next-day return > 0, else 0
*   **t+5**: 1 if return over the next 5 trading days > 0, else 0

## Methodology

*   **Time-ordered split**: 80% train, 20% test (no shuffling).
*   **Tiny validation inside the training set**: last fold of a `TimeSeriesSplit` used only to tune the probability threshold for Logistic Regression.
*   **Models**:
    *   Logistic Regression (standardized features via `StandardScaler`)
    *   Random Forest (`n_estimators=500`, `max_depth=8`, `min_samples_leaf=5`)
*   **Metrics**: Accuracy, F1, ROC AUC; confusion matrices.
*   **Plots**: equity curve vs buy-and-hold (test set), RF feature importances.

## Results (from the provided AAPL run)

### t+1 (next day)
*   **Logistic Regression**: Accuracy ≈ 0.496, F1 ≈ 0.603, AUC ≈ 0.483
*   **Random Forest**: Accuracy ≈ 0.514, F1 ≈ 0.662, AUC ≈ 0.485

### t+5 (weekly)
*   **Logistic Regression**: Accuracy ≈ 0.548, F1 ≈ 0.704, AUC ≈ 0.470
*   **Random Forest**: Accuracy ≈ 0.501, F1 ≈ 0.562, AUC ≈ 0.468

### Interpretation
Results are close to chance by AUC, which is expected for short-horizon price direction with price-only technical features. The pipeline is clean and leakage-free, and the metrics honestly reflect the challenge.

## How to Run

1.  **Clone the repository:**
    ```
    git clone <your_repo_url>.git
    cd assignment_1
    ```
2.  **Install dependencies (example):**
    ```
    pip install numpy pandas scikit-learn matplotlib
    ```
3.  **Run:**
    ```
    python3 assignment1_divya.py
    ```
    Outputs generated in the repo directory:
    *   `equity_curve_test.png`
    *   `feature_importances.png`
    *   Console prints with dataset info, validation threshold, metrics, and confusion matrices

## Reproducibility and Design Notes

*   Time-aware split prevents look-ahead.
*   Threshold tuning uses only the training data via a final validation fold.
*   The same feature set is used for both t+1 and t+5; only the target definition changes.

## Use of AI Tools

AI assistance (ChatGPT/GPT-5) was used to:
*   Refining code structure.
*   Enhancing the grammar of the report.

All core modeling choices, feature engineering, and result interpretations were made by the author. The AI served as a tool for improving code and text quality, not for autonomous analysis.
