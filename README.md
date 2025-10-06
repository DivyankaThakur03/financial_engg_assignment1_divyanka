# MSAI 451 – Programming Assignment 1: NVDA Daily Direction Prediction

## Overview: Building an Active-Management Classifier

This project constructs a predictive pipeline to classify the next-day price direction (Up/Down) for NVIDIA Corporation (NVDA) using only historical OHLCV data. The core goal is to test the hypothesis: Can carefully engineered technical features provide a reliable, non-random edge in predicting short-term stock movement?

This repository contains the complete codebase for data preparation, feature engineering, time-series cross-validation, hyperparameter tuning (using XGBoost), and rigorous performance evaluation.

### Asset and Target

- **Asset:** NVIDIA Corporation (NVDA)
- **Target:** Next-day log-return sign \(\text{if } r_{t+1} > 0 \text{, else } 0\)

### Key Technologies

| Category | Tools |
|----------|-------|
| Data/Features | Python, Polars (for high-speed data manipulation) |
| Modeling | XGBoost, Scikit-learn |
| Evaluation | Scikit-learn's TimeSeriesSplit, Matplotlib/Seaborn |

## For Users: Getting Started

This section guides you through running the predictive pipeline to generate the results and diagnostic plots.

### Prerequisites

You need a Python environment installed. The required packages are:

| Package | Purpose |
|---------|---------|
| polars | Fast data loading and feature engineering |
| scikit-learn | Cross-validation and model selection |
| xgboost | The classification model |
| matplotlib | Plotting diagnostics (ROC, Confusion Matrix) |

To install the dependencies:

`pip install polars scikit-learn xgboost matplotlib seaborn`


### Running the Code

1. **Clone the Repository:**

`git clone <your_repo_url>.git`
`cd assignment_1`


2. **Execute the Main Script:**
`python3 assignment1_divya.py`


**Expected Output:** The script will print console diagnostics (CV scores, best hyperparameters, final metrics) and save three key diagnostic figures to the repository root:
- `Figure_1.png` (ROC Curve)
- `Figure_2.png` (Confusion Matrix)
- `Figure_3.png` (Feature Correlation Heatmap)

## 🛠️ For Developers: Building and Testing

This section provides details for those who want to examine, modify, and build the code.

### Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `451_pa1_jump_start_v001.py` | Main execution script (end-to-end pipeline: data → features → CV → model → diagnostics) |
| `msds_getdata_yfinance_nvdl.csv` | Input data file (NVDA OHLCV) |
| `report.pdf` | The detailed written analysis and conclusions |

### Feature Engineering Overview 

The model's features are designed to prevent look-ahead bias by using only data strictly prior to the prediction date. Key feature families include:

- **Lags:** Previous day's Close, HML (High - Low), OMC (Open - Close), and Volume (e.g., `CloseLag1..3`)
- **Momentum:** Exponential Moving Averages (`CloseEMA2`, `CloseEMA4`, `CloseEMA8`) computed from lagged data
- **Spreads:** Intraday range metrics (HML, OMC)

### Development Notes

The entire pipeline is built around Polars for efficient data processing, which is crucial when dealing with large time series datasets and extensive feature engineering.


## Use of AI Tools

AI assistance (ChatGPT/GPT-4) was primarily used for chceking the grammar and the word count of the report. All core  choices, feature design, and result interpretations were made by the author.
