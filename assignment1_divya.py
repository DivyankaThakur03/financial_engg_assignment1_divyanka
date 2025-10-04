#!/usr/bin/env python3

# DEPENDENCIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)


# CONFIG 
CSV_PATH = "msds_getdata_yfinance_aapl.csv"  
SEED = 42


# 1) DATA LOADING (CSV)
def load_from_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read raw; do robust Date cleaning before parsing
    df = pd.read_csv(csv_path)

    if "Date" not in df.columns:
        raise ValueError("CSV missing 'Date' column")

    # Clean common timezone/offset suffixes that break parsing (e.g., ' UTC-05:00', ' +05:30', ' GMT-03:00')
    date_str = (
        df["Date"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s*(UTC|GMT)[+-]\d{2}:\d{2}$", "", regex=True)  
        .str.replace(r"\s*[+-]\d{2}:\d{2}$", "", regex=True)           
    )

    # Parse; drop tz if any slipped through
    parsed = pd.to_datetime(date_str, errors="coerce", infer_datetime_format=True)
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)

    df["Date"] = parsed

    # Drop rows that still failed to parse
    before = len(df)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    after = len(df)
    print(f"[info] Date dtype: {df['Date'].dtype}")
    if before != after:
        print(f"[info] Dropped {before - after} rows with invalid Date values (after cleaning).")

    # Auto-create Adj Close if not present
    if "Adj Close" not in df.columns:
        if "Close" not in df.columns:
            raise ValueError("CSV missing 'Close' column required to create 'Adj Close'.")
        df["Adj Close"] = df["Close"]

    required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Sort by Date
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# 2) FEATURE ENGINEERING / TARGET
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic return & vol
    df["Return"] = df["Adj Close"].pct_change()
    df["Vol10"] = df["Return"].rolling(10).std()

    # MAs (simple) + RSI
    df["MA5"]  = df["Adj Close"].rolling(5).mean()
    df["MA20"] = df["Adj Close"].rolling(20).mean()
    df["RSI14"] = compute_rsi(df["Adj Close"], 14)

    # Candlesticks
    df["HML"] = df["High"] - df["Low"]
    df["OMC"] = df["Open"] - df["Close"]

    # EMAs / MACD
    df["EMA12"] = df["Adj Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = df["EMA12"] - df["EMA26"]
    df["EMA5"]  = df["Adj Close"].ewm(span=5, adjust=False).mean()

    # Bollinger (20, 2)
    ma20 = df["Adj Close"].rolling(20).mean()
    sd20 = df["Adj Close"].rolling(20).std()
    df["BB_Upper"] = ma20 + 2 * sd20
    df["BB_Lower"] = ma20 - 2 * sd20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / ma20

    # Lags
    for k in [1, 2, 3]:
        df[f"AdjClose_L{k}"] = df["Adj Close"].shift(k)
        df[f"HML_L{k}"]      = df["HML"].shift(k)
        df[f"OMC_L{k}"]      = df["OMC"].shift(k)
        df[f"Vol_L{k}"]      = df["Volume"].shift(k)

    # Calendar feature
    df["DOW"] = df["Date"].dt.dayofweek  # 0=Mon
    df["IsFri"] = (df["DOW"] == 4).astype(int)

    # Target: next-day direction
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)

    return df

# 3) SPLIT, TRAIN, THRESHOLD TUNE, PLOTS
def build_models():
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )
    return logit, rf

def compute_metrics(y_true, y_pred, y_proba):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        out["roc_auc"] = float("nan")
    return out

def time_based_split(df: pd.DataFrame, feature_cols, target_col="Target", split_ratio=0.8):
    m = df.dropna(subset=feature_cols + [target_col]).copy()
    split_idx = int(len(m) * split_ratio)
    train, test = m.iloc[:split_idx], m.iloc[split_idx:]
    X_train, y_train = train[feature_cols].values, train[target_col].values
    X_test,  y_test  = test[feature_cols].values,  test[target_col].values
    return train, test, X_train, y_train, X_test, y_test

def pick_threshold_time_split(model, X, y, n_splits=5):
    """
    Use the last fold of a TimeSeriesSplit as a small validation set to pick the best probability threshold.
    Returns chosen threshold and a dict of validation metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    tr_idx, va_idx = list(tscv.split(X, y))[-1]  # last fold
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]

    grid = np.linspace(0.3, 0.7, 41)  # 0.30..0.70 step 0.01
    best_t, best_acc = 0.5, -1
    for t in grid:
        pred = (p >= t).astype(int)
        acc = accuracy_score(yva, pred)
        if acc > best_acc:
            best_acc, best_t = acc, t

    pred_best = (p >= best_t).astype(int)
    val_metrics = compute_metrics(yva, pred_best, p)
    val_metrics["threshold"] = best_t
    return best_t, val_metrics

def plot_equity_curve(dates, test_returns, proba, threshold=0.55, out_path="equity_curve_test.png"):
    signal = (proba >= threshold).astype(int)
    strat_ret = signal * test_returns
    eq_curve = (1 + strat_ret).cumprod()
    bh_curve = (1 + test_returns).cumprod()

    plt.figure()
    plt.plot(dates, bh_curve, label="Buy & Hold")
    plt.plot(dates, eq_curve, label=f"Logit Strat (p >= {threshold:.2f})")
    plt.title("Equity Curve on Test Set")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def plot_feature_importance(rf_model, feature_cols, out_path="feature_importances.png"):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    plt.figure()
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), [feature_cols[i] for i in idx], rotation=45, ha="right")
    plt.title("RandomForest Top-10 Feature Importances")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return out_path, idx, importances


# MAIN
def main():
    # Load & feature-build
    df = load_from_csv(CSV_PATH)
    print(f"Loaded CSV with {len(df):,} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")
    df_feat = build_features(df)

    feature_cols = [
        # base techs
        "MA5","MA20","Vol10","HML","OMC","RSI14",
        "AdjClose_L1","AdjClose_L2","AdjClose_L3",
        "HML_L1","HML_L2","HML_L3","OMC_L1","OMC_L2","OMC_L3",
        "Vol_L1","Vol_L2","Vol_L3",
        # extra cheap alphas
        "EMA5","EMA12","EMA26","MACD","BB_Width",
        "IsFri"
    ]

    # BASELINE: next-day (t+1)
    train, test, Xtr, ytr, Xte, yte = time_based_split(df_feat, feature_cols, "Target", 0.8)
    print(f"Train size: {len(train):,} | Test size: {len(test):,}")
    print(f"Train date range: {train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"Test  date range: {test['Date'].min().date()}  to {test['Date'].max().date()}")
    print(f"Train class balance (mean Target): {train['Target'].mean():.3f}")
    print(f"Test  class balance (mean Target): {test['Target'].mean():.3f}")

    logit, rf = build_models()

    # Threshold tuning on TRAIN (tiny validation via last TimeSeriesSplit fold)
    thr, val_metrics = pick_threshold_time_split(logit, Xtr, ytr, n_splits=5)
    print("\n=== VALIDATION (threshold tuning for logit, t+1) ===")
    print(f"Chosen threshold: {val_metrics['threshold']:.2f}")
    print(f"Val acc: {val_metrics['accuracy']:.3f} | Val f1: {val_metrics['f1']:.3f} | Val AUC: {val_metrics['roc_auc']:.3f}")

    # Retrain on full train, evaluate on test
    logit.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)

    p_logit = logit.predict_proba(Xte)[:, 1]
    yhat_logit = (p_logit >= thr).astype(int)
    m_logit = compute_metrics(yte, yhat_logit, p_logit)

    p_rf = rf.predict_proba(Xte)[:, 1]
    yhat_rf = (p_rf >= 0.5).astype(int)
    m_rf = compute_metrics(yte, yhat_rf, p_rf)

    # Plots
    eq_path = plot_equity_curve(test["Date"].values, test["Return"].values, p_logit, threshold=thr)
    fi_path, idx, importances = plot_feature_importance(rf, feature_cols)

    print("\nTop-10 RF feature importances (t+1):")
    for rank, i in enumerate(idx, 1):
        print(f"{rank:>2}. {feature_cols[i]:<15}  {importances[i]:.4f}")

    print("\n=== TEST METRICS (t+1) ===")
    print(f"Logit (thr={thr:.2f}) -> acc: {m_logit['accuracy']:.3f} | f1: {m_logit['f1']:.3f} | AUC: {m_logit['roc_auc']:.3f}")
    print(f"Random Forest       -> acc: {m_rf['accuracy']:.3f} | f1: {m_rf['f1']:.3f} | AUC: {m_rf['roc_auc']:.3f}")
    print("\nArtifacts written (t+1):")
    print(f"  - {eq_path}")
    print(f"  - {fi_path}")

    # SECOND RUN: t+5 (weekly direction)
    print("\n" + "="*70)
    print("Running t+5 (weekly) variant...")
    df_feat_t5 = df_feat.copy()
    # 5-day forward return & direction
    df_feat_t5["Ret_t5"] = df_feat_t5["Adj Close"].pct_change(periods=5)
    df_feat_t5["Target_t5"] = (df_feat_t5["Ret_t5"].shift(-5) > 0).astype(int)

    # split with t+5 target
    train5, test5, Xtr5, ytr5, Xte5, yte5 = time_based_split(df_feat_t5, feature_cols, target_col="Target_t5", split_ratio=0.8)
    print(f"Train size (t+5): {len(train5):,} | Test size (t+5): {len(test5):,}")
    print(f"Train date range (t+5): {train5['Date'].min().date()} to {train5['Date'].max().date()}")
    print(f"Test  date range (t+5): {test5['Date'].min().date()}  to {test5['Date'].max().date()}")
    print(f"Train class balance (t+5): {train5['Target_t5'].mean():.3f}")
    print(f"Test  class balance (t+5): {test5['Target_t5'].mean():.3f}")

    # fresh models
    logit5, rf5 = build_models()

    # threshold tuning for t+5 on TRAIN only
    thr5, val_metrics5 = pick_threshold_time_split(logit5, Xtr5, ytr5, n_splits=5)
    print("\n=== VALIDATION (threshold tuning for logit, t+5) ===")
    print(f"Chosen threshold (t+5): {val_metrics5['threshold']:.2f}")
    print(f"Val acc (t+5): {val_metrics5['accuracy']:.3f} | Val f1 (t+5): {val_metrics5['f1']:.3f} | Val AUC (t+5): {val_metrics5['roc_auc']:.3f}")

    # retrain & evaluate on t+5 test
    logit5.fit(Xtr5, ytr5)
    rf5.fit(Xtr5, ytr5)

    p_logit5 = logit5.predict_proba(Xte5)[:, 1]
    yhat_logit5 = (p_logit5 >= thr5).astype(int)
    m_logit5 = compute_metrics(yte5, yhat_logit5, p_logit5)

    p_rf5 = rf5.predict_proba(Xte5)[:, 1]
    yhat_rf5 = (p_rf5 >= 0.5).astype(int)
    m_rf5 = compute_metrics(yte5, yhat_rf5, p_rf5)
    eq_path5 = plot_equity_curve(test5["Date"].values, test5["Return"].values, p_logit5, threshold=thr5, out_path="equity_curve_test_t5.png")

    print("\n=== TEST METRICS (t+5) ===")
    print(f"Logit (thr={thr5:.2f}) -> acc: {m_logit5['accuracy']:.3f} | f1: {m_logit5['f1']:.3f} | AUC: {m_logit5['roc_auc']:.3f}")
    print(f"Random Forest       -> acc: {m_rf5['accuracy']:.3f} | f1: {m_rf5['f1']:.3f} | AUC: {m_rf5['roc_auc']:.3f}")
    print("\nArtifacts written (t+5):")
    print(f"  - {eq_path5}")

if __name__ == "__main__":
    main()