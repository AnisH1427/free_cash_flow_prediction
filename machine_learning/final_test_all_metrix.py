import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def main():
    print("🚀 Running ML for Hydropower (with Outlier Handling & Imputation)")
    data_path = "master_ml_dataset.csv"

    try:
        df = pd.read_csv(data_path)
        print("✅ Loaded data successfully.")
    except FileNotFoundError:
        print("❌ File not found.")
        return

    target_column = "FCF"
    if target_column not in df.columns:
        print(f"❌ Target column '{target_column}' not found in the dataset.")
        return

    # 1. Drop non-predictive columns
    potential_drops = ['company_id', 'symbol', 'fiscal_year', 'company_name', 'date', 'Unnamed: 0']
    cols_to_drop = [col for col in potential_drops if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"🧹 Dropped non-predictive identifiers: {cols_to_drop}")

    # 2. Clean infinite values and drop missing targets
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_column])

    # FIX A: Remove extreme target outliers (Drop top 5% and bottom 5%)
    lower_bound = df[target_column].quantile(0.05)
    upper_bound = df[target_column].quantile(0.95)
    df = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]
    print(f"✂️ Clipped extreme FCF outliers. Remaining rows: {len(df)}")

    # FIX B: Median Imputation instead of filling with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.fillna('Unknown')

    # --- Leakage check: how many test-set companies were already seen in training? ---
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
    train_tickers = set(df.loc[idx_train, 'Ticker']) if 'Ticker' in df.columns else set()
    test_tickers = set(df.loc[idx_test, 'Ticker']) if 'Ticker' in df.columns else set()
    if test_tickers:
        overlap_pct = 100 * len(train_tickers & test_tickers) / len(test_tickers)
        print(f"⚠️  {overlap_pct:.0f}% of test-set companies ({len(train_tickers & test_tickers)}/{len(test_tickers)}) "
              f"also appear in the training set (panel data + random split = leakage)\n")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Data split complete. Training on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

    def report(name, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"   ↳ {name} MSE:  {mse:,.4f}")
        print(f"   ↳ {name} RMSE: {rmse:,.4f}")
        print(f"   ↳ {name} MAE:  {mae:,.4f}")
        print(f"   ↳ {name} R²:   {r2:.4f}\n")
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    print("🌲 Random Forest Regressor")
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_metrics = report("Random Forest", y_test, rf_predictions)

    print("⚡ XGBoost Regressor")
    xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_metrics = report("XGBoost", y_test, xgb_predictions)

    results_df = pd.DataFrame({
        'Actual_FCF': y_test,
        'RF_Predicted_FCF': rf_predictions,
        'XGB_Predicted_FCF': xgb_predictions,
        'RF_Error': y_test - rf_predictions,
        'XGB_Error': y_test - xgb_predictions
    })

    save_path = "model_predictions.csv"
    results_df.to_csv(save_path, index=False)
    print(f"📁 Predictions saved to {save_path}")

if __name__ == "__main__":
    main()