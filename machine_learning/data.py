import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import xgboost as xgb

def main():
    print("ml for hydropower")
    data_path = "master_ml_dataset.csv"
    try:
        df = pd.read_csv(data_path)
        print(f"loaded data")
    except FileNotFoundError:
        print(f"File not found.")
        return
    target_column = "FCF"
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in the dataset.")
        return
   
    potential_drops = ['company_id', 'symbol', 'fiscal_year', 'company_name', 'date', 'Unnamed: 0']
    cols_to_drop = [col for col in potential_drops if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"🧹 Dropped non-predictive identifiers: {cols_to_drop}")
        
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_column])
    df = df.fillna(0)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split complete. Training on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

    print("random forest regressor")
    rf_model = RandomForestRegressor(n_estimators=150, max_depth = 10, random_state=42, n_jobs=1)
    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_rmse = root_mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Random Forest R^2: {rf_r2:.4f}\n")

    print("xgboost regressor")
    xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=1)
    xgb_model.fit(X_train, y_train)

    xgb_predictions = xgb_model.predict(X_test)
    
    xgb_rmse = root_mean_squared_error(y_test, xgb_predictions)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    print(f"XGBoost R^2: {xgb_r2:.4f}\n")

    results_df = pd.DataFrame({
        'Actual_FCF': y_test,
        'RF_Predicted_FCF': rf_predictions,
        'XGB_Predicted_FCF': xgb_predictions,
        'RF_Error': y_test - rf_predictions,
        'XGB_Error': y_test - xgb_predictions
    })

    save_path = "model_predictions.csv"
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

if __name__ == "__main__":
    main()