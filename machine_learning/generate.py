import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# ==========================================
# 0. Global Settings for Publication Quality
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif', # Standard for academic papers
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,      # High resolution (300 DPI)
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def generate_all_plots():
    # ==========================================
    # PLOT 1: Actual vs. Predicted FCFF (Line Chart)
    # ==========================================
    print("Generating Plot 1: Time-Series Actual vs Predicted...")
    years = np.array([2019, 2020, 2021, 2022, 2023])
    companies = ["CHCL", "API", "UPPER"]

    # Sample historical tracking data
    data = {
        "CHCL":  {"actual": [100, 115, 95, 140, 175], "ml": [105, 112, 100, 138, 170], "dcf": [80, 95, 110, 100, 145]},
        "API":   {"actual": [150, 160, 155, 180, 200], "ml": [148, 165, 160, 175, 198], "dcf": [130, 140, 170, 150, 160]},
        "UPPER": {"actual": [200, 190, 210, 230, 250], "ml": [195, 195, 205, 225, 248], "dcf": [170, 160, 180, 200, 210]}
    }

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig1.suptitle("Actual vs. Predicted FCFF (2019-2023)", fontsize=18, fontweight='bold', y=1.05)

    for i, company in enumerate(companies):
        ax = axes1[i]
        ax.plot(years, data[company]["actual"], marker='o', linewidth=2.5, color='#1f77b4', label='Actual FCFF')
        ax.plot(years, data[company]["ml"], marker='s', linewidth=2.5, color='#2ca02c', linestyle='--', label='ML Predicted (XGBoost)')
        ax.plot(years, data[company]["dcf"], marker='^', linewidth=2.5, color='#d62728', linestyle=':', label='Traditional DCF')
        
        ax.set_title(f"Company: {company}", fontweight='bold')
        ax.set_xlabel("Fiscal Year")
        ax.set_xticks(years)
        
        if i == 0:
            ax.set_ylabel("Free Cash Flow (in millions)")
        ax.legend()

    plt.tight_layout()
    plt.savefig('Plot_1_TimeSeries.png', bbox_inches='tight')
    plt.close()


    # ==========================================
    # PLOT 2: Model Accuracy Comparison (Bar Chart)
    # ==========================================
    print("Generating Plot 2: Model Accuracy Comparison...")
    models = ['XGBoost', 'Random Forest', 'Traditional DCF']
    rmse_values = [45.2, 53.8, 89.4]  # Example metrics
    mae_values = [32.1, 39.5, 65.2]

    x = np.arange(len(models))
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    rects1 = ax2.bar(x - width/2, rmse_values, width, label='RMSE', color='#4c72b0', edgecolor='black')
    rects2 = ax2.bar(x + width/2, mae_values, width, label='MAE', color='#dd8452', edgecolor='black')

    ax2.set_ylabel('Error Magnitude (Lower is Better)', fontweight='bold')
    ax2.set_title('Model Accuracy Comparison: ML vs Traditional DCF', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight='bold')
    ax2.legend()

    # Add numeric labels above bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1, ax2)
    autolabel(rects2, ax2)

    plt.tight_layout()
    plt.savefig('Plot_2_Error_Comparison.png', bbox_inches='tight')
    plt.close()


    # ==========================================
    # PLOT 3: Feature Importance Explainer (SHAP)
    # ==========================================
    print("Generating Plot 3: SHAP Summary Plot...")
    
    # Create mock dataset and model just to generate the SHAP visual 
    # (Replace this block with your actual trained XGBoost model and X_test data)
    np.random.seed(42)
    X_mock = pd.DataFrame({
        'PLF': np.random.normal(0.6, 0.1, 100),
        'GDP_Growth': np.random.normal(5.0, 1.5, 100),
        'Inflation_Rate': np.random.normal(6.0, 2.0, 100),
        'Interest_Rate': np.random.normal(8.0, 1.0, 100),
        'CapEx': np.random.normal(50, 10, 100)
    })
    y_mock = X_mock['PLF'] * 100 + X_mock['GDP_Growth'] * 50 - X_mock['Inflation_Rate'] * 30 + np.random.normal(0, 10, 100)
    
    mock_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    mock_xgb.fit(X_mock, y_mock)

    # Generate SHAP Plot
    explainer = shap.Explainer(mock_xgb)
    shap_values = explainer(X_mock)

    fig3 = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_mock, show=False, plot_type="dot")
    plt.title("Impact of Operational & Macro Variables on FCFF Prediction", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('Plot_3_SHAP_Summary.png', bbox_inches='tight')
    plt.close()


    # ==========================================
    # PLOT 4: Macro-Shock Scenario Analysis
    # ==========================================
    print("Generating Plot 4: Macro-Shock Scenario Analysis...")
    forecast_years = np.array([2024, 2025, 2026, 2027, 2028])

    base_fcf = np.array([190, 210, 230, 245, 260])
    inflation_5_fcf = base_fcf * np.array([1.0, 0.98, 0.97, 0.96, 0.95])
    inflation_15_fcf = base_fcf * np.array([0.90, 0.82, 0.75, 0.68, 0.60])

    fig4, ax4 = plt.subplots(figsize=(10, 6))

    ax4.plot(forecast_years, inflation_5_fcf, marker='o', linewidth=3, color='#2ca02c', label='Baseline Scenario (5% Inflation)')
    ax4.plot(forecast_years, inflation_15_fcf, marker='s', linewidth=3, color='#d62728', linestyle='--', label='Macro-Shock (15% Inflation Spike)')

    # Fill between to show Value at Risk
    ax4.fill_between(forecast_years, inflation_15_fcf, inflation_5_fcf, color='red', alpha=0.1, label='Value at Risk (Lost Cash Flow)')

    ax4.set_title("5-Year FCFF Projection Under Macroeconomic Shocks (Test Company: CHCL)", fontsize=16, fontweight='bold')
    ax4.set_xlabel("Forecast Year", fontweight='bold')
    ax4.set_ylabel("Predicted Free Cash Flow (in millions)", fontweight='bold')
    ax4.set_xticks(forecast_years)
    ax4.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('Plot_4_Macro_Shock.png', bbox_inches='tight')
    plt.close()

    print("✅ All plots successfully generated and saved as PNG files!")

# Run the function to generate plots
if __name__ == "__main__":
    generate_all_plots()