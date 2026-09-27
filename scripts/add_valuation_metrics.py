import pandas as pd
import numpy as np

def clean_money(val):
    if pd.isna(val) or val == '-':
        return 0.0
    if isinstance(val, str):
        val = val.replace(',', '').strip()
        if val == '' or val == '-':
            return 0.0
    try:
        return float(val)
    except:
        return 0.0

def main():
    print("Loading datasets...")
    hydro_df = pd.read_csv("data/hydro.csv")
    master_df = pd.read_csv("data/master_ml_dataset.csv")
    
    if "Market_Cap" in master_df.columns:
        master_df = master_df.drop(columns=["Market_Cap"])
    if "Total_Debt" in master_df.columns:
        master_df = master_df.drop(columns=["Total_Debt"])
    
    hydro_df = hydro_df[hydro_df["Quarter"].astype(str) == "4"].copy()
    hydro_df["Year"] = hydro_df["Year"].astype(str).str.split("/").str[0]
    hydro_df["Year_Int"] = pd.to_numeric(hydro_df["Year"], errors="coerce")
    
    hydro_df["QuarterEndPrice"] = hydro_df["QuarterEndPrice"].apply(clean_money)
    hydro_df["OrdinaryShares"] = hydro_df["OrdinaryShares"].apply(clean_money)
    hydro_df["LtLiabilities"] = hydro_df["LtLiabilities"].apply(clean_money)
    hydro_df["StLiabilities"] = hydro_df["StLiabilities"].apply(clean_money)
    
    # Financial scale:
    # OperatingCashFlow_Proxy in master_df is EBIT from hydro.csv.
    # In Nepalese statements, if EBIT is in Rs 000s, so are Liabilities and Shares.
    # We will assume QuarterEndPrice is in Rs (e.g. 300) and OrdinaryShares is in '000s.
    # So Market_Cap = Price * Shares is in '000s, matching CashFlow and Debt.
    
    hydro_df["Market_Cap"] = hydro_df["QuarterEndPrice"] * hydro_df["OrdinaryShares"]
    hydro_df["Total_Debt"] = (hydro_df["LtLiabilities"] + hydro_df["StLiabilities"])
    
    print("Merging metrics into master dataset...")
    sub_hydro = hydro_df[["Ticker", "Year_Int", "Market_Cap", "Total_Debt"]].drop_duplicates(subset=["Ticker", "Year_Int"])
    
    merged = pd.merge(master_df, sub_hydro, on=["Ticker", "Year_Int"], how="left")
    
    if merged["Market_Cap"].isna().any():
        merged["PS_Ratio"] = merged["Market_Cap"] / merged["Revenue"].replace(0, np.nan)
        yearly_med_ps = merged.groupby("Year_Int")["PS_Ratio"].median().sort_index()
        expanding_med_ps = yearly_med_ps.expanding().median()
        merged["Expanding_Med_PS"] = merged["Year_Int"].map(expanding_med_ps)
        merged["Expanding_Med_PS"] = merged["Expanding_Med_PS"].ffill().bfill()
        merged["Market_Cap"] = merged["Market_Cap"].fillna(merged["Revenue"] * merged["Expanding_Med_PS"])
        merged = merged.drop(columns=["PS_Ratio", "Expanding_Med_PS"])
        
    if merged["Total_Debt"].isna().any():
        merged["Total_Debt"] = merged["Total_Debt"].fillna(0)
    
    merged.to_csv("data/master_ml_dataset.csv", index=False)
    print(f"Successfully added Market_Cap and Total_Debt to master dataset! ({len(merged)} rows)")

if __name__ == "__main__":
    main()
