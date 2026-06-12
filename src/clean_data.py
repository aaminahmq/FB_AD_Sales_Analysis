
#clean_data.py
#Cleans the Facebook ad campaign dataset (Sales Conversion Optimization) and
#adds marketing KPIs plus a binary model target

#input data/raw/KAG_conversion_data.csv --output data/cleaned/conversion_data_clean.csv


import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_div(a, b):
    #Divide a by b, returning NaN wherever b == 0 (avoids divide-by-zero)
    return np.where(b == 0, np.nan, a / b)


def clean(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # 1. standardize column names to snake_case (warehouse-friendly)
    df.columns = [c.lower() for c in df.columns]

    # 2. flag zero-spend rows instead of dropping them
    df["is_zero_spend"] = df["spent"] == 0

    # 3. derived KPIs with safe division
    df["ctr"]                      = safe_div(df["clicks"], df["impressions"])
    df["cpc"]                      = safe_div(df["spent"], df["clicks"])
    df["cost_per_total_conv"]      = safe_div(df["spent"], df["total_conversion"])
    df["cost_per_approved_conv"]   = safe_div(df["spent"], df["approved_conversion"])
    df["click_to_enquiry_rate"]    = safe_div(df["total_conversion"], df["clicks"])
    df["enquiry_to_purchase_rate"] = safe_div(df["approved_conversion"], df["total_conversion"])

    # 4. binary model target: did this ad produce at least one purchase?
    df["purchased"] = (df["approved_conversion"] > 0).astype(int)

    # round rate/cost columns for readability
    rate_cols = [
        "ctr", "cpc", "cost_per_total_conv", "cost_per_approved_conv",
        "click_to_enquiry_rate", "enquiry_to_purchase_rate",
    ]
    df[rate_cols] = df[rate_cols].round(4)

    # write output, creating the folder if needed
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Zero-spend rows flagged: {int(df['is_zero_spend'].sum())}")
    print("Target balance (purchased):")
    print(df["purchased"].value_counts(normalize=True).round(3).to_string())
    print(f"Saved -> {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the FB ad campaign dataset.")
    parser.add_argument("--input",  default="data/raw/KAG_conversion_data.csv")
    parser.add_argument("--output", default="data/cleaned/conversion_data_clean.csv")
    args = parser.parse_args()
    clean(args.input, args.output)