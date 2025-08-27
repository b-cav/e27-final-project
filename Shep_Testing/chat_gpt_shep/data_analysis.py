# simple_probs_runner.py
# Drop this into VS Code and click the green "Run Python File" button.
# It reads raw.csv in the current working directory and prints pooled probabilities:
#  - ber (overall bit error rate)
#  - p_0to1 (probability 0 flips to 1)
#  - p_1to0 (probability 1 flips to 0)
# It also saves two tiny CSVs: simple_probs_by_length.csv and simple_probs_overall.csv

import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_PATH = Path("raw.csv")  # your raw file in the current working directory

# Required grouping columns
REQ_GROUP_COLS = {"pattern_type", "length"}

# Preferred count columns (we'll compute them if missing and input_bits/output_bits are available)
COUNT_COLS = {"zeros_in", "ones_in", "flips_0_to_1", "flips_1_to_0"}

def _compute_counts_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure zeros_in, ones_in, flips_0_to_1, flips_1_to_0 exist.
    If missing and input_bits/output_bits exist, compute them.
    """
    have_counts = COUNT_COLS.issubset(df.columns)
    have_bits = {"input_bits", "output_bits"}.issubset(df.columns)

    if have_counts:
        return df

    if not have_bits:
        missing = COUNT_COLS - set(df.columns)
        raise ValueError(
            "raw.csv is missing count columns and also lacks input_bits/output_bits to compute them.\n"
            f"Missing: {sorted(missing)}"
        )

    # Compute counts row-wise
    zeros_in = []
    ones_in = []
    flips_01 = []
    flips_10 = []
    for _, r in df.iterrows():
        x = str(r["input_bits"])
        y = str(r["output_bits"])
        if len(x) != len(y):
            raise ValueError("Mismatch in input_bits/output_bits length in raw.csv")
        zeros_in.append(x.count("0"))
        ones_in.append(x.count("1"))
        f01 = sum(1 for a, b in zip(x, y) if a == "0" and b == "1")
        f10 = sum(1 for a, b in zip(x, y) if a == "1" and b == "0")
        flips_01.append(f01)
        flips_10.append(f10)

    df = df.copy()
    df["zeros_in"] = zeros_in
    df["ones_in"] = ones_in
    df["flips_0_to_1"] = flips_01
    df["flips_1_to_0"] = flips_10
    return df

def _safe_div(num, den):
    return num / den if den else np.nan

def simple_probs_from_path(raw_csv_path: Path):
    if not raw_csv_path.exists():
        print(f"Could not find file: {raw_csv_path.resolve()}")
        sys.exit(1)

    df = pd.read_csv(raw_csv_path)

    # Ensure required grouping columns exist
    missing_grp = REQ_GROUP_COLS - set(df.columns)
    if missing_grp:
        raise ValueError(f"raw.csv is missing required columns: {sorted(missing_grp)}")

    # Ensure counts exist (or compute from input/output bits)
    df = _compute_counts_if_needed(df)

    # By (pattern_type, length): pooled (weighted) rates from summed counts
    by_len = (df
        .groupby(["pattern_type","length"], as_index=False)[["zeros_in","ones_in","flips_0_to_1","flips_1_to_0"]]
        .sum())
    denom_bits = by_len["zeros_in"] + by_len["ones_in"]
    by_len["ber"]    = (by_len["flips_0_to_1"] + by_len["flips_1_to_0"]) / denom_bits.replace({0: np.nan})
    by_len["p_0to1"] = by_len["flips_0_to_1"] / by_len["zeros_in"].replace({0: np.nan})
    by_len["p_1to0"] = by_len["flips_1_to_0"] / by_len["ones_in"].replace({0: np.nan})

    # Overall pooled across lengths (one line per pattern_type)
    overall = (df
        .groupby("pattern_type", as_index=False)[["zeros_in","ones_in","flips_0_to_1","flips_1_to_0"]]
        .sum())
    denom_bits_o = overall["zeros_in"] + overall["ones_in"]
    overall["ber"]    = (overall["flips_0_to_1"] + overall["flips_1_to_0"]) / denom_bits_o.replace({0: np.nan})
    overall["p_0to1"] = overall["flips_0_to_1"] / overall["zeros_in"].replace({0: np.nan})
    overall["p_1to0"] = overall["flips_1_to_0"] / overall["ones_in"].replace({0: np.nan})

    # Keep only concise columns for readability
    by_len_out = by_len[["pattern_type","length","ber","p_0to1","p_1to0"]].sort_values(["pattern_type","length"]).reset_index(drop=True)
    overall_out = overall[["pattern_type","ber","p_0to1","p_1to0"]].sort_values(["pattern_type"]).reset_index(drop=True)

    # Print clean tables
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\nPooled probabilities by (pattern_type, length):")
    print(by_len_out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\nPooled probabilities across lengths (one line per pattern):")
    print(overall_out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # Save tiny CSVs
    by_len_out.to_csv("simple_probs_by_length.csv", index=False)
    overall_out.to_csv("simple_probs_overall.csv", index=False)
    print("\nWrote: simple_probs_by_length.csv, simple_probs_overall.csv")

if __name__ == "__main__":
    # Click 'Run Python File' in VS Code and it will analyze ./raw.csv
    simple_probs_from_path(RAW_PATH)