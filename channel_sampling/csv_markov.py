"""
csv_markov.py

ENGS 27 Final Project Mass HTTP Sampling
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

8/27 modification of original markov_test.py Ben C
"""


import pandas as pd

def bit_flip_stats(df: pd.DataFrame, return_dict: bool = False):
    """
    Computes conditional bit flip probabilities from dataframe.
    Normalizes columns to strings so filtering works even if CSV was read as integers.

    Columns expected:
      - pattern: '0','1','00','11','10','01' (as strings; if read as ints, '00'/'01' become '0'/'1')
      - sent: last bit sent ('0' or '1')
      - received: last bit received ('0' or '1')

    If you already loaded with integers and lost leading zeros, re-read with:
      df = pd.read_csv("markov_results.csv", dtype={'pattern': str, 'sent': str, 'received': str})
    """
    df = df.copy()

    # Normalize to strings and strip whitespace
    for c in ['pattern', 'sent', 'received']:
        df[c] = df[c].astype(str).str.strip()

    # Flip indicator (last bit flipped or not)
    df['flip'] = (df['sent'] != df['received']).astype(int)

    def mean_flip(mask):
        subset = df.loc[mask]
        if subset.empty:
            return None
        return float(subset['flip'].mean())

    results = {}

    # Single-bit: no preceding
    for bit in ['0', '1']:
        results[f'flip|no_preceding,{bit}'] = mean_flip(df['pattern'] == bit)

    # Two-bit: preceding same/different (Markov behavior on the second bit)
    for bit in ['0', '1']:
        same_pattern = bit * 2              # '00' for 0, '11' for 1
        diff_pattern = ('1' if bit == '0' else '0') + bit  # '10' for 0, '01' for 1
        results[f'flip|preceding_same,{bit}'] = mean_flip(df['pattern'] == same_pattern)
        results[f'flip|preceding_diff,{bit}'] = mean_flip(df['pattern'] == diff_pattern)

    # Pretty print
    def fmt(v):
        return f"{v:.3f}" if v is not None else "N/A (no data)"

    for bit in ['0', '1']:
        print(f"P(flip | no preceding, {bit}) = {fmt(results[f'flip|no_preceding,{bit}'])}")
    for bit in ['0', '1']:
        print(f"P(flip | preceding same, {bit}) = {fmt(results[f'flip|preceding_same,{bit}'])}")
        print(f"P(flip | preceding different, {bit}) = {fmt(results[f'flip|preceding_diff,{bit}'])}")

    if return_dict:
        return results
# Example usage:
# 1. Run markov test once to generate results
#markov_test("markov_results.csv", trials=100)

# 2. Load results and compute probabilities
df = pd.read_csv("markov_results.csv", dtype={'pattern': str, 'sent': str, 'received': str})
bit_flip_stats(df)
