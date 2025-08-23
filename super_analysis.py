# super_analysis.py - from ChatGPT
#
# Ben Cavanagh
# 08-23-2025
# Description: Analysis for e27 project
#

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess

# --------------------------
# User-defined functions
# --------------------------

def noisy_channel(bits: str) -> str:
    cmd = [
        "curl",
        "-X", "POST",
        "-d", f"bits={bits}",
        "https://engs27.host.dartmouth.edu/cgi-bin/noisychannel.py"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (code {result.returncode}):\n{result.stderr}")
    return result.stdout.split("<body>")[1].split("</body>")[0]

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must have the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def sample_noisy_channel(test_input, times):
    samples = []
    for _ in range(times):
        noisy_output = noisy_channel(test_input)
        dist = hamming_distance(test_input, noisy_output)
        samples.append({'input': test_input, 'output': noisy_output, 'hamming_distance': dist})
    return pd.DataFrame(samples)

def analyze_and_save(df, bit_pattern_name, length, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    mean_dist = df['hamming_distance'].mean()
    std_dist = df['hamming_distance'].std()

    bit_array = np.array([[int(b) for b in out] for out in df['output']])
    if "alternating" in bit_pattern_name:
        bit_val_array = np.array([[int(c) for c in df['input'][0]]]*len(df))
    else:
        bit_val = int(df['input'][0][0])
        bit_val_array = bit_val * np.ones(bit_array.shape)

    per_bit_errors = np.mean(bit_array != bit_val_array, axis=0)

    df_per_bit = pd.DataFrame({'bit_position': np.arange(len(per_bit_errors)),
                               'error_probability': per_bit_errors})
    df_per_bit.to_csv(os.path.join(output_dir, f"{bit_pattern_name}_length{length}_per_bit.csv"), index=False)

    plt.figure(figsize=(8,4))
    sns.barplot(x=np.arange(len(per_bit_errors)), y=per_bit_errors)
    plt.xlabel("Bit Position")
    plt.ylabel("Error Probability")
    plt.title(f"Per-Bit Error Probability: {bit_pattern_name}, Length={length}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bit_pattern_name}_length{length}_per_bit.png"))
    plt.close()

    bit_errors = (bit_array != bit_val_array).astype(int)
    corr_matrix = np.corrcoef(bit_errors.T)
    pd.DataFrame(corr_matrix).to_csv(os.path.join(output_dir, f"{bit_pattern_name}_length{length}_correlation.csv"), index=False)

    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation of Bit Errors: {bit_pattern_name}, Length={length}")
    plt.xlabel("Bit Position")
    plt.ylabel("Bit Position")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bit_pattern_name}_length{length}_correlation.png"))
    plt.close()

    return mean_dist, std_dist

# --------------------------
# Main workflow with nested progress bars
# --------------------------

if __name__ == "__main__":
    lengths = [2**i for i in range(9)]  # 1,2,4,...,256
    times = 100
    output_dir = "results"
    summary_records = []

    total_tasks = len(lengths) * 4  # 4 patterns per length
    with tqdm(total=total_tasks, desc="Overall progress") as outer_pbar:
        for length in lengths:
            patterns = {
                "all_0s": "0"*length,
                "all_1s": "1"*length,
                "alternating_0_start": "".join(['0' if i%2==0 else '1' for i in range(length)]),
                "alternating_1_start": "".join(['1' if i%2==0 else '0' for i in range(length)])
            }

            for pattern_name, test_input in patterns.items():
                # Inner progress bar for samples
                df_samples = []
                for _ in tqdm(range(times), desc=f"{pattern_name} length {length}", leave=False):
                    noisy_output = noisy_channel(test_input)
                    dist = hamming_distance(test_input, noisy_output)
                    df_samples.append({'input': test_input, 'output': noisy_output, 'hamming_distance': dist})
                df_samples = pd.DataFrame(df_samples)

                mean_dist, std_dist = analyze_and_save(df_samples, pattern_name, length, output_dir=output_dir)
                summary_records.append({
                    'length': length,
                    'pattern': pattern_name,
                    'mean_hamming_distance': mean_dist,
                    'std_hamming_distance': std_dist
                })
                outer_pbar.update(1)

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)
    print("All data and plots saved in folder:", output_dir)

