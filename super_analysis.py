# super_analysis.py - from ChatGPT
#
# Ben Cavanagh
# 08-23-2025
# Description: Analysis for e27 project
#

import numpy as np
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# --------------------------
# User-defined functions
# --------------------------

def noisy_channel(bits: str) -> str:
    """Call the noisy channel via curl."""
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
    """Compute Hamming distance between two bit strings."""
    if len(str1) != len(str2):
        raise ValueError("Strings must have the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def sample_noisy_channel_parallel(test_input, num_samples, max_workers=16):
    """Sample the noisy channel in parallel."""
    results = []

    def single_sample(_):
        output = noisy_channel(test_input)
        dist = hamming_distance(test_input, output)
        return output, dist

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_sample, i) for i in range(num_samples)]
        for f in tqdm(as_completed(futures), total=num_samples, desc=f"Sampling len {len(test_input)}"):
            output, dist = f.result()
            results.append({'input': test_input, 'output': output, 'hamming_distance': dist})

    return pd.DataFrame(results)

def compute_per_bit_errors(df):
    """Compute per-bit error probabilities."""
    bit_length = len(df['input'][0])
    outputs = np.array([[int(b) for b in out] for out in df['output']])
    inputs = np.array([[int(b) for b in inp] for inp in df['input']])
    errors = outputs != inputs
    per_bit_probs = errors.mean(axis=0)
    return pd.DataFrame({'bit_position': np.arange(bit_length), 'error_probability': per_bit_probs})

# --------------------------
# Main workflow
# --------------------------

if __name__ == "__main__":
    lengths = [128, 256, 512]
    num_samples = 40
    max_workers = 16
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    summary_records = []

    for length in lengths:
        patterns = {
            "all_0s": "0"*length,
            "all_1s": "1"*length,
            "alternating_0_start": "".join(['0' if i%2==0 else '1' for i in range(length)]),
            "alternating_1_start": "".join(['1' if i%2==0 else '0' for i in range(length)])
        }

        for pattern_name, test_input in patterns.items():
            print(f"Sampling: length={length}, pattern={pattern_name}")
            df_samples = sample_noisy_channel_parallel(test_input, num_samples, max_workers=max_workers)

            # Compute summary stats
            mean_hd = df_samples['hamming_distance'].mean()
            std_hd = df_samples['hamming_distance'].std()

            # Save raw samples
            sample_file = os.path.join(output_dir, f"{pattern_name}_len{length}_samples.csv")
            df_samples.to_csv(sample_file, index=False)

            # Compute per-bit errors
            df_per_bit = compute_per_bit_errors(df_samples)
            per_bit_file = os.path.join(output_dir, f"{pattern_name}_len{length}_perbit.csv")
            df_per_bit.to_csv(per_bit_file, index=False)

            # Record summary stats
            summary_records.append({
                'length': length,
                'pattern': pattern_name,
                'mean_hamming_distance': mean_hd,
                'std_hamming_distance': std_hd,
                'num_samples': num_samples
            })

    # Save overall summary
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)
    print("All data saved in folder:", output_dir)

