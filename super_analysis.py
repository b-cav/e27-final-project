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
    if len(str1) != len(str2):
        raise ValueError("Strings must have the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def sample_noisy_channel_parallel(test_input, times, max_workers=16):
    """Sample the noisy channel in parallel using ThreadPoolExecutor."""
    results = []

    def single_sample(_):
        output = noisy_channel(test_input)
        dist = hamming_distance(test_input, output)
        return {'input': test_input, 'output': output, 'hamming_distance': dist}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_sample, i) for i in range(times)]
        for f in tqdm(as_completed(futures), total=times, desc=f"Sampling length {len(test_input)}"):
            results.append(f.result())

    return pd.DataFrame(results)

# --------------------------
# Main workflow
# --------------------------

if __name__ == "__main__":
    lengths = [256, 512, 1048]
    times = 100
    max_workers = 16  # tune based on network capacity
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

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
                # Sample in parallel
                df_samples = sample_noisy_channel_parallel(test_input, times, max_workers=max_workers)

                # Compute summary stats
                mean_dist = df_samples['hamming_distance'].mean()
                std_dist = df_samples['hamming_distance'].std()

                # Save raw samples
                df_samples.to_csv(os.path.join(output_dir, f"{pattern_name}_length{length}_samples.csv"), index=False)

                summary_records.append({
                    'length': length,
                    'pattern': pattern_name,
                    'mean_hamming_distance': mean_dist,
                    'std_hamming_distance': std_dist
                })

                outer_pbar.update(1)

    # Save overall summary
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)
    print("All data saved in folder:", output_dir)

