# chat_analysis.py - Sampling noisy channel E27
#
# Ben Cavanagh
# 08-23-2025
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Noisy Channel Access
# --------------------------
import subprocess

def noisy_channel(bits: str) -> str:
    """
    Calls curl to POST the given bits to the test.py endpoint
    and returns the stdout response as a string.
    """
    # Build the curl command and arguments:
    cmd = [
        "curl",
        "-X", "POST",
        "-d", f"bits={bits}",
        "https://engs27.host.dartmouth.edu/cgi-bin/noisychannel.py"
    ]

    # Run the command, capture stdout/stderr
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,       # return strings instead of bytes
        check=False      # we’ll inspect returncode manually
    )

    if result.returncode != 0:
        # curl failed. You can raise, log, or return stderr.
        raise RuntimeError(f"curl failed (code {result.returncode}):\n{result.stderr}")

    return result.stdout.split("<body>")[1].split("</body>")[0]

# --------------------------
# User-defined functions
# --------------------------

def hamming_distance(str1, str2):
    """Compute Hamming distance between two strings."""
    if len(str1) != len(str2):
        raise ValueError("Strings must have the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def sample_noisy_channel(length, bit_val, times):
    """Sample the noisy channel multiple times."""
    test_input = str(bit_val) * length
    samples = []

    for _ in range(times):
        noisy_output = noisy_channel(test_input)  # must be defined by user
        dist = hamming_distance(test_input, noisy_output)
        samples.append({
            'input': test_input,
            'output': noisy_output,
            'hamming_distance': dist
        })
    
    df = pd.DataFrame(samples)
    return df

# --------------------------
# Analysis and Visualization
# --------------------------

def analyze_noisy_channel(df, bit_val=1):
    """Compute statistics, per-bit errors, and correlations."""
    # Overall Hamming stats
    mean_dist = df['hamming_distance'].mean()
    std_dist = df['hamming_distance'].std()
    print(f"Mean Hamming distance: {mean_dist}")
    print(f"Std Dev: {std_dist}")

    # Convert output strings to NumPy array
    bit_array = np.array([[int(b) for b in out] for out in df['output']])
    bit_val_array = bit_val * np.ones(bit_array.shape)
    
    # Per-bit error probability
    per_bit_errors = np.mean(bit_array != bit_val_array, axis=0)
    plt.figure(figsize=(8,4))
    sns.barplot(x=np.arange(len(per_bit_errors)), y=per_bit_errors)
    plt.xlabel("Bit Position")
    plt.ylabel("Error Probability")
    plt.title("Per-Bit Error Probability")
    plt.show()

    # Correlation matrix between bit errors
    bit_errors = (bit_array != bit_val_array).astype(int)
    corr_matrix = np.corrcoef(bit_errors.T)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation of Bit Errors Between Positions")
    plt.xlabel("Bit Position")
    plt.ylabel("Bit Position")
    plt.show()

    return mean_dist, std_dist, per_bit_errors, corr_matrix

# --------------------------
# Main workflow
# --------------------------

if __name__ == "__main__":
    # Parameters
    LENGTH = 10       # length of input string
    BIT_VAL = 1       # value of repeated bit
    TIMES = 1000      # number of samples

    # Sample the noisy channel
    df_samples = sample_noisy_channel(LENGTH, BIT_VAL, TIMES)

    # Analyze
    analyze_noisy_channel(df_samples, bit_val=BIT_VAL)


