# edit_distance.py - 
#
# Ben Cavanagh
# 09-02-2025
# Description: 
#

import os
import sys
import random
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# -----------------------------
# Add main folder to Python path
# -----------------------------
main_folder = os.path.abspath(os.path.join("../../main"))
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)

from huffman import final_compression as fc
from error_correction import hamming as hm
from error_correction import raid

# -----------------------------
# Helper functions
# -----------------------------
def sample_text(full_text, sample_len=500):
    start = random.randint(0, max(0, len(full_text) - sample_len))
    return full_text[start:start + sample_len]

def edit_distance_ratio(original, received):
    """Normalized similarity: 1 = perfect match, 0 = completely different"""
    matcher = SequenceMatcher(None, original, received)
    return matcher.ratio()

def ensure_plot_folder(folder="./raid_testing_plots"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

# -----------------------------
# Pipelines
# -----------------------------
def run_pipeline(message, codebook, bigrams, tree, use_raid=True, use_hamming=True):
    # -----------------
    # Encode
    compressed = fc.encode_message(fc.replace_bigrams(message, bigrams), codebook)
    
    if use_hamming:
        encoded = hm.EHC_16_11_encode(compressed)
    else:
        encoded = list(compressed)  # treat as single-bit packets if Hamming disabled

    if use_raid and use_hamming:
        protected = raid.RAID_protect(encoded)
    else:
        protected = encoded

    # -----------------
    # Transmit through noisy channel
    received = hm.noisy_channel(protected if use_hamming else ["".join(format(ord(c), "08b") for c in message)], 0 if use_hamming else 1)

    # -----------------
    # Recover
    if use_hamming and use_raid:
        raid_recovered, _, _ = raid.RAID_remove(received)
        cleaned = hm.EHC_16_11_clean(raid_recovered)
        unpadded = hm.remove_padding(cleaned)
        decoded = fc.decode_message(unpadded, tree)
    elif use_hamming and not use_raid:
        cleaned = hm.EHC_16_11_clean(received)
        unpadded = hm.remove_padding(cleaned)
        decoded = fc.decode_message(unpadded, tree)
    else:  # no error correction
        ascii_received = []
        for packet in received:
            ascii_received.append("".join(chr(int(packet[i:i+8], 2)) for i in range(0, len(packet), 8)))
        decoded = "".join(ascii_received)

    return decoded

# -----------------------------
# Main testing loop
# -----------------------------
if __name__ == "__main__":
    # Load WarAndPeace
    main_path = os.path.join(f"{main_folder}/huffman", "WarAndPeace.txt")
    with open(main_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("BUILDING TREE...")
    codebook, bigrams, tree = fc.huffman_init(main_path)

    sample_len = 500  # fixed length for comparison
    num_samples = 50  # number of random samples

    edit_unprotected = []
    edit_hamming_only = []
    edit_hamming_raid = []

    for _ in range(num_samples):
        sample = sample_text(text, sample_len)

        # Run all three conditions
        dec_unprot = run_pipeline(sample, codebook, bigrams, tree, use_raid=False, use_hamming=False)
        dec_hamming = run_pipeline(sample, codebook, bigrams, tree, use_raid=False, use_hamming=True)
        dec_raid = run_pipeline(sample, codebook, bigrams, tree, use_raid=True, use_hamming=True)

        edit_unprotected.append(edit_distance_ratio(sample, dec_unprot))
        edit_hamming_only.append(edit_distance_ratio(sample, dec_hamming))
        edit_hamming_raid.append(edit_distance_ratio(sample, dec_raid))

    # -----------------------------
    # Plot line comparison
    # -----------------------------
    folder = ensure_plot_folder()

    plt.figure(figsize=(12,6))
    plt.plot(edit_unprotected, label="Unprotected", marker='o')
    plt.plot(edit_hamming_only, label="Hamming-only", marker='x')
    plt.plot(edit_hamming_raid, label="Hamming + RAID", marker='s')
    plt.xlabel("Random Sample #")
    plt.ylabel("Edit Distance Similarity")
    plt.title("Message Similarity After Transmission (Edit Distance)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, "edit_distance_comparison_line.png"))
    plt.show()

    # -----------------------------
    # Plot histograms
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.hist(edit_unprotected, bins=20, alpha=0.6, label="Unprotected")
    plt.hist(edit_hamming_only, bins=20, alpha=0.6, label="Hamming-only")
    plt.hist(edit_hamming_raid, bins=20, alpha=0.6, label="Hamming + RAID")
    plt.xlabel("Edit Distance Similarity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Edit Distance Similarity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, "edit_distance_comparison_hist.png"))
    plt.show()

    # -----------------------------
    # Summary stats
    # -----------------------------
    print(f"Average similarity unprotected: {sum(edit_unprotected)/num_samples:.4f}")
    print(f"Average similarity Hamming-only: {sum(edit_hamming_only)/num_samples:.4f}")
    print(f"Average similarity Hamming + RAID: {sum(edit_hamming_raid)/num_samples:.4f}")

