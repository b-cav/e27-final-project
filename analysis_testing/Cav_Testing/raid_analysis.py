# raid_analysis.py
#
# Ben Cavanagh
# 09-02-2025
# Description: Sample pipeline, understand raid benefits
#

import sys, os, random
import matplotlib.pyplot as plt

# Ensure ../../main is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../main")))

from huffman import final_compression as fc
from error_correction import hamming as hm
from error_correction import raid


def random_snippet(path, length=200):
    """Pick a random snippet of given length from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    start = random.randint(0, max(0, len(text) - length))
    return text[start:start+length]


def bit_error_rate(original_bits, received_bits):
    """Compute BER given two binary strings."""
    if len(original_bits) != len(received_bits):
        m = min(len(original_bits), len(received_bits))
        original_bits, received_bits = original_bits[:m], received_bits[:m]
    errors = sum(a != b for a, b in zip(original_bits, received_bits))
    return errors / len(original_bits) if original_bits else 0


def sample_and_plot(codebook, bigrams, tree, trials=100, snippet_len=200):
    results = {
        "ascii": {"ber": [], "success": []},
        "ecc": {"ber": [], "success": []},
        "raid": {"ber": [], "success": [], "lost_frac": []},
    }

    for _ in range(trials):
        message = random_snippet("../../main/huffman/WarAndPeace.txt", length=snippet_len)

        # ============ ASCII baseline ============
        ascii_original = "".join(format(ord(c), "08b") for c in message)
        ascii_noisy = hm.noisy_channel([ascii_original], 1)[0]
        ber_ascii = bit_error_rate(ascii_original, ascii_noisy)
        msg_ascii = "".join(chr(int(ascii_noisy[i:i+8], 2)) for i in range(0, len(ascii_noisy), 8))
        results["ascii"]["ber"].append(ber_ascii)
        results["ascii"]["success"].append(int(msg_ascii == message))

        # ============ ECC only ============
        compressed = fc.encode_message(fc.replace_bigrams(message, bigrams), codebook)
        encoded = hm.EHC_16_11_encode(compressed)
        noisy_ecc = hm.noisy_channel(encoded, 0)  # no RAID, just ECC
        cleaned_ecc = hm.EHC_16_11_clean(noisy_ecc)
        unpadded_ecc = hm.remove_padding(cleaned_ecc)
        decoded_ecc = fc.decode_message(unpadded_ecc, tree)

        ber_ecc = bit_error_rate("".join(encoded), "".join(cleaned_ecc))
        results["ecc"]["ber"].append(ber_ecc)
        results["ecc"]["success"].append(int(decoded_ecc == message))

        # ============ ECC + RAID ============
        protected = raid.RAID_protect(encoded)
        noisy_raid = hm.noisy_channel(protected, 0)
        recovered, lost_stripes, stripe_count = raid.RAID_remove(noisy_raid)
        cleaned_raid = hm.EHC_16_11_clean(recovered)
        unpadded_raid = hm.remove_padding(cleaned_raid)
        decoded_raid = fc.decode_message(unpadded_raid, tree)

        ber_raid = bit_error_rate("".join(encoded), "".join(cleaned_raid))
        results["raid"]["ber"].append(ber_raid)
        results["raid"]["success"].append(int(decoded_raid == message))
        results["raid"]["lost_frac"].append(lost_stripes / stripe_count if stripe_count else 0)

    # ---- Save plots ----
    out_dir = "./raid_testing_plots"
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 10))

    # BER histogram
    plt.subplot(2, 2, 1)
    plt.hist(results["ascii"]["ber"], bins=20, alpha=0.5, label="ASCII")
    plt.hist(results["ecc"]["ber"], bins=20, alpha=0.5, label="ECC only")
    plt.hist(results["raid"]["ber"], bins=20, alpha=0.5, label="ECC + RAID")
    plt.xlabel("Bit Error Rate")
    plt.ylabel("Frequency")
    plt.title("BER Distribution")
    plt.legend()

    # Lost stripes (RAID only)
    plt.subplot(2, 2, 2)
    plt.hist(results["raid"]["lost_frac"], bins=20, alpha=0.7, color="red")
    plt.xlabel("Fraction of stripes lost")
    plt.ylabel("Frequency")
    plt.title("RAID lost stripe fraction")

    # Message-level success counts
    plt.subplot(2, 2, 3)
    bars = [
        sum(results["ascii"]["success"]),
        sum(results["ecc"]["success"]),
        sum(results["raid"]["success"])
    ]
    fails = [
        len(results["ascii"]["success"]) - bars[0],
        len(results["ecc"]["success"]) - bars[1],
        len(results["raid"]["success"]) - bars[2]
    ]
    plt.bar(["ASCII", "ECC", "ECC+RAID"], bars, label="Perfect", color="green")
    plt.bar(["ASCII", "ECC", "ECC+RAID"], fails, bottom=bars, label="Imperfect", color="orange")
    plt.ylabel("Count")
    plt.title("Message-level success")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"raid_test_{trials}_trials.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    print("Building Huffman tree...")
    codebook, bigrams, tree = fc.huffman_init("../../main/huffman/WarAndPeace.txt")
    sample_and_plot(codebook, bigrams, tree, trials=200, snippet_len=200)

