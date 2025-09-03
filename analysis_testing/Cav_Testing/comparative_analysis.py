# comparative_analysis.py - 
#
# Ben Cavanagh
# 09-02-2025
# Description: 
#

# analysis/experiment/channel_analysis.py
import argparse
import random
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

# Paths
THIS_FILE = Path(__file__).resolve()  # .../analysis/experiment/channel_analysis.py
PROJECT_ROOT = THIS_FILE.parents[2]   # project/
MAIN_DIR = PROJECT_ROOT / "main"

# Ensure `main` is importable
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

# Project imports
try:
    from huffman import final_compression as fc
    from error_correction import hamming as hm
    from error_correction import raid
except Exception as e:
    raise RuntimeError(
        "Failed to import project modules. Ensure this script is at analysis/experiment/, "
        "and that your project structure is correct."
    ) from e


# ---------------------------
# Utility helpers
# ---------------------------

def ascii_bitstring(s: str) -> str:
    return "".join(format(ord(c), "08b") for c in s)


def bitstring_to_ascii(bits: str) -> str:
    n = len(bits)
    if n % 8 != 0:
        bits = bits[: n - (n % 8)]
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return "".join(chars)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            ))
        prev = curr
    return prev[-1]


def bit_error_rate_from_ascii_strings(orig: str, decoded: str) -> float:
    a = ascii_bitstring(orig)
    b = ascii_bitstring(decoded)
    if not a and not b:
        return 0.0
    n = max(len(a), len(b))
    if len(a) < n:
        a = a + "0" * (n - len(a))
    if len(b) < n:
        b = b + "0" * (n - len(b))
    errors = sum(1 for i in range(n) if a[i] != b[i])
    return errors / n


# ---------------------------
# Pipelines using the project's channel (HTTP-backed hm.noisy_channel)
# ---------------------------

def run_raw_ascii_project(message: str, ascii_channel_arg: int) -> str:
    # Match main.py: unprotected baseline
    bits = ascii_bitstring(message)
    noisy_list = hm.noisy_channel([bits], ascii_channel_arg)
    noisy_bits = noisy_list[0] if noisy_list else ""
    return bitstring_to_ascii(noisy_bits)


def run_huffman_hamming_project(message: str,
                                codebook,
                                bigram_list,
                                tree,
                                fec_channel_arg: int) -> str:
    try:
        compressed = fc.encode_message(fc.replace_bigrams(message, bigram_list), codebook)
        encoded_pkts = hm.EHC_16_11_encode(compressed)
        noisy_pkts = hm.noisy_channel(encoded_pkts, fec_channel_arg)
        cleaned_pkts = hm.EHC_16_11_clean(noisy_pkts)
        unpadded = hm.remove_padding(cleaned_pkts)
        decoded = fc.decode_message(unpadded, tree)
        return decoded
    except Exception:
        return ""


def run_huffman_hamming_raid_project(message: str,
                                     codebook,
                                     bigram_list,
                                     tree,
                                     fec_channel_arg: int) -> str:
    try:
        compressed = fc.encode_message(fc.replace_bigrams(message, bigram_list), codebook)
        encoded_pkts = hm.EHC_16_11_encode(compressed)
        protected_pkts = raid.RAID_protect(encoded_pkts)
        noisy_pkts = hm.noisy_channel(protected_pkts, fec_channel_arg)
        recovered_pkts, lost, count = raid.RAID_remove(noisy_pkts)
        cleaned_pkts = hm.EHC_16_11_clean(recovered_pkts)
        unpadded = hm.remove_padding(cleaned_pkts)
        decoded = fc.decode_message(unpadded, tree)
        return decoded
    except Exception:
        return ""


# ---------------------------
# Data generation
# ---------------------------

def load_training_text() -> str:
    wp_path = MAIN_DIR / "huffman" / "WarAndPeace.txt"
    with open(wp_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def sample_message_from_corpus(corpus: str, length: int, rng: random.Random) -> str:
    if len(corpus) <= length:
        return corpus
    start = rng.randrange(0, len(corpus) - length)
    return corpus[start:start + length]


# ---------------------------
# Experiment (project channel; no probability sweep)
# ---------------------------

def run_experiments_project(trials: int,
                            msg_len: int,
                            seed: int,
                            ascii_channel_arg: int,
                            fec_channel_arg: int,
                            save_dir: Path) -> None:
    rng = random.Random(seed)

    print("Building Huffman codebook/tree...")
    codebook, bigrams, tree = fc.huffman_init(str(MAIN_DIR / "huffman" / "WarAndPeace.txt"))
    corpus = load_training_text()

    schemes = ["raw_ascii", "huff_hamm", "huff_hamm_raid"]
    ber_acc: Dict[str, List[float]] = {s: [] for s in schemes}
    edit_acc: Dict[str, List[float]] = {s: [] for s in schemes}
    perf_acc: Dict[str, List[float]] = {s: [] for s in schemes}

    print("Running trials using project hm.noisy_channel (HTTP-backed)...")
    for t in range(trials):
        msg = sample_message_from_corpus(corpus, msg_len, rng)

        out_ascii = run_raw_ascii_project(msg, ascii_channel_arg)
        ber_acc["raw_ascii"].append(bit_error_rate_from_ascii_strings(msg, out_ascii))
        edit_acc["raw_ascii"].append(levenshtein(msg, out_ascii))
        perf_acc["raw_ascii"].append(1.0 if out_ascii == msg else 0.0)

        out_hh = run_huffman_hamming_project(msg, codebook, bigrams, tree, fec_channel_arg)
        ber_acc["huff_hamm"].append(bit_error_rate_from_ascii_strings(msg, out_hh))
        edit_acc["huff_hamm"].append(levenshtein(msg, out_hh))
        perf_acc["huff_hamm"].append(1.0 if out_hh == msg else 0.0)

        out_hhr = run_huffman_hamming_raid_project(msg, codebook, bigrams, tree, fec_channel_arg)
        ber_acc["huff_hamm_raid"].append(bit_error_rate_from_ascii_strings(msg, out_hhr))
        edit_acc["huff_hamm_raid"].append(levenshtein(msg, out_hhr))
        perf_acc["huff_hamm_raid"].append(1.0 if out_hhr == msg else 0.0)

    # Aggregate
    ber_avg = {s: float(np.mean(ber_acc[s])) for s in schemes}
    edit_avg = {s: float(np.mean(edit_acc[s])) for s in schemes}
    perf_avg = {s: float(np.mean(perf_acc[s])) for s in schemes}

    # Optional: 95% CI error bars
    def mean_ci(values):
        vals = np.array(values, dtype=float)
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * se
        return m, ci95

    ber_ci = {s: mean_ci(ber_acc[s])[1] for s in schemes}
    edit_ci = {s: mean_ci(edit_acc[s])[1] for s in schemes}
    perf_ci = {s: mean_ci(perf_acc[s])[1] for s in schemes}

    save_dir.mkdir(parents=True, exist_ok=True)

    def bar_metric(avg: Dict[str, float], ci: Dict[str, float], ylabel: str, fname: str, ylim=None):
        plt.figure(figsize=(7, 5))
        x = np.arange(len(schemes))
        y = [avg[s] for s in schemes]
        err = [ci[s] for s in schemes]
        colors = ["#444", "#1f77b4", "#2ca02c"]
        plt.bar(x, y, color=colors, yerr=err, capsize=6)
        plt.xticks(x, schemes, rotation=15)
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        out_path = save_dir / fname
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    bar_metric(ber_avg, ber_ci, "Average BER (decoded vs original)", "ber_project_channel.png", ylim=(0, 1))
    bar_metric(edit_avg, edit_ci, "Average edit distance (chars)", "edit_distance_project_channel.png")
    bar_metric(perf_avg, perf_ci, "Percent perfect (fraction)", "perfect_rate_project_channel.png", ylim=(0, 1))

    print("Done.")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare Raw ASCII vs Hamming vs Hamming+RAID using the project's noisy channel."
    )
    ap.add_argument("--trials", type=int, default=200, help="Number of trials.")
    ap.add_argument("--msg_len", type=int, default=128, help="Message length in characters.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (sampling messages).")
    # Match main.py usage:
    # - Raw ASCII baseline used noisy_channel(..., 1)
    # - FEC pipelines used noisy_channel(..., 0)
    ap.add_argument("--ascii_channel_arg", type=int, default=1,
                    help="Second argument to hm.noisy_channel for raw ASCII.")
    ap.add_argument("--fec_channel_arg", type=int, default=0,
                    help="Second argument to hm.noisy_channel for FEC schemes.")
    ap.add_argument(
        "--outdir",
        type=str,
        # Save under ./chat_analysis_plots where '.' is this file's directory (analysis/experiment)
        default=str(THIS_FILE.parent / "chat_analysis_plots"),
        help="Output directory for plots (relative to this script by default)."
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.outdir)
    run_experiments_project(
        trials=args.trials,
        msg_len=args.msg_len,
        seed=args.seed,
        ascii_channel_arg=args.ascii_channel_arg,
        fec_channel_arg=args.fec_channel_arg,
        save_dir=out_dir
    )
