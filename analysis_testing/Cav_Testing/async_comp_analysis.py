# async_comp_analysis.py - 
#
# Ben Cavanagh
# 09-03-2025
# Description: 
#
# analysis/experiment/channel_analysis.py
import argparse
import asyncio
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import aiohttp

# Paths
THIS_FILE = Path(__file__).resolve()  # .../analysis/experiment/channel_analysis.py
PROJECT_ROOT = THIS_FILE.parents[2]   # project/
MAIN_DIR = PROJECT_ROOT / "main"

# Ensure imports
for p in [str(MAIN_DIR), str(THIS_FILE.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from huffman import final_compression as fc
from error_correction import hamming as hm
from error_correction import raid
from edit_distance import edit_distance as ed  # your Levenshtein with Hamming fast-path + Myers

# Schemes and colors
SCHEMES = ["raw_ascii", "huffman_only", "huff_hamm", "huff_hamm_raid"]
COLORS = {
    "raw_ascii": "#1f77b4",
    "huffman_only": "#ff7f0e",
    "huff_hamm": "#2ca02c",
    "huff_hamm_raid": "#d62728",
}

# --------------- utils ---------------
def ascii_bitstring(s: str) -> str:
    return "".join(format(ord(c), "08b") for c in s)

def bitstring_to_ascii(bits: str) -> str:
    n = len(bits)
    if n % 8 != 0:
        bits = bits[: n - (n % 8)]
    return "".join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

def bit_error_rate_from_ascii_strings(orig: str, decoded: str) -> float:
    a = ascii_bitstring(orig)
    b = ascii_bitstring(decoded)
    if not a and not b:
        return 0.0
    n = max(len(a), len(b))
    if len(a) < n: a += "0" * (n - len(a))
    if len(b) < n: b += "0" * (n - len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) / n

def mean_ci(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    vals = np.array(values, dtype=float)
    m = float(np.mean(vals))
    if len(vals) < 2:
        return m, 0.0
    se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
    return m, 1.96 * se

def chunk_string(bits: str, chunk_size: int, count: Optional[int] = None) -> List[str]:
    if count is None:
        count = len(bits) // chunk_size
    return [bits[i*chunk_size:(i+1)*chunk_size] for i in range(count)]

def combined_range(d: Dict[str, List[float]], pad: float = 0.0) -> Optional[Tuple[float, float]]:
    vals = [v for vs in d.values() for v in vs]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return (lo - 0.5, hi + 0.5)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)

def clamp_range_01(r: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if r is None:
        return None
    lo, hi = r
    return (max(0.0, lo), min(1.0, hi))

def means_text(means: Dict[str, float], precision: int = 4) -> str:
    parts = []
    for s in SCHEMES:
        if s in means and not np.isnan(means[s]):
            parts.append(f"{s}={means[s]:.{precision}g}")
    return " | ".join(parts)

# --------------- channel ---------------
async def http_channel(bits: str, session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        async with session.post(url, data={"bits": bits}) as resp:
            text = await resp.text()
            lower = text.lower()
            i, j = lower.find("<body>"), lower.find("</body>")
            if i != -1 and j != -1 and i + 6 <= j:
                return text[i+6:j].strip()
            return text.strip()

def local_channel(bits: str, mode_arg: int) -> str:
    try:
        out = hm.noisy_channel(bits, mode_arg)  # str->str
    except TypeError:
        out = hm.noisy_channel([bits], mode_arg)  # list
    return "".join(out) if isinstance(out, list) else out

# --------------- scheme encode/decode ---------------
def build_bits_and_sent(message: str,
                        codebook,
                        alt_codebook,
                        bigram_list,
                        tree,
                        alt_tree) -> Dict[str, Tuple[str, int, Dict]]:
    result: Dict[str, Tuple[str, int, Dict]] = {}
    # raw ascii
    ascii_bits = ascii_bitstring(message)
    result["raw_ascii"] = (ascii_bits, len(ascii_bits), {})
    # huffman-only
    msg_bi = fc.replace_bigrams(message, bigram_list)
    compressed = fc.compress_message(msg_bi, codebook, alt_codebook)
    result["huffman_only"] = (compressed, len(compressed), {"tree": tree, "alt_tree": alt_tree})
    # huff+hamm
    enc_pkts = hm.EHC_16_11_encode(compressed)
    hh_joined = "".join(enc_pkts)
    result["huff_hamm"] = (hh_joined, len(enc_pkts) * 16, {"n_pkts": len(enc_pkts), "tree": tree, "alt_tree": alt_tree})
    # huff+hamm+raid
    prot_pkts = raid.RAID_protect(enc_pkts)
    hhr_joined = "".join(prot_pkts)
    result["huff_hamm_raid"] = (hhr_joined, len(prot_pkts) * 16, {"n_pkts": len(prot_pkts), "tree": tree, "alt_tree": alt_tree})
    return result

def decode_after_channel(scheme: str,
                         received_bits: str,
                         aux: Dict,
                         tree,
                         alt_tree) -> str:
    if scheme == "raw_ascii":
        return bitstring_to_ascii(received_bits)
    if scheme == "huffman_only":
        return fc.decompress_message(received_bits, tree, alt_tree)
    if scheme == "huff_hamm":
        n_pkts = aux["n_pkts"]
        noisy_pkts = chunk_string(received_bits, 16, count=n_pkts)
        cleaned_pkts = hm.EHC_16_11_clean(noisy_pkts)
        unpadded = hm.remove_padding(cleaned_pkts)
        return fc.decompress_message(unpadded, tree, alt_tree)
    if scheme == "huff_hamm_raid":
        n_pkts = aux["n_pkts"]
        noisy_pkts = chunk_string(received_bits, 16, count=n_pkts)
        try:
            res = raid.RAID_remove(noisy_pkts, 0)
        except TypeError:
            res = raid.RAID_remove(noisy_pkts)
        recovered_pkts = res[0] if isinstance(res, tuple) else res
        cleaned_pkts = hm.EHC_16_11_clean(recovered_pkts)
        unpadded = hm.remove_padding(cleaned_pkts)
        return fc.decompress_message(unpadded, tree, alt_tree)
    raise ValueError(f"Unknown scheme {scheme}")

# --------------- plotting ---------------
def plot_overlaid_histogram(metric_by_scheme: Dict[str, List[float]],
                            title: str,
                            xlabel: str,
                            fname: Path,
                            bins: int = 40,
                            value_range: Optional[Tuple[float, float]] = None,
                            density: bool = False,
                            show_means: bool = True):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # Histograms
    for s in SCHEMES:
        vals = metric_by_scheme[s]
        if not vals:
            continue
        ax.hist(vals, bins=bins, range=value_range, alpha=0.35, label=s, color=COLORS[s], density=density)

    # Set x-limits to the chosen range for fine-grained scaling
    if value_range is not None:
        ax.set_xlim(value_range)

    # Means as small text and vertical lines
    if show_means:
        means = {s: (float(np.mean(metric_by_scheme[s])) if metric_by_scheme[s] else float("nan")) for s in SCHEMES}
        for s in SCHEMES:
            v = means[s]
            if not np.isnan(v):
                ax.axvline(v, color=COLORS[s], linestyle="--", linewidth=2, alpha=0.9)
        ax.text(0.01, 0.98,
                "Means: " + means_text(means, precision=4),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_binary_histogram(metric_by_scheme: Dict[str, List[float]],
                          title: str,
                          xlabel: str,
                          fname: Path,
                          show_means: bool = True):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bins = [-0.5, 0.5, 1.5]

    for s in SCHEMES:
        vals = metric_by_scheme[s]
        if not vals:
            continue
        ax.hist(vals, bins=bins, alpha=0.35, label=s, color=COLORS[s])

    if show_means:
        means = {s: (float(np.mean(metric_by_scheme[s])) if metric_by_scheme[s] else float("nan")) for s in SCHEMES}
        for s in SCHEMES:
            v = means[s]
            if not np.isnan(v):
                ax.axvline(v, color=COLORS[s], linestyle="--", linewidth=2, alpha=0.9)
        ax.text(0.01, 0.98,
                "Means: " + means_text(means, precision=4),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    ax.set_xticks([0, 1], ["Imperfect (0)", "Perfect (1)"])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# --------------- async experiment with incremental plotting and CSV updates ---------------
async def run_mass_test_async(trials: int,
                              msg_lens: List[int],
                              seed: int,
                              ascii_channel_arg: int,
                              fec_channel_arg: int,
                              save_dir: Path,
                              channel_url: Optional[str],
                              http_concurrency: int,
                              process_workers: int,
                              update_every: int):
    rng = random.Random(seed)
    codebook, alt_codebook, bigrams, tree, alt_tree = fc.huffman_init(str(MAIN_DIR / "huffman" / "WarAndPeace.txt"))

    wp_path = MAIN_DIR / "huffman" / "WarAndPeace.txt"
    corpus = wp_path.read_text(encoding="utf-8", errors="ignore")

    ber_results: Dict[str, Dict[int, List[float]]] = {s: {L: [] for L in msg_lens} for s in SCHEMES}
    ed_results: Dict[str, Dict[int, List[float]]] = {s: {L: [] for L in msg_lens} for s in SCHEMES}
    norm_ed_results: Dict[str, Dict[int, List[float]]] = {s: {L: [] for L in msg_lens} for s in SCHEMES}
    perfect_results: Dict[str, Dict[int, List[float]]] = {s: {L: [] for L in msg_lens} for s in SCHEMES}
    sent_bits_results: Dict[str, Dict[int, List[int]]] = {s: {L: [] for L in msg_lens} for s in SCHEMES}
    completed_counts: Dict[int, int] = {L: 0 for L in msg_lens}

    sem = asyncio.Semaphore(http_concurrency) if channel_url else None
    session_ctx = aiohttp.ClientSession() if channel_url else None
    plot_lock = asyncio.Lock()
    executor = ProcessPoolExecutor(max_workers=process_workers) if process_workers > 0 else None

    def agg(values_by_len: Dict[int, List[float]]) -> Tuple[List[int], List[float], List[float]]:
        lens = sorted(values_by_len.keys())
        means, cis = [], []
        for L in lens:
            m, ci = mean_ci(values_by_len[L]); means.append(m); cis.append(ci)
        return lens, means, cis

    def agg_int(values_by_len: Dict[int, List[int]]) -> Tuple[List[int], List[float], List[float]]:
        lens = sorted(values_by_len.keys())
        means, cis = [], []
        for L in lens:
            vals = list(map(float, values_by_len[L])); m, ci = mean_ci(vals); means.append(m); cis.append(ci)
        return lens, means, cis

    def write_summary_csv():
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "summary_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scheme", "msg_len_chars",
                "avg_ber", "ci_ber",
                "avg_edit_distance", "ci_edit_distance",
                "avg_norm_edit_distance", "ci_norm_edit_distance",
                "percent_perfect", "ci_percent_perfect",
                "avg_sent_bits", "ci_sent_bits"
            ])
            for s in SCHEMES:
                Ls, ber_mean, ber_ci = agg(ber_results[s])
                _, ed_mean, ed_ci = agg(ed_results[s])
                _, n_ed_mean, n_ed_ci = agg(norm_ed_results[s])
                _, perf_mean, perf_ci = agg(perfect_results[s])
                _, sb_mean, sb_ci = agg_int(sent_bits_results[s])
                for i, L in enumerate(Ls):
                    writer.writerow([
                        s, L,
                        ber_mean[i], ber_ci[i],
                        ed_mean[i], ed_ci[i],
                        n_ed_mean[i], n_ed_ci[i],
                        perf_mean[i], perf_ci[i],
                        sb_mean[i], sb_ci[i]
                    ])

    # sync plotting with captured state, include mean lines and fine-grained ranges
    def sync_update_plots_for_length(L: int):
        save_dir.mkdir(parents=True, exist_ok=True)
        # BER
        ber_L = {s: ber_results[s][L] for s in SCHEMES}
        rng_ber = clamp_range_01(combined_range(ber_L, pad=0.05))
        plot_overlaid_histogram(
            ber_L,
            title=f"BER Distribution (L={L} Chars)",
            xlabel="Bit Error Rate",
            fname=save_dir / f"hist_ber_L{L}.png",
            bins=80,
            value_range=rng_ber,
            show_means=True,
        )
        # Edit distance
        ed_L = {s: ed_results[s][L] for s in SCHEMES}
        rng_ed = combined_range(ed_L, pad=0.05)
        plot_overlaid_histogram(
            ed_L,
            title=f"Edit Distance Distribution (L={L} Chars)",
            xlabel="Edit Distance (Characters)",
            fname=save_dir / f"hist_edit_distance_L{L}.png",
            bins=60,
            value_range=rng_ed,
            show_means=True,
        )
        # Normalized edit distance [0..1]
        n_ed_L = {s: norm_ed_results[s][L] for s in SCHEMES}
        rng_ned = clamp_range_01(combined_range(n_ed_L, pad=0.05))
        plot_overlaid_histogram(
            n_ed_L,
            title=f"Normalized Edit Distance Distribution (L={L} Chars)",
            xlabel="Normalized Edit Distance",
            fname=save_dir / f"hist_norm_edit_distance_L{L}.png",
            bins=80,
            value_range=rng_ned,
            show_means=True,
        )
        # Perfect vs imperfect
        perf_L = {s: perfect_results[s][L] for s in SCHEMES}
        plot_binary_histogram(
            perf_L,
            title=f"Perfect Message Indicator (L={L} Chars)",
            xlabel="Outcome",
            fname=save_dir / f"hist_perfect_L{L}.png",
            show_means=True,
        )
        # Sent bits
        sb_L = {s: list(map(float, sent_bits_results[s][L])) for s in SCHEMES}
        rng_sb = combined_range(sb_L, pad=0.05)
        plot_overlaid_histogram(
            sb_L,
            title=f"Sent Bits Distribution (L={L} Chars)",
            xlabel="Bits Sent Over Channel",
            fname=save_dir / f"hist_sent_bits_L{L}.png",
            bins=40,
            value_range=rng_sb,
            show_means=True,
        )

    async def update_plots_and_csv_for_length(L: int):
        async with plot_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sync_update_plots_for_length, L)
            await loop.run_in_executor(None, write_summary_csv)

    async def process_one_message(msg: str, L: int):
        bits_map = build_bits_and_sent(msg, codebook, alt_codebook, bigrams, tree, alt_tree)

        # channel per scheme
        tasks = {}
        for scheme, (bits, _, _) in bits_map.items():
            if channel_url:
                tasks[scheme] = asyncio.create_task(http_channel(bits, session_ctx, channel_url, sem))
            else:
                mode_arg = ascii_channel_arg if scheme in ("raw_ascii", "huffman_only") else fec_channel_arg
                loop = asyncio.get_running_loop()
                tasks[scheme] = loop.run_in_executor(None, local_channel, bits, mode_arg)

        received: Dict[str, str] = {scheme: await t for scheme, t in tasks.items()}

        # decode and metrics except ED
        decoded: Dict[str, str] = {}
        for scheme, (_, sent_len, aux) in bits_map.items():
            decoded_msg = decode_after_channel(scheme, received[scheme], aux, tree, alt_tree)
            decoded[scheme] = decoded_msg
            ber_results[scheme][L].append(bit_error_rate_from_ascii_strings(msg, decoded_msg))
            perfect_results[scheme][L].append(1.0 if decoded_msg == msg else 0.0)
            sent_bits_results[scheme][L].append(sent_len)

        # edit distance in process pool (and normalized by original length)
        loop = asyncio.get_running_loop()
        ed_futs = {s: loop.run_in_executor(executor, ed, msg, decoded[s]) for s in SCHEMES}
        for s in SCHEMES:
            ed_val = float(await ed_futs[s])
            ed_results[s][L].append(ed_val)
            norm = ed_val / len(msg) if len(msg) > 0 else 0.0
            norm_ed_results[s][L].append(norm)

        # incremental plotting + CSV
        completed_counts[L] += 1
        if update_every > 0 and (completed_counts[L] % update_every == 0):
            await update_plots_and_csv_for_length(L)

    try:
        if channel_url:
            async with session_ctx:
                for L in msg_lens:
                    print(f"- Message length: {L} chars")
                    messages = []
                    if len(corpus) <= L:
                        messages = [corpus] * trials
                    else:
                        for _ in range(trials):
                            start = rng.randrange(0, len(corpus) - L)
                            messages.append(corpus[start:start + L])
                    await asyncio.gather(*(process_one_message(m, L) for m in messages))
                    await update_plots_and_csv_for_length(L)
        else:
            for L in msg_lens:
                print(f"- Message length: {L} chars")
                messages = []
                if len(corpus) <= L:
                    messages = [corpus] * trials
                else:
                    for _ in range(trials):
                        start = rng.randrange(0, len(corpus) - L)
                        messages.append(corpus[start:start + L])
                for m in messages:
                    await process_one_message(m, L)
                await update_plots_and_csv_for_length(L)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=False)

# --------------- CLI ---------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Mass test with incremental plotting + CSV: Raw ASCII vs Huffman-only vs Hamming vs Hamming+RAID."
    )
    ap.add_argument("--trials", type=int, default=200, help="Trials per message length.")
    ap.add_argument("--msg_lens", type=str, default="50,500", help="Comma-separated message lengths (chars).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling messages.")
    ap.add_argument("--ascii_channel_arg", type=int, default=1, help="Arg to hm.noisy_channel for raw/huffman-only.")
    ap.add_argument("--fec_channel_arg", type=int, default=0, help="Arg to hm.noisy_channel for FEC schemes.")
    ap.add_argument("--channel_url", type=str, default="http://10.135.164.86:8080", help="HTTP channel URL (if set, use aiohttp instead of local).")
    ap.add_argument("--http_concurrency", type=int, default=200, help="Max concurrent HTTP requests.")
    ap.add_argument("--process_workers", type=int, default=os.cpu_count() or 2, help="Workers for edit distance.")
    ap.add_argument("--update_every", type=int, default=5, help="Update plots and CSV every N completed trials per length.")
    ap.add_argument("--outdir", type=str, default=str(THIS_FILE.parent / "comp_analysis_plots"),
                    help="Output directory for plots and CSV.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.outdir)
    msg_lens = [int(x.strip()) for x in args.msg_lens.split(",") if x.strip()]
    asyncio.run(
        run_mass_test_async(
            trials=args.trials,
            msg_lens=msg_lens,
            seed=args.seed,
            ascii_channel_arg=args.ascii_channel_arg,
            fec_channel_arg=args.fec_channel_arg,
            save_dir=out_dir,
            channel_url=args.channel_url or None,
            http_concurrency=args.http_concurrency,
            process_workers=args.process_workers,
            update_every=args.update_every,
        )
    )
