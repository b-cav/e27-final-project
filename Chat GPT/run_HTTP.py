import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

def noisy_channel(bits: str) -> str:
    """
    Calls curl to POST the given bits to the test.py endpoint
    and returns the stdout response as a string.
    """
    cmd = [
        "curl",
        "-X", "POST",
        "-d", f"bits={bits}",
        "https://engs27.host.dartmouth.edu/cgi-bin/noisychannel.py"
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (code {result.returncode}):\n{result.stderr}")
    return result.stdout.split("<body>")[1].split("</body>")[0].strip()

# Hamming distance between two bit strings
def hamming_distance(message1: str, message2: str) -> int:
    return sum(1 for a, b in zip(message1, message2) if a != b)


def gen_pattern(pattern_type: str, n: int) -> str:
    if pattern_type == "all0":
        return "0" * n
    elif pattern_type == "all1":
        return "1" * n
    elif pattern_type == "alt01":
        return ("01" * ((n + 1) // 2))[:n]
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

def analyze_once(pattern_type: str, n: int, run_idx: int):
    inp = gen_pattern(pattern_type, n)
    out = noisy_channel(inp)
    if len(out) != len(inp):
        raise ValueError(f"Length mismatch for {pattern_type} N={n} run={run_idx}: sent {len(inp)}, got {len(out)}")
    ham = hamming_distance(inp, out)
    flips_0_to_1 = sum(1 for a, b in zip(inp, out) if a == '0' and b == '1')
    flips_1_to_0 = sum(1 for a, b in zip(inp, out) if a == '1' and b == '0')
    zeros_in = inp.count('0')
    ones_in = inp.count('1')
    return {
        "pattern_type": pattern_type,
        "length": n,
        "run_idx": run_idx,
        "input_bits": inp,
        "output_bits": out,
        "hamming_distance": ham,
        "error_rate": ham / n if n else np.nan,
        "zeros_in": zeros_in,
        "ones_in": ones_in,
        "flips_0_to_1": flips_0_to_1,
        "flips_1_to_0": flips_1_to_0,
        "p_0to1": flips_0_to_1 / zeros_in if zeros_in else np.nan,
        "p_1to0": flips_1_to_0 / ones_in if ones_in else np.nan,
    }

def run_experiments(
    num_runs: int = 1,
    lengths = None,
    pattern_types = ("all0", "all1", "alt01"),
    raw_csv_path: str = None,
    summary_csv_path: str = None,
    show_preview: bool = True
):
    if lengths is None:
        lengths = [2 ** k for k in range(0, 9)]  # 1,2,4,...,256

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if raw_csv_path is None:
        raw_csv_path = f"noisy_channel_raw_{ts}.csv"
    if summary_csv_path is None:
        summary_csv_path = f"noisy_channel_summary_{ts}.csv"

    rows = []
    total = len(pattern_types) * len(lengths) * num_runs
    step = 0

    for ptype in pattern_types:
        for n in lengths:
            for r in range(num_runs):
                step += 1
                print(f"[{step}/{total}] Sending: {ptype}, N={n}, run={r}")
                rows.append(analyze_once(ptype, n, r))

    raw_df = pd.DataFrame(rows).sort_values(by=["pattern_type", "length", "run_idx"]).reset_index(drop=True)
    raw_df.to_csv(raw_csv_path, index=False)

    # Summary stats across runs
    agg = raw_df.groupby(["pattern_type", "length"], as_index=False).agg(
        runs=("run_idx", "count"),
        mean_hamming=("hamming_distance", "mean"),
        std_hamming=("hamming_distance", "std"),
        mean_error_rate=("error_rate", "mean"),
        std_error_rate=("error_rate", "std"),
        mean_p_0to1=("p_0to1", "mean"),
        mean_p_1to0=("p_1to0", "mean"),
    )
    agg.to_csv(summary_csv_path, index=False)

    if show_preview:
        print("Saved raw to:", raw_csv_path)
        print("Saved summary to:", summary_csv_path)
        print("\nSummary preview:")
        print(agg.to_string(index=False))

    return raw_df, agg, raw_csv_path, summary_csv_path

#Example usage:
raw_df, summary_df, raw_path, summary_path = run_experiments(num_runs=100)
#To increase repeats per pattern: run_experiments(num_runs=5)
import numpy as np
import pandas as pd

def prop_diff_z(k1, n1, k2, n2):
    """Two-proportion z statistic (pooled). Returns z, p_pool, se."""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (k1/n1 - k2/n2) / se if se > 0 else np.nan
    return z, p_pool, se

def confirm_markov_effects(raw_df: pd.DataFrame):
    """
    For each (pattern_type, length), compute:
      - overall BER
      - P(flip | prev=0) vs P(flip | prev=1)  (value memory)
      - P(flip | same) vs P(flip | transition) (transition effect)
      - P(error | previous error) vs P(error)  (burstiness)
      - z-scores for key differences
    """
    rows = []
    for (ptype, n), grp in raw_df.groupby(["pattern_type", "length"]):
        total_bits = 0
        total_errs = 0

        # prev value
        prev0_n = prev0_err = 0
        prev1_n = prev1_err = 0

        # transition vs same
        same_n = same_err = 0
        trans_n = trans_err = 0

        # previous error
        eprev1_n = eprev1_err = 0
        eprev0_n = eprev0_err = 0

        for _, r in grp.iterrows():
            x = r["input_bits"]
            y = r["output_bits"]
            if not isinstance(x, str) or not isinstance(y, str) or len(x) != len(y) or len(x) == 0:
                continue
            L = len(x)
            e = [1 if a != b else 0 for a, b in zip(x, y)]

            total_bits += L
            total_errs += sum(e)

            for i in range(1, L):
                prev = x[i-1]
                cur  = x[i]
                erri = e[i]

                if prev == '0':
                    prev0_n += 1; prev0_err += erri
                else:
                    prev1_n += 1; prev1_err += erri

                if prev == cur:
                    same_n += 1; same_err += erri
                else:
                    trans_n += 1; trans_err += erri

                if e[i-1] == 1:
                    eprev1_n += 1; eprev1_err += erri
                else:
                    eprev0_n += 1; eprev0_err += erri

        p = total_errs / total_bits if total_bits else np.nan
        p_prev0 = prev0_err / prev0_n if prev0_n else np.nan
        p_prev1 = prev1_err / prev1_n if prev1_n else np.nan
        p_same  = same_err  / same_n  if same_n  else np.nan
        p_trans = trans_err / trans_n if trans_n else np.nan
        p_e1    = eprev1_err / eprev1_n if eprev1_n else np.nan
        p_e0    = eprev0_err / eprev0_n if eprev0_n else np.nan

        # z-tests (|z| ≳ 2 suggests a real effect)
        z_prev, _, _ = prop_diff_z(prev1_err, prev1_n, prev0_err, prev0_n)
        z_trans, _, _ = prop_diff_z(trans_err, trans_n, same_err, same_n)
        z_burst, _, _ = prop_diff_z(eprev1_err, eprev1_n, eprev0_err, eprev0_n)

        rows.append({
            "pattern_type": ptype,
            "length": n,
            "overall_error": p,
            "p_flip_prev0": p_prev0,
            "p_flip_prev1": p_prev1,
            "z_prev1_minus_prev0": z_prev,
            "p_flip_same": p_same,
            "p_flip_trans": p_trans,
            "z_trans_minus_same": z_trans,
            "p_err_given_prev_err1": p_e1,
            "p_err_given_prev_err0": p_e0,
            "z_prevErr1_minus_prevErr0": z_burst,
            "counts": {
                "prev0": (prev0_err, prev0_n),
                "prev1": (prev1_err, prev1_n),
                "same":  (same_err, same_n),
                "trans": (trans_err, trans_n),
                "prevErr1": (eprev1_err, eprev1_n),
                "prevErr0": (eprev0_err, eprev0_n),
                "total": (total_errs, total_bits)
            }
        })

    out = pd.DataFrame(rows).sort_values(["pattern_type", "length"]).reset_index(drop=True)
    return out

# After data collection:
# raw_df, summary_df, raw_path, summary_path = run_experiments(num_runs=100)
# mk = confirm_markov_effects(raw_df)
# display(mk)