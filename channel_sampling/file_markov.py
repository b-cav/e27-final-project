"""
file_markov.py

ENGS 27 Final Project Mass HTTP Sampling
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

Estimate Markov bit flip probabilities/model given I/O data
"""

import csv
import numpy as np

# Probability of a single bit flipping
def first_bit_flip(thou=100) :
    fbf = [0, 0]

    for bit in [0, 1] :
        with open(f"./results_{str(thou)}K/{str(bit)}_results.csv", "r") as file :
            for row in file :
                if row.rstrip() != str(bit) :
                    fbf[bit] += 1

        fbf[bit] = fbf[bit]/(thou*1000)

    print(f"Pr[fbf | b0 = \"0\"] = {fbf[0]}")
    print(f"Pr[fbf | b0 = \"1\"] = {fbf[1]}")

# Probability of a bit flipping at each position
def pos_flips(code, thou=1000) :
    n = len(code)
    noisy_outputs = np.loadtxt(f"./cont_results_{str(thou)}K/{str(code)}_results.csv", dtype=str)
    bitflips = np.zeros(n, dtype=int)

    for sample in noisy_outputs :
        for i in range(n) :
            if code[i] != sample[i] :
                bitflips[i] += 1

    return bitflips/len(noisy_outputs)

# Print positional flip probabilities
def save_pos_flips(codes) :
    writer = csv.writer(open("./prob_results/pos_flips.csv", "w", newline=""))
    writer.writerow(["Code"] + [f"Pos {i+1}" for i in range(max(len(code) for code in codes))])

    for code in codes :
        probs = pos_flips(code)
        row = [code] + [f"{prob:.4f}" for prob in probs]
        writer.writerow(row)

def cond_flips(codes, prevs, dynamic = False, thou = 1000) :
    # 3-dim matrix where each entry (i,j, k)
    # --> indexes to codeword i, bit value j = 0/1, prev k
    # --> contains the probability of a bit flip in that situation

    flips_detected = np.zeros((len(codes), 2, len(prevs)), dtype=int)
    case_counts = np.zeros((len(codes), 2, len(prevs)), dtype=int)

    for i in range(len(codes)) :
        code = codes[i]
        n = len(code)
        noisy_outputs = np.loadtxt(f"./cont_results_{str(thou)}K/{str(code)}_results.csv", dtype=str)

        for sample in noisy_outputs :
            sample = str(sample.strip())
            for j in range(n) :
                curr_bit = sample[j]
                for k in range(len(prevs)) :
                    prev = str(prevs[k].strip())
                    prevlen = len(prev)
                    flag = False

                    if j >= prevlen :
                        # Dynamic checking calculates probabilities on received values
                        # (i.e. markov based on flipped bits) otherwise check sent value
                        if dynamic == False and str(code[j-prevlen:j].strip()) == prev :
                            flag = True
                        elif dynamic == True and sample[j-prevlen:j] == prev :
                            flag = True

                    if flag == True :
                        case_counts[i, int(code[j]), k] += 1
                        if curr_bit != code[j] :
                            flips_detected[i, int(code[j]), k] += 1

    with np.errstate(divide = "ignore", invalid = "ignore") :
        probs = np.where(case_counts > 0, flips_detected/case_counts, None)
    return(probs)

def print_cond_probs(codes, prevs, dynamic):
    probs = cond_flips(codes, prevs, dynamic)
    if dynamic == True :
        ofile = open("./prob_results/1M_dynamic_cond_probs.csv", "w", newline="")
    else :
        ofile = open("./prob_results/1M_static_cond_probs.csv", "w", newline="")
    for i in range(len(codes)) :
        code = codes[i]
        ofile.write(f"Codeword: {code}\n")
        ofile.write("Prev|  " + "   |   ".join(prevs) + "\n")
        ofile.write("-" * (6 + len(prevs) * 8) + "\n")
        for bit in (0, 1) :
            row = []
            for j in range(len(prevs)) :
                val = probs[i, bit, j]
                if val is None or np.isnan(val) :
                    row.append("  -  ")
                else :
                    row.append(f"{val:0.3f}")
            ofile.write(f" {bit}  | " + " | ".join(row) + "\n\n")

if __name__ == "__main__" :
    #codes = ["000", "001", "010", "011", "100", "101", "110", "111"]
    codes = ["0"*100, "1"*100, "01"*50, "10"*50]
    prevs = ["0", "1", "00", "01", "10", "11"]

    save_pos_flips(codes)
    print_cond_probs(codes, prevs, False)
    print_cond_probs(codes, prevs, True)
