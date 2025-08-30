"""
est_probs.py

ENGS 27 Final Project Sample CSV Processing and Data Analysis
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

"""

import subprocess
import numpy as np
import pandas as pd

def hamming_distance(str1, str2, row) :
    dist = 0
    if len(str1) != len(str2) :
        print(f"Different string lengths at row: {row}")
    else :
        for char1, char2 in zip(str1, str2):
            if char1 != char2 :
                dist += 1
    return(dist)

def avg_dist(code) :
    data = pd.read_csv(f"./results_100K/{code}_results.csv", header=None, dtype=str)
    lengths = np.array([hamming_distance(code, data.iloc[i,0], i) for i in range(len(data))])
    print(f"Avg. H-dist of {code} samples: {np.mean(lengths):.4f}, STD: {np.std(lengths):.4f}")

def transition_probs

if __name__ == "__main__" :
    codes = ["0", "1", "00", "01", "10", "11", "000", "001", "010", "011", "100", "101", "110", "111"]
    for code in codes :
        avg_dist(code)
