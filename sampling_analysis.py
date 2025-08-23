# -*- coding: utf-8 -*-
"""
sampling_analysis.py

ENGS 27 Final Project Sampling Analysis
Ben Sheppard, Joshua Johnson, Ben Cavanagh

The following code interfaces with the noisy channel, hosted on a Dartmouth website.
"""

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



print(noisy_channel("000101010101010101010100010101010101010101010101010101010101010101010"))

def hamming_distance(str1, str2) :
    dist = 0

    if len(str1) != len(str2) :
        print("Different string lengths")
    else :
        for char1, char2 in zip(str1, str2):
            if char1 != char2 :
                dist += 1
    return(dist)

def avg_dist(len, times) :
    zeros = '0' * len
    tot = 0
    
    for i in range(times) :
        feedback = noisy_channel(zeros)
        tot += hamming_distance(zeros, feedback)
        # print(f"Input: {zeros}, Output: {feedback}")

    return(tot/times)

n = 100
t = 100
print(f"Average bit flips of {n}-length string run {t} times: {avg_dist(n, t)}")
