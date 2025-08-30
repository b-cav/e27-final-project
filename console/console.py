"""
console.py

ENGS 27 Final Project Noisy Channel Communication Console
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

"""
import requests

# Noisy channel simulator
# Send message packet-by-packet as array of bitstrings
def noisy_channel(data_packets) :
    url = "https://engs27.host.dartmouth.edu/cgi-bin/noisychannel.py"
    outputs = []

    for packet in data_packets :
        rec_data = requests.post(url, data={"bits":packet})
        rec_data = rec_data.text.strip()
        outputs.append(rec_data.split("<body>")[1].split("</body>")[0])

    return(outputs)

def console() :
    message = input("ENTER MESSAGE: ")

    ascii_bin_string = "".join(format(ord(char), "08b") for char in message)
    received = noisy_channel([ascii_bin_string])
    ascii_received_string

    print(f"RECEIVED MESSAGE: {ascii_received_string}")
