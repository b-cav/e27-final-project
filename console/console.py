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
    # For comparison: unprotected, basic ASCII routine
    message = input("ENTER MESSAGE: ")
    sending = "".join(format(ord(char), "08b") for char in message)
    #print(f"SENDING: {sending}")
    received = noisy_channel([sending])
    #print(f"RECEIVED: {received}")
    ascii_received_packets = []
    for packet in received :
        ascii_received_packets.append("".join(chr(int(packet[i:i+8], 2)) for i in range(0, len(packet), 8)))
    ascii_received_string = "".join(ascii_received_packets)
    print(f"RECEIVED MESSAGE: {ascii_received_string}")

if __name__ == "__main__" :
    while True :
        console()
