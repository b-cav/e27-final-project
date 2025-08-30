"""
hamming.py

ENGS 27 Final Project Hamming Error Correction
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

(16,11) extended hamming codes
"""

import requests

def packetize(message, n = 11) :
    if not all(bit in "01" for bit in message) :
        raise ValueError(f"not bin string")
    message = str(message).strip()
    remainder = len(message)%n
    pad_len = 0
    if  remainder != 0 :
        pad_len = n-remainder
        message += "0"*(pad_len)

    packets = [message[i*n:i*n+n] for i in range(len(message)//n)]

    # stop packet which indicates how many bits were padded
    packets.append(format(pad_len, "011b"))
    print(f"Padding {pad_len} 0s")
    return(packets)

def EHC_16_11_encode(message) :
    data_packets = packetize(message, 11)
    p_bits = [0, 1, 2, 4, 8]
    coded_packets = []
    for packet in data_packets :
        coded = []
        j = 0

        # package with parity bit placeholders
        for i in range(16) :
            if i in p_bits :
                coded.append("0")
            else :
                coded.append(packet[j])
                j += 1

        for p in p_bits[1:] :
            par = 0
            for i in range(1, 16) :
                # bitwise and b/c powers of 2 only "1" in one place
                if i!= p and i & p and coded[i] == "1" :
                    par ^= 1
            coded[p] = str(par)

        ext_par = 0
        for i in range(1, 16) :
            if coded[i] == "1" :
                ext_par ^= 1
        coded[0] = str(ext_par)

        coded_packets.append("".join(coded))

    return(coded_packets)

def EHC_16_11_decode(data_packets) :
    p_bits = [0, 1, 2, 4, 8]
    decoded_bits = []
    for packet in data_packets :
        # break up cause python strings immut
        expanded = []
        for i in range(16) :
            expanded.append(packet[i])

        # Norm parity bits to locate source of error
        error_loc = 0
        for p in p_bits[1:] : # ignore extension bit
            par = 0
            for i in range(1, 16) :
                if i & p and expanded[i] == "1" :
                    par ^= 1
            if par == 1 :
                error_loc += p

        # p0 parity bit to check for double error
        ext_par = 0
        for i in range(1, 16) :
            if expanded[i] == "1" :
                ext_par ^= 1

        # Check if extension parity bit doesnt match
        if ext_par != int(expanded[0]) :
            expanded[error_loc] = str(int(expanded[error_loc])^1)
        elif error_loc == 0 :
            # Ext bit matches b/c no error
            pass
        else :
            print("!!!!!!!!!!!!!!!")
            print("MULTI-BIT ERROR")
            print("!!!!!!!!!!!!!!!")
            break

        decoded = []
        for i in range(16) :
            if i not in p_bits :
                decoded.append(str(expanded[i]))

        decoded_bits.extend(decoded)

    # Put back into 11 bit packets for padding removal
    decoded_packets = []
    for i in range(0, len(decoded_bits), 11) :
        decoded_packets.append("".join(decoded_bits[i:i+11]))
    return(decoded_packets)

def noisy_channel(data_packets) :
    url = "https://engs27.host.dartmouth.edu/cgi-bin/noisychannel.py"
    outputs = []

    for packet in data_packets :
        rec_data = requests.post(url, data={"bits":packet})
        rec_data = rec_data.text.strip()
        outputs.append(rec_data.split("<body>")[1].split("</body>")[0])

    return(outputs)

def remove_padding(decoded_packets) :
    if len(decoded_packets) <= 1 :
        raise ValueError("Missing stop packet")

    stop_packet = decoded_packets[-1]
    pad_len = int(stop_packet, 2)

    penult_packet = decoded_packets[-2]
    if pad_len > 0 :
        penult_packet = penult_packet[:-pad_len]
        print(f"removing {pad_len} 0s")

    data_packets = decoded_packets[:-2] + [penult_packet]
    return("".join(data_packets))

if __name__ == "__main__" :
    test_messages = ["0"*22, "1"*22, "01"*11, "10"*11, "0"*30, "1"*30, "01"*20, "10"*20]

    for message in test_messages :
        print(f"Testing input: {message}")
        enc = EHC_16_11_encode(message)
        print(f"Sending:  {enc}")
        rec = noisy_channel(enc)
        print(f"Received: {rec}")
        dec = EHC_16_11_decode(rec)
        print(f"Decoded: {"".join(dec)}")
        clean = remove_padding(dec)
        print(f"Cleaned: {clean}")

