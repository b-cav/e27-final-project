"""
raid.py

ENGS 27 Final Project Packet Protection
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

RAID 5-style P packet (XOR parity) protection/recovery
"""

# Given a bunch of data packets
# Every s packets (stripe) add a parity packet
# Return list of ordered data/parity packets
def RAID_protect(data_packets, n = 16, s = 4) :
    out_packets = []

    i = 0
    while i < len(data_packets) :
        stripe = data_packets[i:i+s] # Adds P to remainder stripe even if size < 4
        out_packets.extend(stripe)

        P = ""
        for j in range(n) :
            p_bit = 0
            for packet in stripe :
                p_bit ^= int(packet[j])
            P += str(p_bit)

        out_packets.append(P)

        i += len(stripe)

    return out_packets

# Given a stream of Hamming-corrected data/P packets
# Recover lost packets and clean out P packets
def RAID_remove(data_packets, n = 16, s = 4) :
    out_packets = []
    stripe_count = 0
    lost_stripes = 0

    i = 0
    while i < len(data_packets) :
        mult_err_loc = -1
        unrecoverable = 0
        stripe = data_packets[i:i+s+1]
        stripe_count += 1
        # Check for multi-bit errors
        # Make sure only one has happened in the stripe
        for j in range(len(stripe)) :
            repaired_16_bit, mult_err = multi_err_detect(stripe[j])
            if mult_err :
                if mult_err_loc == -1 :
                    mult_err_loc = j
                else :
                    unrecoverable = 1
            stripe[j] = repaired_16_bit

        if not unrecoverable :
            out_packets.extend(RAID_recover(stripe, mult_err_loc))
        else :
            out_packets.extend(stripe[:-1])
            """
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("UNRECOVERABLE MULTI-BIT ERROR")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            """
            lost_stripes += 1
        i += (len(stripe))

    return(out_packets, lost_stripes, stripe_count)

# Given a stripe of s data packets and 1 parity packet
# and corrupted packet index i recover packet at index i
# if necessary, return just data packets
def RAID_recover(stripe, i, n = 16, s = 4) :
    # i is multi-error packet location.
    # If i = -1 then no error
    if i != -1 :
        uncorrupted = stripe[:i] + stripe[i+1:]

        recovered = ""
        for j in range(n) :
            p_bit = 0
            for packet in uncorrupted :
                p_bit ^= int(packet[j])
            recovered += str(p_bit)

        return(stripe[:i] + [recovered] + stripe[i+1:-1])
    else :
        return(stripe[:-1])

# Detect multi-bit errors, correct single bit errors, but don't shorten
# Somewhat redundant but wanted black-box link to Hamming decoding
def multi_err_detect(packet, n = 16) :
    multi_err = False
    p_bits = [0, 1, 2, 4, 8]

    if len(packet) != n :
        raise ValueError("Packet length not 16")

    expanded = []
    for i in range(n) :
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
        # 1 error, fix it
        expanded[error_loc] = str(int(expanded[error_loc])^1)
    elif error_loc == 0 :
        # No errors, pass through
        pass
    else :
        """
        print("MULTI-BIT ERROR")
        print("ATTEMPTING TO RECOVER...")
        """
        multi_err = True

    return("".join(expanded), multi_err)

if __name__ == "__main__" :
    packets = ["0"*16, "1"*16, "01"*8, "10"*8]

    print(f"Packets: {packets}")
    protected = RAID_protect(packets)
    print(f"Protected: {protected}")
    for j in range(5) :
        print(RAID_recover(protected, j))

    print(f"RAID removed: {RAID_remove(protected)}")
