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
    # Add one parity packet after every 4 data packets

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

        print(f"Parity packet: {P}")
        out_packets.append(P)

        i += len(stripe)

    print(f"Detected {i} data packets")
    return out_packets

# Given a stripe of s data packets and 1 parity packet
# and corrupted packet index i recover packet at index i
# if necessary, return just data packets
def RAID_recover(stripe, i, n = 16, s = 4) :
    # i is multi-error packet location.
    # If i = s (0-indexed) then no error or error only in the parity packet
    if i != s :
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

if __name__ == "__main__" :
    packets = ["0"*16, "1"*16, "01"*8, "10"*8]

    print(f"Packets: {packets}")
    protected = RAID_protect(packets)
    for j in range(5) :
        print(RAID_recover(protected, j))
