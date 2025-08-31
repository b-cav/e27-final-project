"""
main.py

ENGS 27 Final Project Noisy Channel Communication Console
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

"""
from huffman import final_compression as fc
from error_correction import hamming as hm
from error_correction import raid


def console(opt_codebook, bigram_list, opt_tree) :
    message = input("ENTER MESSAGE OR EXIT: ")

    if message == "EXIT" :
        exit()

    # ************************************************
    # For comparison: unprotected, basic ASCII routine
    # ************************************************
    sending = "".join(format(ord(char), "08b") for char in message)
    #print(f"SENDING: {sending}")
    received = hm.noisy_channel([sending], 1)
    #print(f"RECEIVED: {received}")
    ascii_received_packets = []
    for packet in received :
        ascii_received_packets.append("".join(chr(int(packet[i:i+8], 2)) for i in range(0, len(packet), 8)))
    ascii_received_string = "".join(ascii_received_packets)
    print(f"UNPROTECTED MESSAGE: {ascii_received_string}")


    # ************************************************
    # Full error correction and recovery process
    # ************************************************

    # ------------------------------------------------
    # 2) Encode message:
    compressed = fc.encode_message(fc.replace_bigrams(message, bigram_list), opt_codebook)
    # pass in a list of symbols to encode with bigram codebook

    # ------------------------------------------------
    # 3) Packetize, add parity bits:
    encoded = hm.EHC_16_11_encode(compressed) # Splits data into 11 bit packets, pads last one

    # ------------------------------------------------
    # 4) Add RAID P packets:
    protected = raid.RAID_protect(encoded)

    # ------------------------------------------------
    # 5) Transmit packets:
    received_packets = hm.noisy_channel(protected, 0)

    # ------------------------------------------------
    # 6) Recover multi-bit packets, remove RAID P packets
    raid_recovered, lost, count = raid.RAID_remove(received_packets)

    # ------------------------------------------------
    # 7) Clean Hamming parity bits
    cleaned = hm.EHC_16_11_clean(raid_recovered)

    # ------------------------------------------------
    # 8) Remove padding:
    unpadded = hm.remove_padding(cleaned)

    # ------------------------------------------------
    # 9) Decode:
    decoded = fc.decode_message(unpadded, opt_tree)
    print(f"PROTECTED MESSAGE: {decoded}")

if __name__ == "__main__" :
    print("BUILDING TREE...")
    codebook, bigrams, tree = fc.huffman_init("./huffman/WarAndPeace.txt")
    while True :
        console(codebook, bigrams, tree)
