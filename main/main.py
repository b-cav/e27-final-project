"""
main.py

ENGS 27 Final Project Noisy Channel Communication Console
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

"""
from huffman import final_compression as fc
from error_correction import hamming as hm


def console(opt_codebook, bigram_list, opt_tree) :
    message = input("ENTER MESSAGE: ")

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
    # 2) Encode Message:
    compressed = fc.encode_message(fc.replace_bigrams(message, bigram_list), opt_codebook)
    # pass in a list of symbols to encode with bigram codebook

    # ------------------------------------------------
    # 3) Packetize, Add Parity Bits:
    protected = hm.EHC_16_11_encode(compressed) # Splits data into 11 bit packets, pads last one
    # ------------------------------------------------
    # 4) Transmit Packets:
    received_packets = hm.noisy_channel(protected, 0)

    # ------------------------------------------------
    # 5) Error Correct Received Packets:
    corrected = hm.EHC_16_11_decode(received_packets)

    # ------------------------------------------------
    # 6) Remove Padding:
    cleaned = hm.remove_padding(corrected)

    # ------------------------------------------------
    # 7) Decode:
    decoded = fc.decode_message(cleaned, opt_tree)
    print(f"PROTECTED MESSAGE: {decoded}")
   
 
if __name__ == "__main__" :
    print("BUILDING TREE...")
    codebook, bigrams, tree = fc.huffman_init("./huffman/WarAndPeace.txt")
    while True :
        console(codebook, bigrams, tree)
