"""
main.py

ENGS 27 Final Project Noisy Channel Communication Console
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

"""
from huffman import final_compression as fc
from error_correction import hamming as hm
from error_correction import raid
import os

def console(opt_codebook, bigram_list, opt_tree) :
    message = input("ENTER MESSAGE OR FILE OR EXIT: ")

    # ***********************************************
    # Handle special inputs
    # ***********************************************
    if message.strip() == "EXIT" :
        exit()
    elif message.strip() == "FILE":
        dir = "./test_files"
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        if not files:
            print("No files found in test directory.")
            message = " "
        else:
            print("Available files:")
            for idx, fname in enumerate(files):
                print(f"  {idx}: {fname}")
            try:
                choice = int(input("Select file number: ").strip())
                if 0 <= choice < len(files):
                    with open(os.path.join(dir, files[choice]), "r", encoding="utf-8") as f:
                        message = f.read()
                else:
                    print("Invalid choice, using empty message.")
                    message = " "
            except Exception as e:
                print(f"Invalid input: {e}")
                message = " "
    elif message.strip() == "" :
        message = "EMPTY"
    else :
        message = message.strip()

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
    raid_recovered, lost_stripes, damaged_stripes, stripe_count = raid.RAID_remove(received_packets)

    # ------------------------------------------------
    # 7) Clean Hamming parity bits
    cleaned = hm.EHC_16_11_clean(raid_recovered)

    # ------------------------------------------------
    # 8) Remove padding:
    unpadded = hm.remove_padding(cleaned)

    # ------------------------------------------------
    # 9) Decode:
    decoded = fc.decode_message(unpadded, opt_tree)

    print(f"SENT MESSAGE      : {message}")
    print(f"HUFFMAN COMPRESSED: {compressed}")
    print(f"HAMMING ENDCODED  : {encoded}")
    print(f"RAID PROTECTED    : {protected}")
    print(f"RECEIVED BITS     : {received_packets}")
    print(f"DECODED           : {raid_recovered}")
    print(f"PARITY REMOVED    : {cleaned}")
    print(f"UNPADDED          : {unpadded}")
    print(f"DECOMPRESSED      : {decoded}")
    print(f"PROTECTED MESSAGE : {decoded}")
    print(f"---------------------------")
    print(f"STRIPES DAMAGED: {damaged_stripes}/{stripe_count} = {100*(damaged_stripes/stripe_count):.3f}%")
    print(f"STRIPES LOST: {lost_stripes}/{stripe_count} = {100*(lost_stripes/stripe_count):.3f}%")
    print("Damaged stripe means multiple multi-bit errors in a stripe of 4 data packets, 1 parity packet")
    print(f"---------------------------\n")
 
if __name__ == "__main__" :
    print("BUILDING TREE...")
    codebook, bigrams, tree = fc.huffman_init("./huffman/WarAndPeace.txt")
    while True :
        console(codebook, bigrams, tree)
