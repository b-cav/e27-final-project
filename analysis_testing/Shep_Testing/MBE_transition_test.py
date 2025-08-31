"""
main.py

ENGS 27 Final Project Noisy Channel Communication Console
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

This is a test main to show the improvement from reducing bit transitions. Run it 2 times: once with final_compression including the 
transition reduction and once without it and compare teh mean number of multi-bit-errors for both.

My results:
With reduction mean mbe: 6.71
Without reduction mean mbe: 6.91
"""
from huffman import transition_test_fc as fc
from error_correction import hamming as hm

   
multibit_count = 0
 
if __name__ == "__main__" :
    print("BUILDING TREE...")
    codebook, bigrams, tree = fc.huffman_init("./huffman/WarAndPeace.txt")
    compressed = fc.encode_message(fc.replace_bigrams("A paragraph is a self-contained unit of writing that discusses a single main idea or topic, consisting of a series of related sentences. A well-structured paragraph includes a topic sentence that states the main idea, supporting sentences that develop that idea with details and evidence, and a concluding sentence that provides closure or transitions to the next paragraph. Paragraphs are essential for organizing writing, guiding the reader through an essay's main points, and demonstrating the structure of an argument."
, bigrams), codebook)
    protected = hm.EHC_16_11_encode(compressed) # Splits data into 11 bit packets, pads last one
    for i in range(0,100):
        received_packets = hm.noisy_channel(protected, 0)
        corrected, multibit_count = hm.EHC_16_11_decode(received_packets, multibit_count)
        cleaned = hm.remove_padding(corrected)
        decoded = fc.decode_message(cleaned, tree)
# mean errors over 100 trials
print(multibit_count/100) 