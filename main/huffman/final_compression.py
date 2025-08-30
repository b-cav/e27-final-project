# Ben Sheppard, Ben Cavanaugh, and Joshua Johnson
# ENGS 27 Final Project
# Huffman coding with bigrams and transition minimization

# There are two main goals of our code:
# 1) Implement a Huffman coding scheme that uses both single characters and bigrams as symbols. This is more effective than using single characters becuase it reduces the expected bits per unique character.
# 2) Attempt to reduce the number of bit transitoins in the encoded message. This is done by remapping the huffman tree to assign high frequency symbols to codes with few internal transitions.
#    This does not change the expected bits per character or the shape of the tree, but does change which symbols are at which leaves.
# Used ChatGPT to help write code

import heapq 
from collections import Counter

#################################################################################
#IO Helper functions
#################################################################################

# Reads in a file and returns the text as a string
def readfile(pathname):
    with open(pathname, "r") as f:
        text = f.read()
    return text

# Counts the frequency of each character in a file and returns a dictionary
# Planning on sending in War and Peace txt, which is a large sample of the English language
def count_frequencies(pathname):
    freq = {}
    with open(pathname, "r") as f:
        for char in f.read():
            freq[char] = freq.get(char, 0) + 1
    return freq


#################################################################################
# Huffman
#################################################################################

# Class for a binary tree node. Holds a character, frequency of that character, and two children
class BTNode:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    # "Magic" function that allows you to use it in a priority queue
    # We are using a min queue so want to pull out lowest frequency first
    def __lt__(self, other):
        return self.freq < other.freq

# Builds a regular huffman tree from a dictionary (freqs) of character frequencies
# returns the root of the tree (type BTNode)
def build_huff_tree(freqs):
    # Make a priority queue which is heapified by minimum frequency 
    priority_queue = [BTNode(ch, f) for ch, f in freqs.items()]
    heapq.heapify(priority_queue)

    # loop while it is length greater than 1, pulling out two elements and then merging them
    while(len(priority_queue) > 1):
        t1 = heapq.heappop(priority_queue)
        t2 = heapq.heappop(priority_queue)
        tree = BTNode(None, t1.freq + t2.freq) # make new tree with t1 on left, t2 on right
        tree.left = t1
        tree.right = t2
        heapq.heappush(priority_queue, tree) # add the tree back onto the priority queue and loop
    huff_tree = priority_queue[0] # return the the one remaining node, the priority tree
    return huff_tree

# recursively builds a codebook from a huffman tree, which maps characters to their codes 
# returns a dictionary mapping characters to their codes
def build_codebook(tree, prefix = "", codebook = None):
    if codebook is None: 
        codebook = {}
    if tree.char is not None: # Base case: if we are at a leaf, add the 
        codebook[tree.char] = prefix
    else:
        build_codebook(tree.left, prefix + "0", codebook)
        build_codebook(tree.right, prefix + "1", codebook)
    return codebook


# Identifies the top K bigrams from the text in terms of freqeuncy
# returns a list of the K most common bigrams
def top_bigrams(text, K):
    counts = Counter(text[i:i+2] for i in range(len(text)-1)) # uses Counter package to count occurences of bigrams in text
    return list(bg for bg,_ in counts.most_common(K)) # returns the K most common bigrams as a list

# Walks through the text and returns a list of symbols ready for Huffman counting to make into a tree.
# importnt note: overlapping bigrams are always given priority on the left (ex: for the, th is picked over he)
# returns a list of single characters and bigrams which when assembled in order recreate the original text
def replace_bigrams(text, bigram_list):
    out = []
    i = 0
    while i < len(text):
        if i+1 < len(text) and text[i:i+2] in bigram_list: # look at next 2 elements, if in bigram list, add that combination to list
            out.append(text[i:i+2])
            i += 2
        else: # else, just add the character normally
            out.append(text[i])
            i += 1
    return out

# Counts the number of internal bit transitions in a binary code, useful for minimizing transitions
# returns the number of internal bit flips
def internal_flips(code):
    if not code:
        return 0
    flips = 0
    for a, b in zip(code, code[1:]):
        if a != b:
            flips += 1
    return flips

# Given a root of a tree, a code, and a symbol, walk the tree to that code and set the leaf's char to sym (a letter)
# Will be used to remap tree to minimize transitions
def set_leaf_char(root, code, sym):
    # Walk the existing tree along the code's bits and set the leaf's symbol.
    node = root
    for b in code:
        node = node.left if b == '0' else node.right
    node.char = sym

# Remap the tree to minimize internal transitions
# Note: this does not change the shape of the tree or the code lengths, but does change which symbols are at which leaves
# returns the root of the remapped tree
def remap_tree(root, codebook, freqs):

    # Group symbols and codes by length
    syms_by_len = {}
    codes_by_len = {}
    for sym, code in codebook.items():
        L = len(code)
        syms_by_len.setdefault(L, []).append(sym)
        codes_by_len.setdefault(L, []).append(code)

    # Remap per length
    for L in syms_by_len.keys():
        syms_sorted = sorted(syms_by_len[L], key=lambda s: (-freqs.get(s, 0), s))
        codes_sorted = sorted(codes_by_len[L], key=lambda c: (internal_flips(c), c.count('1'), c))
        for sym, code in zip(syms_sorted, codes_sorted):
            set_leaf_char(root, code, sym)

    return root

##############################################################################
# Encoding and decoding functions
##############################################################################

# encodes a message given a particular codebook
# returns the encoded message as a string of bits
def encode_message(message, codes):
    for ch in message:
        if ch not in codes:
            print(f"Warning: No code for character '{ch}'")
    encoded_message = "".join(codes[ch] for ch in message) # uses generator expression to create encoded message
    return encoded_message


# decodes a message. Note, requires the huff_tree for decoding
# returns the decoded message as a string of characters
def decode_message(encoded_message, huff_tree):
    result = [] # holds decoded message
    node = huff_tree
    for bit in encoded_message:
        node = node.left if bit == "0" else node.right # go left if 0, go right if 1
        if node.char is not None:  # once you get to leaf
            result.append(node.char)  # add the correct character to the decoded message
            node = huff_tree # start back at the top for next iteration
    decoded_message = "".join(result)
    return decoded_message

##################################################################################
# Init function for the top-level main
##################################################################################

def huffman_init(training_file) :
    training_text = readfile(training_file)  # Training text for frequencies

    # Make bigram huffman tree
    K = 1000 # number of top bigrams to use
    bigram_list = top_bigrams(training_text, K) # get the top K bigrams, sorted by frequency
    freqs = Counter(replace_bigrams(training_text, bigram_list)) # count the frequencies of the new symbols

    bigram_tree = build_huff_tree(freqs) # build the huffman tree
    bigram_codebook = build_codebook(bigram_tree) # build the codebook

    # Remap the tree to minimize internal transitions
    # Note: this does not change the shape of the tree or the code lengths
    opt_tree = remap_tree(bigram_tree, bigram_codebook, freqs)
    opt_codebook = build_codebook(opt_tree) # build the optimized codebook

    return(opt_codebook, bigram_list, opt_tree)

##################################################################################
# Main function for running the compression
##################################################################################

def main():
    # raw training text and frequencies
    training_text = readfile("Shep_Testing/Huffman/WarAndPeace.txt")  # or reuse 'text' if you set it earlier

    # Make bigram huffman tree
    K = 1000 # number of top bigrams to use
    bigram_list = top_bigrams(training_text, K) # get the top K bigrams, sorted by frequency

    freqs = Counter(replace_bigrams(training_text, bigram_list)) # count the frequencies of the new symbols

    bigram_tree = build_huff_tree(freqs) # build the huffman tree
    bigram_codebook = build_codebook(bigram_tree) # build the codebook

    # # Remap the tree to minimize internal transitions
    # # Note: this does not change the shape of the tree or the code lengths
    opt_tree = remap_tree(bigram_tree, bigram_codebook, freqs)
    opt_codebook = build_codebook(opt_tree) # build the optimized codebook
    
    # Testing (will be replaced by command line input or file for larger texts)
    test_message = readfile('Shep_Testing/Huffman/test_message.txt')
    encoded_test = encode_message(replace_bigrams(test_message, bigram_list), opt_codebook) #pass in a list of symbols to encode with bigram codebook
    print(f"Encoded test message length (bits): {len(encoded_test)}")
    decoded_test = decode_message(encoded_test, opt_tree)
    print(f"Decoded test message matches original: {decoded_test == test_message}")
    
if __name__ == "__main__":
    main()

