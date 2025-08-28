import heapq 

# Takes in file and counts frequencies of each character in the file
# Planning on sending in War and Peace txt, which is a large sample of the English language
def count_frequencies(pathname):
    freq = {}
    with open(pathname, "r") as f:
        for char in f.read():
            freq[char] = freq.get(char, 0) + 1
    return freq

# Node of a binary tree
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

def build_huffman_tree(pathname):
    freqs = count_frequencies(pathname)

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
#print(count_frequencies("Shep_Testing/Huffman/WarAndPeace.txt"))

def minimize_error(huff_tree):
    # WRITE SOME FUNCTION THAT STARTS AT TOP AND MODIFIES HUFFMAN TREE TO MINIMIZE ERROR
    return

# recursively builds the codebook, which maps characters to their codes 
def build_codebook(node, prefix = "", codebook = None):
    if codebook is None: 
        codebook = {}
    if node.char is not None: # Base case: if we are at a leaf, add the 
        codebook[node.char] = prefix
    else:
        build_codebook(node.left, prefix + "0", codebook)
        build_codebook(node.right, prefix + "1", codebook)
    return codebook

def encode_message(message, codes):
    encoded_message = "".join(codes[ch] for ch in message) # uses generator expression to create encoded message
    return encoded_message

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


if __name__ == "__main__":
    text = "hello world"
    tree = build_huffman_tree("Shep_Testing/Huffman/WarAndPeace.txt")
    codes = build_codebook(tree)
    encoded = encode_message(text, codes)
    
    print("Codes:", codes)
    print("Encoded:", encoded)
    print("Decoded:", decode_message(encoded, tree))