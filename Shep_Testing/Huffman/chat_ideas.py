import heapq 
from collections import Counter
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

def minimize_flips_char(huff_tree):
    # WRITE SOME FUNCTION THAT STARTS AT TOP AND MODIFIES HUFFMAN TREE TO MINIMIZE ERROR
    init_swap_tree = huff_tree
    
    # In the case where there are no preceding characters, we want to put higher frequency
    # child on the left (for start of message 0) since 0 has lower probability of flipping
    if(init_swap_tree.right.freq > init_swap_tree.left.freq):
        swap_children(init_swap_tree)
    
    # call helper on left subtree starting with a 0
    left_subtree = minimize_error_helper(init_swap_tree.left, 0)

    # call helper on right subtree starting with a 1
    right_subtree = minimize_error_helper(init_swap_tree.right, 1)
    final_tree = BTNode(None, left_subtree.freq + right_subtree.freq)
    final_tree.left = left_subtree
    final_tree.right = right_subtree
    return final_tree

def minimize_error_helper(tree, preceding_bit):
    # Base case, you are done
    if (tree.left is None and tree.right is None):
        return tree
    if preceding_bit == 0:
        if(tree.left is not None and tree.right is not None):
            if(tree.right.freq > tree.left.freq):
                swap_children(tree) # put higher frequency one with 0 if preceding bit is 0
            tree.left = minimize_error_helper(tree.left, preceding_bit)
            tree.right = minimize_error_helper(tree.right, not preceding_bit)
    elif preceding_bit == 1:
        if(tree.left is not None and tree.right is not None):
            if(tree.left.freq > tree.right.freq):
                swap_children(tree)
            tree.left = minimize_error_helper(tree.left, not preceding_bit)
            tree.right = minimize_error_helper(tree.right, preceding_bit)
    return tree
# swap chidren of a node, return that node
def swap_children(node):    
    temp = node.left
    node.left = node.right
    node.right = temp
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
    encoded_message = "".join(codes[ch] for ch in message if ch in codes) # skip chars that aren't in training set
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

def count_bit_flips(bits):
    return sum(b1 != b2 for b1, b2 in zip(bits, bits[1:]))

def most_common_transitions(pathname, top_n=20):
    with open(pathname, "r") as f:
        text = f.read()
    # Only consider letters (you can change this if you want all characters)
    transitions = [(text[i], text[i+1]) for i in range(len(text)-1)]    
    counter = Counter(transitions)
    for (a, b), freq in counter.most_common(top_n):
        print(f"{a}->{b}: {freq}")



def readfile(pathname):
    with open(pathname, "r", encoding="utf-8-sig") as f:
        text = f.read()
    return text
def compute_subtree_freqs(node):
    # Ensure node.freq is the sum of leaf freqs in this subtree
    if node is None:
        return 0
    if node.left is None and node.right is None:
        # leaf: node.freq already set by build_huffman_tree
        return node.freq
    node.freq = compute_subtree_freqs(node.left) + compute_subtree_freqs(node.right)
    return node.freq

def orient_left_heavy(node):
    if node is None or (node.left is None and node.right is None):
        return
    # Put higher-frequency subtree on the left (bit 0)
    if node.right is not None and node.left is not None and node.right.freq > node.left.freq:
        swap_children(node)
    orient_left_heavy(node.left)
    orient_left_heavy(node.right)

def clone_tree(node):
    if node is None:
        return None
    new = BTNode(node.char, node.freq)
    new.left = clone_tree(node.left)
    new.right = clone_tree(node.right)
    return new

def mirror_tree(node):
    if node is None:
        return None
    new = BTNode(node.char, node.freq)
    new.left = mirror_tree(node.right)
    new.right = mirror_tree(node.left)
    return new

def encode_choose_mirrored(message, base_tree):
    # Original
    codes0 = build_codebook(base_tree)
    bits0 = encode_message(message, codes0)
    flips0 = count_bit_flips(bits0)
    # Mirrored
    mirrored = mirror_tree(base_tree)
    codes1 = build_codebook(mirrored)
    bits1 = encode_message(message, codes1)
    flips1 = count_bit_flips(bits1)
    # Choose
    if flips1 < flips0:
        return "1" + bits1, True  # header '1' => mirrored
    else:
        return "0" + bits0, False  # header '0' => original

def decode_choose_mirrored(encoded_bits, base_tree):
    if not encoded_bits:
        return ""
    flag = encoded_bits[0]
    payload = encoded_bits[1:]
    tree = mirror_tree(base_tree) if flag == "1" else base_tree
    return decode_message(payload, tree)
def invert_bits(bits):
    return "".join("1" if b == "0" else "0" for b in bits)

def encode_global_invert(message, codes, start_prev_bit=0):
    bits = encode_message(message, codes)
    if not bits:
        return "0"  # nothing, flag 0
    # Compare boundary flip cost; internal flips are identical under inversion
    first = 1 if bits[0] == "1" else 0
    first_inv = 1 - first
    boundary0 = 1 if start_prev_bit != first else 0
    boundary1 = 1 if start_prev_bit != first_inv else 0
    if boundary1 < boundary0:
        return "1" + invert_bits(bits)  # inverted, flag=1
    else:
        return "0" + bits  # original, flag=0

def decode_global_invert(encoded_bits, tree):
    if not encoded_bits:
        return ""
    flag = encoded_bits[0]
    payload = encoded_bits[1:]
    if flag == "1":
        payload = invert_bits(payload)
    return decode_message(payload, tree)
def encode_bus_invert_blocks(bits, block_size=256, start_prev_bit=0):
    out = []
    prev = start_prev_bit
    for i in range(0, len(bits), block_size):
        block = bits[i:i+block_size]
        if not block:
            break
        first = 1 if block[0] == "1" else 0
        # If boundary would flip, invert this block
        if prev != first:
            out.append("1")               # flag: inverted
            inv = invert_bits(block)
            out.append(inv)
            prev = 1 if inv[-1] == "1" else 0
        else:
            out.append("0")               # flag: not inverted
            out.append(block)
            prev = 1 if block[-1] == "1" else 0
    return "".join(out)

def decode_bus_invert_blocks(encoded_bits, tree, block_size=256, start_prev_bit=0):
    # We need to reconstruct blocks with flags, then Huffman-decode the concatenated payload.
    i = 0
    prev = start_prev_bit
    recovered = []
    while i < len(encoded_bits):
        flag = encoded_bits[i]; i += 1
        block = encoded_bits[i:i+block_size]; i += len(block)
        if flag == "1":
            block = invert_bits(block)
        if not block:
            break
        recovered.append(block)
        prev = 1 if block[-1] == "1" else 0
        if len(block) < block_size:
            # last partial block
            break
    payload = "".join(recovered)
    return decode_message(payload, tree)
def diff_encode(bits, start_prev_bit=0):
    out = []
    prev = start_prev_bit
    for ch in bits:
        b = 1 if ch == "1" else 0
        d = b ^ prev
        out.append("1" if d == 1 else "0")
        prev = b
    return "".join(out)

def diff_decode(diffs, start_prev_bit=0):
    out = []
    prev = start_prev_bit
    for ch in diffs:
        d = 1 if ch == "1" else 0
        b = d ^ prev
        out.append("1" if b == 1 else "0")
        prev = b
    return "".join(out)

def encode_with_diff(message, codes, start_prev_bit=0):
    bits = encode_message(message, codes)
    # Optionally choose start_prev_bit to match the first bit to avoid initial flip
    return diff_encode(bits, start_prev_bit=start_prev_bit)

def decode_with_diff(encoded_diffs, tree, start_prev_bit=0):
    bits = diff_decode(encoded_diffs, start_prev_bit=start_prev_bit)
    return decode_message(bits, tree)

def test(message):
    tree_basic = build_huffman_tree("Shep_Testing/Huffman/WarAndPeace.txt")
    codes_basic = build_codebook(tree_basic)
    encoded_basic = encode_message(message, codes_basic)
    print("Baseline")
    print("Num Chars:", len(message))
    print("Num Bits:", len(encoded_basic))
    print("Num Bit Flips:", count_bit_flips(encoded_basic))

    tree_opt = clone_tree(tree_basic)
    minimize_flips_char(tree_opt)
    codes_opt = build_codebook(tree_opt)
    encoded_opt = encode_message(message, codes_opt)
    print("Minimize intra-codeword flips (same lengths)")
    print("Num Chars:", len(message))
    print("Num Bits:", len(encoded_opt))
    print("Num Bit Flips:", count_bit_flips(encoded_opt))

    tree_lh = clone_tree(tree_basic)
    orient_left_heavy(tree_lh)
    codes_lh = build_codebook(tree_lh)
    bits_lh = encode_message(message, codes_lh)
    print("Idea1: Left-heavy orientation")
    print("Num Chars:", len(message))
    print("Num Bits:", len(bits_lh))
    print("Num Bit Flips:", count_bit_flips(bits_lh))

    bits_mirror = encode_choose_mirrored(message, tree_basic)
    payload_mirror = bits_mirror[1:]
    print("Idea2: Choose mirrored (1-bit flag)")
    print("Num Chars:", len(message))
    print("Num Bits:", len(bits_mirror))
    print("Num Bit Flips:", count_bit_flips(payload_mirror))

    bits_inv = encode_global_invert(message, codes_basic, start_prev_bit=0)
    payload_inv = bits_inv[1:]
    print("Idea3: Global invert (1-bit flag)")
    print("Num Chars:", len(message))
    print("Num Bits:", len(bits_inv))
    print("Num Bit Flips:", count_bit_flips(payload_inv))

    bits_base = encoded_basic
    bits_bus = encode_bus_invert_blocks(bits_base, block_size=256, start_prev_bit=0)
    # strip flags for flip count
    payload_blocks = []
    i = 0
    while i < len(bits_bus):
        if i >= len(bits_bus): break
        flag = bits_bus[i]; i += 1
        block = bits_bus[i:i+256]; i += len(block)
        payload_blocks.append(block)
        if len(block) < 256: break
    payload_bus = "".join(payload_blocks)
    print("Idea4: Bus-invert blocks (+1 bit per block)")
    print("Num Chars:", len(message))
    print("Num Bits:", len(bits_bus))
    print("Num Bit Flips:", count_bit_flips(payload_bus))

    bits_diff = encode_with_diff(message, codes_basic, start_prev_bit=0)
    print("Idea5: Differential bitstream")
    print("Num Chars:", len(message))
    print("Num Bits:", len(bits_diff))
    print("Num Bit Flips:", count_bit_flips(bits_diff))

if __name__ == "__main__":
    test_message = readfile("Shep_Testing/Huffman/book.txt")
    test(test_message)