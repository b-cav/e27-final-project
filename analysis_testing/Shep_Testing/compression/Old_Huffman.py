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
    with open(pathname, "r") as f:
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

def minimize_intra_codeword_transitions(root):
    # DP over the tree. For each node, compute:
    # - cost0: minimal expected number of bit flips below this node
    #          given that the incoming (previous) bit is 0
    # - cost1: same, given incoming bit is 1
    #
    # At an internal node with children L and R, there are two choices:
    #   no-swap: label left=0, right=1
    #   swap:    label left=1, right=0
    #
    # If incoming bit is b, then taking an edge to a child with label x
    # adds 1 flip for each leaf occurrence in that child if b != x (weighted by subtree freq).
    # The first edge from the root has no "incoming bit", so we handle root specially.

    # Attach attributes for DP and decisions
    class _Wrap:
        __slots__ = ("node", "cost0", "cost1", "dec0", "dec1", "left", "right")
        def __init__(self, node):
            self.node = node
            self.cost0 = 0.0
            self.cost1 = 0.0
            self.dec0 = False  # True means "swap" is better when incoming bit is 0
            self.dec1 = False  # True means "swap" is better when incoming bit is 1

    def dp(node):
        w = _Wrap(node)
        if node.left is None and node.right is None:
            # leaf: no further edges => no internal flips
            w.cost0 = 0.0
            w.cost1 = 0.0
            return w

        WL = dp(node.left)
        WR = dp(node.right)
        FL, FR = node.left.freq, node.right.freq

        # Incoming bit = 0
        # no-swap: left labeled 0, right labeled 1
        cost0_noswap = FR + WL.cost0 + WR.cost1  # flips added on right edges (0->1) + subtree costs
        # swap: left labeled 1, right labeled 0
        cost0_swap   = FL + WL.cost1 + WR.cost0  # flips added on left edges (0->1 after swap) + subtree costs
        if cost0_swap < cost0_noswap:
            w.cost0 = cost0_swap
            w.dec0 = True
        else:
            w.cost0 = cost0_noswap
            w.dec0 = False

        # Incoming bit = 1
        cost1_noswap = FL + WL.cost0 + WR.cost1  # flips added on left edges (1->0) + subtree costs
        cost1_swap   = FR + WL.cost1 + WR.cost0  # flips added on right edges (1->0 after swap) + subtree costs
        if cost1_swap < cost1_noswap:
            w.cost1 = cost1_swap
            w.dec1 = True
        else:
            w.cost1 = cost1_noswap
            w.dec1 = False

        # Attach children wrappers so we can traverse later
        w.left = WL
        w.right = WR
        return w

    def apply_decisions(w, incoming_bit):
        n = w.node
        if n.left is None and n.right is None:
            return
        do_swap = w.dec1 if incoming_bit == 1 else w.dec0
        if do_swap:
            swap_children(n)
        # After (maybe) swapping, the bit labels on outgoing edges are:
        # left_bit = 1 if swapped else 0
        # right_bit = 0 if swapped else 1
        left_bit = 1 if do_swap else 0
        right_bit = 0 if do_swap else 1
        apply_decisions(w.left, left_bit)
        apply_decisions(w.right, right_bit)

    compute_subtree_freqs(root)
    W = dp(root)

    # Handle root specially: first edge has no "incoming bit", so no flip at the first edge.
    # Compare the two top-level choices directly:
    # no-swap at root => left labeled 0, right labeled 1 => cost = WL.cost0 + WR.cost1
    # swap at root    => left labeled 1, right labeled 0 => cost = WL.cost1 + WR.cost0
    cost_root_noswap = W.left.cost0 + W.right.cost1
    cost_root_swap   = W.left.cost1 + W.right.cost0
    if cost_root_swap < cost_root_noswap:
        swap_children(root)
        # After swapping root: incoming bits to children are 1 for left, 0 for right
        apply_decisions(W.left, 1)
        apply_decisions(W.right, 0)
    else:
        # No swap at root: incoming bits 0 for left, 1 for right
        apply_decisions(W.left, 0)
        apply_decisions(W.right, 1)

    return root

def test(message):
    # Build basic tree from war and peace
    tree_basic = build_huffman_tree("Shep_Testing/Huffman/WarAndPeace.txt")
    codes_basic = build_codebook(tree_basic)
    encoded_basic = encode_message(message, codes_basic)

    print("Basic Huffman Compression\n")
    print("Num Bits: " + str(len(encoded_basic)))
    print("Num Bit Flips: " + str(count_bit_flips(encoded_basic)))


    tree1 = minimize_intra_codeword_transitions(tree_basic)
    codes1 = build_codebook(tree1)
    encoded1 = encode_message(message, codes1)
    print("Minimizing the number of bit flips per character without changing code set.\n")
    print("Num Bits: " + str(len(encoded1)))
    print("Num Bit Flips: " + str(count_bit_flips(encoded1)))
    
if __name__ == "__main__":
    # text = "hello"
    # tree = build_huffman_tree("Shep_Testing/Huffman/WarAndPeace.txt")
    # codes = build_codebook(tree)
    # encoded = encode_message(text, codes)
    
    # print("Codes:", codes)
    # print("Encoded:", encoded)
    # print("Decoded:", decode_message(encoded, tree))

    # modified_tree = minimize_error(tree)
    # modified_codes = build_codebook(modified_tree)
    # encoded = encode_message(text, modified_codes)
    
    # print("Codes:", modified_codes)
    # print("Encoded:", encoded)
    # print("Decoded:", decode_message(encoded, modified_tree))

    # Example usage:
    test_message = readfile("Shep_Testing/Huffman/test_message.txt")
    test(test_message)
