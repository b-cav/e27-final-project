import net.datastructures.BinaryTree;

import java.io.*;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Class to compress and decompress txt files with Huffman Encoding. The class has War and Peace and the Constitution as large
 * test files. Three other test files are included: emptyFileTest (which does not compress and throws and Exception because
 * there is nothing to compress), oneCharacterTest (which consists of only one character), and oneCharacterRepeatedTest
 * (which consists of one character repeated many times). There is a debug flagger which is currently set to false.
 * In the main method, it prints the countFrequencies map, the binary tree, and the code map for debugging purposes.
 *
 * @author Tara Salli
 * @author Ben Sheppard
 */
public class HuffmanEncoding implements Huffman{
    public static String emptyFileTest = "inputs/emptyFileTest.txt";
    public static String oneCharacterTest = "inputs/oneCharacterTest.txt";
    public static String oneCharacterRepeatedTest = "inputs/oneCharacterRepeatedTest.txt";
    public static String warAndPeace = "inputs/WarAndPeace.txt";
    public static String constitution = "inputs/USConstitution.txt";
    public static final boolean DEBUG = false; // Debug flag to print countMap/Tree/CodeMap
    /**
     * Read file provided in pathName and count how many times each character appears
     * @param pathName - path to a file to read
     * @return - Map with a character as a key and the number of times the character appears in the file as value
     * @throws IOException
     */
    public Map<Character, Long> countFrequencies(String pathName) throws IOException{
        Map<Character, Long> frequencyTable = new HashMap<Character, Long>(); // Map with a character as a key and the number of times the character appears in the file as value
        BufferedReader input;
        try { //Try to open file
             input = new BufferedReader(new FileReader(pathName));
        }
        catch(FileNotFoundException e) { // Throws exception if file not found// Throws exception if file not found
            throw new FileNotFoundException("File not found\n" + e);
        }
        try { //Try to read character from file
            char currentChar;
            int charInt;
            while ((charInt = input.read()) >= 0) { // While not at end of the document (in which case read() would return -1)
                currentChar = (char) charInt; // Casts the read() int to a character
                if (frequencyTable.containsKey(currentChar)) { // Map already contains the character
                    frequencyTable.put(currentChar, frequencyTable.get(currentChar) + 1); // Add one to the value corresponding to that character in the map.
                } else { // Map doesn't contain character
                    frequencyTable.put(currentChar, (long) 1); // Adds the character to the table and set value to 1.
                }
            }
        }
        catch(IOException e){ // Cannot read from file
            try{
                input.close(); // Tries to close
            }
            catch(IOException e2){
                throw new IOException("Cannot read from file and cannot close file\n" + e + e2 ); //Executes if cannot close
            }
            throw new IOException("Cannot read from the file\n" + e); // Executes if can close but cannot read file
        }
        try{ // Try to close file after map was successfully built
            input.close();
        }
        catch(IOException e){
            throw new IOException("Cannot close file\n" + e); // Can't close file
        }
        return frequencyTable; // Returns the map of chars and counts
    }

    /**
     * Construct a code tree from a map of frequency counts. Note: this code should handle the special
     * cases of empty files or files with a single character.
     *
     * @param frequencies a map of Characters with their frequency counts from countFrequencies
     * @return the code tree.
     */
    public BinaryTree<CodeTreeElement> makeCodeTree(Map<Character, Long> frequencies){
        if(frequencies.isEmpty()){ // If the frequency is empty, return null for the tree (Will be handled in the compressFile method).
            return null;
        }
        PriorityQueue<BinaryTree<CodeTreeElement>> nodes = new PriorityQueue<BinaryTree<CodeTreeElement>>((BinaryTree<CodeTreeElement> node1, BinaryTree<CodeTreeElement> node2) -> (int)(node1.getData().getFrequency() - node2.getData().getFrequency())); // Creates Priority Queue using lambda function to pull out tree with lowest frequency
        for(Character character : frequencies.keySet()){ // Iterate over set of all keys in Map
            nodes.add(new BinaryTree<CodeTreeElement>(new CodeTreeElement(frequencies.get(character), character))); // Adds frequency and character as a BinaryTree holding CodeTreeElement to the PQ.
        }
        if(nodes.size() == 1){ // Case where there is only one distinct character in file
            return new BinaryTree<CodeTreeElement>(new CodeTreeElement(nodes.peek().getData().getFrequency(), '|'), nodes.peek(), null); // Return BinaryTree with the single character in file and its frequency. Includes an interior node (with char of '|') which also has that frequency (total frequency of the one character).
        }
       while(nodes.size() > 1) { // Loops while the nodes has more than one element
            BinaryTree<CodeTreeElement> t1 = nodes.remove(); // Make Binary Tree from the lowest frequency character in the p.q.
            BinaryTree<CodeTreeElement> t2 = nodes.remove(); // Make Binary Tree from the next lowest frequency character in the p.q.
            BinaryTree<CodeTreeElement> tree = new BinaryTree<CodeTreeElement>(new CodeTreeElement(t1.getData().getFrequency() + t2.getData().getFrequency(), '|'), t1, t2); // Parent of t1 and t2, stores the frequency of these characters added together.
            nodes.add(tree); // Add this new tree to back to the priority queue. This process resulted in size of priority queue being decreased by one.
       }
        return nodes.peek(); // At this point, nodes only has one element--the correct tree; return that tree.
    }

    /**
     * Computes the code for all characters in the tree and enters them
     * into a map where the key is a character and the value is the code of 1's and 0's representing
     * that character.
     *
     * @param codeTree the tree for encoding characters produced by makeCodeTree
     * @return the map from characters to codes
     */
    public Map<Character, String> computeCodes(BinaryTree<CodeTreeElement> codeTree){
        if(codeTree == null){ // If the code tree is null, return null for the map (Will be handled in the compressFile method).
            return null;
        }
        Map<Character, String> codeMap = new HashMap<Character, String>(); // New map to return
        String path = ""; // String to hold path
        computeCodesHelper(codeMap, codeTree, path); // Calls helper method to allow recursive traversal/
        return codeMap; // Return completed map, after computerCodesHelper traverses tree/
    }

    /**
     * Helper method to allow for recursive traversal of the codeTree. Traverses tree inOrder.
     *
     * @param codeMap Map that is going to be updated with the paths to the characters in codeTree. Will be returned in computeCodes
     * @param codeTree Tree we are traversing
     * @param path Path along tree up to the current node in traversal. Builds string of 0s and 1s as computeCodesHelper is recursively called.
     */
    public void computeCodesHelper(Map<Character, String> codeMap, BinaryTree<CodeTreeElement> codeTree, String path){
        if(codeTree.hasLeft()){
          computeCodesHelper(codeMap, codeTree.getLeft(), path + "0"); // Recurse on the left tree, adds 0 on path to account for going left.
        }
        if(codeTree.isLeaf()){ // Checks if nodes is a leaf (only leaves are added to the map)
           codeMap.put(codeTree.getData().getChar(), path); // Adds character to map and the path along tree to get to that node.
        }
        if(codeTree.hasRight()){
            computeCodesHelper(codeMap, codeTree.getRight(), path + "1"); // Recurse on the right tree, adds 1 on path to account for going right.
        }
    }
    /**
     * Compress the file pathName and store compressed representation in compressedPathName.
     * @param codeMap - Map of characters to codes produced by computeCodes
     * @param pathName - File to compress
     * @param compressedPathName - Store the compressed data in this file
     * @throws IOException
     */
    public void compressFile(Map<Character, String> codeMap, String pathName, String compressedPathName) throws IOException{
        if(codeMap == null){ // Handles case where the file is empty on input.
            throw new IOException("The inputted file is empty");
        }
        BufferedReader input;
        try { //Try to open file
            input = new BufferedReader(new FileReader(pathName));
        }
        catch(FileNotFoundException e) {
            throw new IOException("File not found\n" + e); // Can't open file.
        }
        BufferedBitWriter bitOutput;
        try {
            bitOutput = new BufferedBitWriter(compressedPathName);
        }
        catch(IOException e){
            throw new IOException("Cannot create bit reader " + e); // If can't create BufferedBitWriter
        }
        try{ // Try to read file
            char currentChar;
            int charInt;
            while((charInt = input.read()) >= 0){
                currentChar = (char) charInt; // Casts the int read off the document to a char
                String charCode = codeMap.get(currentChar); // Gets the charCode corresponding to the current character in input document from map.
                for(int i = 0; i < charCode.length(); i++){ // Iterates through each char in the charCode Sting.
                    try{ // Tries to write bit to output file
                        bitOutput.writeBit(charCode.charAt(i) == '1'); // Write true if char i of charCode is a 1, false if it is a 0
                    }
                    catch(IOException e){
                        throw new IOException("Cannot write bit to output file" + e); // If cannot write bit to output file.
                    }
                }
            }
        }
        catch(IOException e){ // Cannot read from file
            try{
                input.close(); // Tries to close input file
            }
            catch(IOException e2){
                throw new IOException("Cannot read from file and cannot close input file\n" + e + e2 ); //Executes if cannot close input file
            }
            try{
                bitOutput.close(); // Tries to close output file
            }
            catch(IOException e2){
                throw new IOException("Cannot read from file and cannot close output file\n" + e + e2 ); // Executes if cannot close output file
            }
            throw new IOException("Cannot read from the file\n" + e); // Executes if can close but cannot read file
        }
        try{ // Try to close file after compression
            input.close(); // Tries to close input file
        }
        catch(IOException e){
            throw new IOException("Cannot close file\n" + e); // Can't close input file
        }
        try{
            bitOutput.close(); // Tries to close output file
        }
        catch(IOException e){
            throw new IOException("Cannot close output file\n" + e); // If cannot close output file.
        }
    }
//
//    /**
//     * Decompress file compressedPathName and store plain text in decompressedPathName.
//     * @param compressedPathName - file created by compressFile
//     * @param decompressedPathName - store the decompressed text in this file, contents should match the original file before compressFile
//     * @param codeTree - Tree mapping compressed data to characters
//     * @throws IOException
//     */
//    public void decompressFile(String compressedPathName, String decompressedPathName, BinaryTree<CodeTreeElement> codeTree) throws IOException{
//        BufferedBitReader bitInput;
//        try{
//            bitInput = new BufferedBitReader(compressedPathName);
//        }
//        catch(FileNotFoundException e){
//            throw new FileNotFoundException("File not found\n" + e);
//        }
//        BufferedWriter output;
//        try{
//            output = new BufferedWriter(new FileWriter(decompressedPathName));
//        }
//        catch(IOException e){
//            throw new IOException("Cannot create new BufferedWriter\n" + e);
//        }
//        try{ // Try to readBit from compressed file
//            BinaryTree<CodeTreeElement> currentTree = codeTree;
//            while(bitInput.hasNext()) {
//                while (!currentTree.isLeaf()) {
//                    boolean bit = bitInput.readBit(); // Sets bit to the boolean value read from compressed file.
//                    if (!bit) { // If bit is false
//                        currentTree = currentTree.getLeft(); // Set current tree to the left child of current tree.
//                    } else { // If bit is true
//                        currentTree = currentTree.getRight(); // Set current tree to the right child of current tree.
//                    }
//                }
//                try{ // Try to write char to decompressed file
//                    output.write(currentTree.getData().getChar());
//                }
//                catch(IOException e){ //If cannot write char to decompressed file
//                    try{ //Try to close input file
//                        bitInput.close();
//                    }
//                    catch(IOException e2){
//                        throw new IOException("Cannot write bit to decompressed file and cannot close compressed file\n" + e + e2);
//                    }
//                    try{ //try to close output file
//                        output.close();
//                    }
//                    catch(IOException e2){
//                        throw new IOException("Cannot write bit to decompressed file and cannot close output file\n" + e + e2);
//                    }
//                    throw new IOException("Cannot write bit to decompressed file\n" + e);
//                }
//                currentTree = codeTree; // Resets currentTree to codeTree
//            }
//        }
//        catch(IOException e){ // If cannot read bit from compressed file
//            try{ //Try to close input
//                bitInput.close();
//            }
//            catch(IOException e2){
//                throw new IOException("Cannot read bit from file and cannot close compressed file\n" + e + e2);
//            }
//            try{ // Try to close output file
//                output.close();
//            }
//            catch(IOException e2){
//                throw new IOException("Cannot read bit from file and cannot close output file\n" + e + e2);
//            }
//            throw new IOException("Cannot read bit from compressed file\n" + e);
//        }
//        //Try to close file --- need these as well because the set of try/catch for closing file above are within the exception that you cannot read file
//        try{ // Try to close input file
//            bitInput.close();
//        }
//        catch(IOException e){
//            throw new IOException("Cannot read bit from file and cannot compressed file\n" + e);
//        }
//        try{ // Try to close output file
//            output.close();
//        }
//        catch(IOException e){
//            throw new IOException("Cannot read bit from file and cannot output file\n" + e);
//        }
//    }
//
//    public static void main(String[] args) throws IOException {
//        HuffmanEncoding tester = new HuffmanEncoding();
//        if(DEBUG){ // Debug flagger is static final.
//            System.out.println(tester.countFrequencies(oneCharacterRepeatedTest)); // Print out frequency map
//            System.out.println(tester.makeCodeTree(tester.countFrequencies(oneCharacterRepeatedTest))); // Print out the codeTree
//            System.out.println(tester.computeCodes(tester.makeCodeTree(tester.countFrequencies(oneCharacterRepeatedTest)))); // Print out the codeMap
//
//        }

        // NOTE, COMMENTED OUT THE CODE TO ACTUALLY COMPRESS/DECOMPRESS THE FILES:


//        tester.compressFile(tester.computeCodes(tester.makeCodeTree(tester.countFrequencies(warAndPeace))), warAndPeace, warAndPeace.substring(0, 18) + "_compressed.txt");
//        tester.decompressFile(warAndPeace.substring(0, 18) + "_compressed.txt", warAndPeace.substring(0, 18) + "_decompressed.txt", tester.makeCodeTree(tester.countFrequencies(warAndPeace)));
//
//        tester.compressFile(tester.computeCodes(tester.makeCodeTree(tester.countFrequencies(constitution))), constitution, constitution.substring(0, 21) + "_compressed.txt");
//        tester.decompressFile(constitution.substring(0, 21) + "_compressed.txt", constitution.substring(0, 21) + "_decompressed.txt", tester.makeCodeTree(tester.countFrequencies(constitution)));
//
//        tester.compressFile(tester.computeCodes(tester.makeCodeTree(tester.countFrequencies(emptyFileTest))), emptyFileTest, emptyFileTest.substring(0, 20) + "_compressed.txt");
//        tester.decompressFile(emptyFileTest.substring(0, 20) + "_compressed.txt", emptyFileTest.substring(0, 20) + "_decompressed.txt", tester.makeCodeTree(tester.countFrequencies(emptyFileTest)));

//        tester.compressFile(tester.computeCodes(tester.makeCodeTree(tester.countFrequencies(oneCharacterTest))), oneCharacterTest, oneCharacterTest.substring(0, 23) + "_compressed.txt");
//        tester.decompressFile(oneCharacterTest.substring(0, 23) + "_compressed.txt", oneCharacterTest.substring(0, 23) + "_decompressed.txt", tester.makeCodeTree(tester.countFrequencies(oneCharacterTest)));
//
//        tester.compressFile(tester.computeCodes(tester.makeCodeTree(tester.countFrequencies(oneCharacterRepeatedTest))), oneCharacterRepeatedTest, oneCharacterRepeatedTest.substring(0, 31) + "_compressed.txt");
//        tester.decompressFile(oneCharacterRepeatedTest.substring(0, 31) + "_compressed.txt", oneCharacterRepeatedTest.substring(0, 31) + "_decompressed.txt", tester.makeCodeTree(tester.countFrequencies(oneCharacterRepeatedTest)));
    }
}
