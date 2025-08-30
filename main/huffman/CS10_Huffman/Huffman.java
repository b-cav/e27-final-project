import java.io.IOException;
import java.util.Map;

/**
 * Interface for PS-3
 * @author Tim Pierson and Henry Kim, Dartmouth CS10 Winter 2023
 */
public interface Huffman {
    /**
     * Read file provided in pathName and count how many times each character appears
     * @param pathName - path to a file to read
     * @return - Map with a character as a key and the number of times the character appears in the file as value
     * @throws IOException
     */
    Map<Character, Long> countFrequencies(String pathName) throws IOException;

    /**
     * Construct a code tree from a map of frequency counts. Note: this code should handle the special
     * cases of empty files or files with a single character.
     *
     * @param frequencies a map of Characters with their frequency counts from countFrequencies
     * @return the code tree.
     */
    BinaryTree<CodeTreeElement> makeCodeTree(Map<Character, Long> frequencies);

    /**
     * Computes the code for all characters in the tree and enters them
     * into a map where the key is a character and the value is the code of 1's and 0's representing
     * that character.
     *
     * @param codeTree the tree for encoding characters produced by makeCodeTree
     * @return the map from characters to codes
     */
    Map<Character, String> computeCodes(BinaryTree<CodeTreeElement> codeTree);

    /**
     * Compress the file pathName and store compressed representation in compressedPathName.
     * @param codeMap - Map of characters to codes produced by computeCodes
     * @param pathName - File to compress
     * @param compressedPathName - Store the compressed data in this file
     * @throws IOException
     */
    void compressFile(Map<Character, String> codeMap, String pathName, String compressedPathName) throws IOException;

    /**
     * Decompress file compressedPathName and store plain text in decompressedPathName.
     * @param compressedPathName - file created by compressFile
     * @param decompressedPathName - store the decompressed text in this file, contents should match the original file before compressFile
     * @param codeTree - Tree mapping compressed data to characters
     * @throws IOException
     */
    void decompressFile(String compressedPathName, String decompressedPathName, BinaryTree<CodeTreeElement> codeTree) throws IOException;
}
