/**
 * A class to hold a (frequency, character) pair for Huffman Code Tree
 * @author Scot Drysdale
 */
public class CodeTreeElement {
	Long myFrequency;       // The frequency with which the character occurred
	Character myChar;       // The character corresponding to the count
	
	/**
	 * Constructor
	 * @param frequency the frequency count for c
	 * @param c the character counted
	 */
	public CodeTreeElement(Long frequency, Character c) {
		myFrequency = frequency;
		myChar = c;
	}
	
	/**
	 * Get the frequency
	 * @return  the frequency
	 */
	public Long getFrequency() {
		return myFrequency;
	}

	/** 
	 * Get the character
	 * @return the character
	 */
	public Character getChar() {
		return myChar;
	}
	
	public String toString() {
		return "character: " + myChar + "  frequency: " + myFrequency;
	}
}
