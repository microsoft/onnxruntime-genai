/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

/**
 * The Tokenizer class is responsible for converting between text and token ids.
 */
public class Tokenizer implements AutoCloseable {
    private long tokenizerHandle;

    public Tokenizer(Model model) {
        tokenizerHandle = createTokenizer(model.nativeHandle());
    }

    /**
     * Encodes a string into a sequence of token ids.
     * 
     * @param string
     * @return a Sequences object with a single sequence in it.
     */
    public Sequences Encode(String string) {
        return EncodeBatch(new String[] {string});
    }

    /**
     * Encodes an array of strings into a sequence of token ids for each input.
     * 
     * @param strings
     * @return a Sequences object with one sequence per input string.
     */
    public Sequences EncodeBatch(String[] strings) {
        long sequencesHandle = tokenizerEncode(tokenizerHandle, strings);
        return new Sequences(sequencesHandle);
    }

    /**
     * Decodes a sequence of token ids into text.
     * 
     * @param sequence
     * @return The text representation of the sequence.
     */
    public String Decode(int[] sequence) {
        return tokenizerDecode(tokenizerHandle, sequence);
    }

    /** 
     * Decodes a batch of sequences of token ids into text.
     * 
     * @param sequences A Sequences object with one or more sequences of token ids.
     * @return An array of strings with the text representation of each sequence.
     */
    public String[] DecodeBatch(Sequences sequences) {
        int numSequences = (int) sequences.numSequences();

        String[] result = new String[numSequences];
        for (int i = 0; i < numSequences; i++) {
            result[i] = Decode(sequences.getSequence(i));
        }

        return result;
    }

    /**
     * Creates a TokenizerStream object for streaming tokenization.
     * This is used with Generator class to provide each token as it is generated.
     * @return The new TokenizerStream instance.
     */
    public TokenizerStream CreateStream() {
        return new TokenizerStream(createTokenizerStream(tokenizerHandle));
    }

    @Override
    public void close() throws Exception {
        if (tokenizerHandle != 0) {
            destroyTokenizer(tokenizerHandle);
            tokenizerHandle = 0;
        }
    }

    static {
        try {
            GenAI.init();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
        }
    }

    private native long createTokenizer(long modelHandle);

    private native void destroyTokenizer(long tokenizerHandle);

    private native long tokenizerEncode(long tokenizerHandle, String[] strings);

    private native String tokenizerDecode(long tokenizerHandle, int[] sequence);

    private native long createTokenizerStream(long tokenizerHandle);
}
