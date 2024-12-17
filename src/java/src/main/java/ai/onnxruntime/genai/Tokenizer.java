/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** The Tokenizer class is responsible for converting between text and token ids. */
public class Tokenizer implements AutoCloseable {
  private long nativeHandle;

  /**
   * Creates a Tokenizer from the given model.
   *
   * @param model The model to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tokenizer(Model model) throws GenAIException {
    assert (model.nativeHandle() != 0); // internal code should never pass an invalid model

    nativeHandle = createTokenizer(model.nativeHandle());
  }

  /**
   * Encodes a string into a sequence of token ids.
   *
   * @param string Text to encode as token ids.
   * @return a Sequences object with a single sequence in it.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Sequences encode(String string) throws GenAIException {
    return encodeBatch(new String[] {string});
  }

  /**
   * Encodes an array of strings into a sequence of token ids for each input.
   *
   * @param strings Collection of strings to encode as token ids.
   * @return a Sequences object with one sequence per input string.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Sequences encodeBatch(String[] strings) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    long sequencesHandle = tokenizerEncode(nativeHandle, strings);
    return new Sequences(sequencesHandle);
  }

  /**
   * Decodes a sequence of token ids into text.
   *
   * @param sequence Collection of token ids to decode to text.
   * @return The text representation of the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String decode(int[] sequence) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerDecode(nativeHandle, sequence);
  }

  /**
   * Decodes a batch of sequences of token ids into text.
   *
   * @param sequences A Sequences object with one or more sequences of token ids.
   * @return An array of strings with the text representation of each sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String[] decodeBatch(Sequences sequences) throws GenAIException {
    int numSequences = (int) sequences.numSequences();

    String[] result = new String[numSequences];
    for (int i = 0; i < numSequences; i++) {
      result[i] = decode(sequences.getSequence(i));
    }

    return result;
  }

  /**
   * Creates a TokenizerStream object for streaming tokenization. This is used with Generator class
   * to provide each token as it is generated.
   *
   * @return The new TokenizerStream instance.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public TokenizerStream createStream() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return new TokenizerStream(createTokenizerStream(nativeHandle));
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyTokenizer(nativeHandle);
      nativeHandle = 0;
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createTokenizer(long modelHandle) throws GenAIException;

  private native void destroyTokenizer(long tokenizerHandle);

  private native long tokenizerEncode(long tokenizerHandle, String[] strings) throws GenAIException;

  private native String tokenizerDecode(long tokenizerHandle, int[] sequence) throws GenAIException;

  private native long createTokenizerStream(long tokenizerHandle) throws GenAIException;
}
