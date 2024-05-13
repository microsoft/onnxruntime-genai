/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

/**
 * The Generator class generates output using a model and generator parameters.
 *
 * <p>The expected usage is to loop until isDone returns false. Within the loop, call computeLogits
 * followed by generateNextToken.
 *
 * <p>The newly generated token can be retrieved with getLastTokenInSequence and decoded with
 * TokenizerStream.Decode.
 *
 * <p>After the generation process is done, GetSequence can be used to retrieve the complete
 * generated sequence if needed.
 */
public class Generator implements AutoCloseable {
  private long nativeHandle = 0;

  /**
   * Constructs a Generator object with the given model and generator parameters.
   *
   * @param model The model.
   * @param generatorParams The generator parameters.
   */
  public Generator(Model model, GeneratorParams generatorParams) throws GenAIException {
    nativeHandle = createGenerator(model.nativeHandle(), generatorParams.nativeHandle());
  }

  /**
   * Checks if the generation process is done.
   *
   * @return true if the generation process is done, false otherwise.
   */
  public boolean isDone() {
    return isDone(nativeHandle);
  }

  /** Computes the logits for the next token in the sequence. */
  public void computeLogits() throws GenAIException {
    computeLogits(nativeHandle);
  }

  /** Generates the next token in the sequence. */
  public void generateNextToken() throws GenAIException {
    generateNextTokenNative(nativeHandle);
  }

  /**
   * Retrieves a sequence of token ids for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return An array of integers with the sequence token ids.
   */
  public int[] getSequence(long sequenceIndex) throws GenAIException {
    return getSequenceImpl(sequenceIndex, false);
  }

  /**
   * Retrieves the last token in the sequence for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return The last token in the sequence.
   */
  public int getLastTokenInSequence(long sequenceIndex) throws GenAIException {
    return getSequenceImpl(sequenceIndex, true)[0];
  }

  /**
   * Closes the Generator and releases any associated resources.
   *
   * @throws Exception if an error occurs while closing the Generator.
   */
  @Override
  public void close() throws Exception {
    if (nativeHandle != 0) {
      destroyGenerator(nativeHandle);
      nativeHandle = 0;
    }
  }

  private int[] getSequenceImpl(long sequenceIndex, boolean lastTokenOnly) throws GenAIException {
    return getSequenceNative(nativeHandle, sequenceIndex, lastTokenOnly);
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createGenerator(long modelHandle, long generatorParamsHandle);

  private native void destroyGenerator(long nativeHandle);

  private native boolean isDone(long nativeHandle);

  private native void computeLogits(long nativeHandle);

  private native void generateNextTokenNative(long nativeHandle);

  private native int[] getSequenceNative(
      long nativeHandle, long sequenceIndex, boolean lastTokenOnly);
}
