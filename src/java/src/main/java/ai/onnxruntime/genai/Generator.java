/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

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
public final class Generator implements AutoCloseable, Iterable<Integer> {
  private long nativeHandle = 0;

  /**
   * Constructs a Generator object with the given model and generator parameters.
   *
   * @param model The model.
   * @param generatorParams The generator parameters.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Generator(Model model, GeneratorParams generatorParams) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalArgumentException("model has been freed and is invalid");
    }

    if (generatorParams.nativeHandle() == 0) {
      throw new IllegalArgumentException("generatorParams has been freed and is invalid");
    }

    nativeHandle = createGenerator(model.nativeHandle(), generatorParams.nativeHandle());
  }

  /**
   * Returns an iterator over elements of type {@code Integer}. A new token is generated each time
   * next() is called, by calling computeLogits and generateNextToken.
   *
   * @return an Iterator.
   */
  @Override
  public java.util.Iterator<Integer> iterator() {
    return new Iterator();
  }

  /**
   * Checks if the generation process is done.
   *
   * @return true if the generation process is done, false otherwise.
   */
  public boolean isDone() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return isDone(nativeHandle);
  }

  /**
   * Appends tokens to the generator.
   *
   * @param inputIDs The tokens to append.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void appendTokens(int[] inputIDs) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    appendTokens(nativeHandle, inputIDs);
  }

  /**
   * Appends token sequences to the generator.
   *
   * @param sequences The sequences to append.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void appendTokenSequences(Sequences sequences) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (sequences.nativeHandle() == 0) {
      throw new IllegalArgumentException("sequences has been freed and is invalid");
    }

    appendTokenSequences(nativeHandle, sequences.nativeHandle());
  }

  /**
   * Rewinds the generator to the given length. This is useful when the user wants to rewind the
   * generator to a specific length and continue generating from that point.
   *
   * @param newLength The desired length in tokens after rewinding.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void rewindTo(long newLength) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    rewindTo(nativeHandle, newLength);
  }

  /**
   * Computes the logits from the model based on the input ids and the past state. The computed
   * logits are stored in the generator.
   *
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void generateNextToken() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    generateNextTokenNative(nativeHandle);
  }

  /**
   * Retrieves a sequence of token ids for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return An array of integers with the sequence token ids.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int[] getSequence(long sequenceIndex) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return getSequenceNative(nativeHandle, sequenceIndex);
  }

  /**
   * Retrieves the last token in the sequence for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return The last token in the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int getLastTokenInSequence(long sequenceIndex) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return getSequenceLastToken(nativeHandle, sequenceIndex);
  }

  /**
   * Returns a copy of the model output identified by the given name as a Tensor.
   *
   * @param name The name of the output needed.
   * @return The tensor.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tensor getOutput(String name) throws GenAIException {
    long tensorHandle = getOutputNative(nativeHandle, name);
    return new Tensor(tensorHandle);
  }

  /**
   * Sets the adapter with the given adapter name as active.
   *
   * @param adapters The Adapters container.
   * @param adapterName The adapter name that was previously loaded.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setActiveAdapter(Adapters adapters, String adapterName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setActiveAdapter(nativeHandle, adapters.nativeHandle(), adapterName);
  }

  /** Closes the Generator and releases any associated resources. */
  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyGenerator(nativeHandle);
      nativeHandle = 0;
    }
  }

  /** The Iterator class for the Generator to simplify usage when streaming tokens. */
  private class Iterator implements java.util.Iterator<Integer> {
    @Override
    public boolean hasNext() {
      return !isDone();
    }

    @Override
    public Integer next() {
      try {
        generateNextToken();
        return getLastTokenInSequence(0);
      } catch (GenAIException e) {
        throw new RuntimeException(e);
      }
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createGenerator(long modelHandle, long generatorParamsHandle)
      throws GenAIException;

  private native void destroyGenerator(long nativeHandle);

  private native boolean isDone(long nativeHandle);

  private native void appendTokens(long nativeHandle, int[] tokens) throws GenAIException;

  private native void appendTokenSequences(long nativeHandle, long sequencesHandle)
      throws GenAIException;

  private native void rewindTo(long nativeHandle, long newLength) throws GenAIException;

  private native void generateNextTokenNative(long nativeHandle) throws GenAIException;

  private native int[] getSequenceNative(long nativeHandle, long sequenceIndex)
      throws GenAIException;

  private native int getSequenceLastToken(long nativeHandle, long sequenceIndex)
      throws GenAIException;

  private native void setActiveAdapter(
      long nativeHandle, long adaptersNativeHandle, String adapterName) throws GenAIException;

  private native long getOutputNative(long nativeHandle, String outputName) throws GenAIException;
}
