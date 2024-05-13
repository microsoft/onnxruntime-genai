/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

/** Represents a collection of encoded prompts/responses. */
public class Sequences implements AutoCloseable {
  private long sequencesHandle;
  private long numSequences;

  protected Sequences(long sequencesHandle) {
    assert (sequencesHandle != 0); // internal usage should never pass an invalid handle

    this.sequencesHandle = sequencesHandle;
    numSequences = getSequencesCount(sequencesHandle);
  }

  /**
   * Gets the number of sequences in the collection. This is equivalent to the batch size.
   *
   * @return The number of sequences.
   */
  public long numSequences() {
    return numSequences;
  }

  /**
   * Gets the sequence at the specified index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return The sequence as an array of integers.
   */
  int[] getSequence(long sequenceIndex) {
    return getSequenceNative(sequencesHandle, sequenceIndex);
  }

  @Override
  public void close() throws Exception {
    if (sequencesHandle != 0) {
      destroySequences(sequencesHandle);
      sequencesHandle = 0;
    }
  }

  protected long nativeHandle() {
    return sequencesHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long getSequencesCount(long sequencesHandle);

  private native int[] getSequenceNative(long sequencesHandle, long sequenceIndex);

  private native void destroySequences(long sequencesHandle);
}
