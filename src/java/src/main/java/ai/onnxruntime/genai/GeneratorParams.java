/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The `GeneratorParams` class represents the parameters used for generating sequences with a model.
 * Set the prompt using setInput, and any other search options using setSearchOption.
 */
public final class GeneratorParams implements AutoCloseable {
  private long nativeHandle = 0;
  private ByteBuffer tokenIdsBuffer;

  GeneratorParams(Model model) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalStateException("model has been freed and is invalid");
    }

    nativeHandle = createGeneratorParams(model.nativeHandle());
  }

  public void setSearchOption(String optionName, double value) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setSearchOptionNumber(nativeHandle, optionName, value);
  }

  public void setSearchOption(String optionName, boolean value) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setSearchOptionBool(nativeHandle, optionName, value);
  }

  /**
   * Sets the prompt/s for model execution. The `sequences` are created by using Tokenizer.Encode or
   * EncodeBatch.
   *
   * @param sequences Sequences containing the encoded prompt.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setInput(Sequences sequences) throws GenAIException {
    if (sequences.nativeHandle() == 0) {
      throw new IllegalArgumentException("sequences has been freed and is invalid");
    }

    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    tokenIdsBuffer = null; // free the token ids buffer if previously used.
    setInputSequences(nativeHandle, sequences.nativeHandle());
  }

  /**
   * Sets the prompt/s token ids for model execution. The `tokenIds` are the encoded
   *
   * @param tokenIds The token ids of the encoded prompt/s.
   * @param sequenceLength The length of each sequence.
   * @param batchSize The batch size
   * @throws GenAIException If the call to the GenAI native API fails.
   *     <p>NOTE: All sequences in the batch must be the same length.
   */
  public void setInput(int[] tokenIds, int sequenceLength, int batchSize) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (sequenceLength * batchSize != tokenIds.length) {
      throw new IllegalArgumentException(
          "tokenIds length must be equal to sequenceLength * batchSize");
    }

    // allocate a direct buffer to store the token ids so that they remain valid throughout the
    // generation process as the GenAI layer does not copy the token ids.
    tokenIdsBuffer = ByteBuffer.allocateDirect(tokenIds.length * Integer.BYTES);
    tokenIdsBuffer.order(ByteOrder.nativeOrder());
    tokenIdsBuffer.asIntBuffer().put(tokenIds);

    setInputIDs(nativeHandle, tokenIdsBuffer, sequenceLength, batchSize);
  }

  /**
   * Add a Tensor as a model input.
   *
   * @param name Name of the model input the tensor will provide.
   * @param tensor Tensor to add.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setInput(String name, Tensor tensor) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (tensor.nativeHandle() == 0) {
      throw new IllegalArgumentException("tensor has been freed and is invalid");
    }

    setModelInput(nativeHandle, name, tensor.nativeHandle());
  }

  /**
   * Add a NamedTensors as a model input.
   *
   * @param namedTensors NamedTensors to add.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setInputs(NamedTensors namedTensors) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (namedTensors.nativeHandle() == 0) {
      throw new IllegalArgumentException("tensor has been freed and is invalid");
    }

    setInputs(nativeHandle, namedTensors.nativeHandle());
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyGeneratorParams(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createGeneratorParams(long modelHandle) throws GenAIException;

  private native void destroyGeneratorParams(long nativeHandle);

  private native void setSearchOptionNumber(long nativeHandle, String optionName, double value)
      throws GenAIException;

  private native void setSearchOptionBool(long nativeHandle, String optionName, boolean value)
      throws GenAIException;

  private native void setInputSequences(long nativeHandle, long sequencesHandle)
      throws GenAIException;

  private native void setModelInput(long nativeHandle, String inputName, long tensorHandle)
      throws GenAIException;

  private native void setInputs(long nativeHandle, long namedTensorsHandle) throws GenAIException;

  private native void setInputIDs(
      long nativeHandle, ByteBuffer tokenIds, int sequenceLength, int batchSize)
      throws GenAIException;
}
