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
   * Add a Tensor as a model input.
   *
   * @param name Name of the model input the tensor will provide.
   * @param tensor Tensor to add.
   * @throws GenAIException
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
   * @throws GenAIException
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

  private native void setModelInput(long nativeHandle, String inputName, long tensorHandle)
      throws GenAIException;
  
  private native void setInputs(long nativeHandle, long namedTensorsHandle)
      throws GenAIException;
}
