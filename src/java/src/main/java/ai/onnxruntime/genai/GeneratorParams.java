/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.nio.ByteBuffer;

/**
 * Represents the parameters used for generating sequences with a model. Set the prompt using
 * setInputs, and any other search options using setSearchOption.
 */
public final class GeneratorParams implements AutoCloseable {
  private long nativeHandle = 0;
  private ByteBuffer tokenIdsBuffer;

  /**
   * Creates a GeneratorParams from the given model.
   *
   * @param model The model to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public GeneratorParams(Model model) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalStateException("model has been freed and is invalid");
    }

    nativeHandle = createGeneratorParams(model.nativeHandle());
  }

  /**
   * Set seach option with double value.
   *
   * @param optionName The option name.
   * @param value The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setSearchOption(String optionName, double value) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setSearchOptionNumber(nativeHandle, optionName, value);
  }

  /**
   * Set search option with boolean value.
   *
   * @param optionName The option name.
   * @param value The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setSearchOption(String optionName, boolean value) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setSearchOptionBool(nativeHandle, optionName, value);
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
}
