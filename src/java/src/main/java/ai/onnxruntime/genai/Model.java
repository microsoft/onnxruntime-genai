/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** An ORT GenAI model. */
public final class Model implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct a Model from folder path.
   *
   * @param modelPath The path of the GenAI model.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Model(String modelPath) throws GenAIException {
    nativeHandle = createModel(modelPath);
  }

  /**
   * Construct a Model from the given Config.
   *
   * @param config The config to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Model(Config config) throws GenAIException {
    nativeHandle = createModelFromConfig(config.nativeHandle());
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyModel(nativeHandle);
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

  private native long createModel(String modelPath) throws GenAIException;

  private native long createModelFromConfig(long configHandle) throws GenAIException;

  private native void destroyModel(long modelHandle);
}
