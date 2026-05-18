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
   * Construct a Model from folder path with an explicit execution provider.
   *
   * <p>The {@code ep} argument selects the execution provider for v4 model packages, bypassing
   * GenAI's compatibility-intersection defaulting. Pass {@code null} or an empty string to fall
   * back to defaulting. In flat-directory (legacy) mode a non-empty {@code ep} raises an error.
   *
   * @param modelPath The path of the GenAI model.
   * @param ep The execution provider name (e.g. {@code "CUDAExecutionProvider"}), or {@code null}
   *     for defaulting.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Model(String modelPath, String ep) throws GenAIException {
    nativeHandle = createModelWithEp(modelPath, ep);
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

  private native long createModelWithEp(String modelPath, String ep) throws GenAIException;

  private native long createModelFromConfig(long configHandle) throws GenAIException;

  private native void destroyModel(long modelHandle);
}
