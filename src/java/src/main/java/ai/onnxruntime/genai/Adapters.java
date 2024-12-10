/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** A container of adapters. */
public final class Adapters implements AutoCloseable {
  private long nativeHandle = 0;

  /**
   * Constructs an Adapters object with the given model.
   *
   * @param model The model.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Adapters(Model model) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalArgumentException("model has been freed and is invalid");
    }

    nativeHandle = createAdapters(model.nativeHandle());
  }

  /**
   * Loads the model adapter from the given adapter file path and adapter name.
   *
   * @param adapterFilePath The path of the adapter.
   * @param adapterName A unique user supplied adapter identifier.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void loadAdapter(String adapterFilePath, String adapterName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    loadAdapter(nativeHandle, adapterFilePath, adapterName);
  }

  /**
   * Unloads the adapter with the given identifier from the previosly loaded adapters. If the
   * adapter is not found, or if it cannot be unloaded (when it is in use), an error is returned.
   *
   * @param adapterName A unique user supplied adapter identifier.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void unloadAdapter(String adapterName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    unloadAdapter(nativeHandle, adapterName);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyAdapters(nativeHandle);
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

  private native long createAdapters(long modelHandle) throws GenAIException;

  private native void destroyAdapters(long nativeHandle);

  private native void loadAdapter(long nativeHandle, String adapterFilePath, String adapterName)
      throws GenAIException;

  private native void unloadAdapter(long nativeHandle, String adapterName) throws GenAIException;
}
