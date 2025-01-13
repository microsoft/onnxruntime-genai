/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * Use Config to set the ORT execution providers (EPs) and their options. The EPs are applied based
 * on insertion order.
 */
public final class Config implements AutoCloseable {
  private long nativeHandle;

  /**
   * Creates a Config from the given configuration directory.
   *
   * @param modelPath The path to the configuration directory.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Config(String modelPath) throws GenAIException {
    nativeHandle = createConfig(modelPath);
  }

  /** Clear the list of providers in the config */
  public void clearProviders() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    clearProviders(nativeHandle);
  }

  /**
   * Add the provider at the end of the list of providers in the given config if it doesn't already
   * exist. If it already exists, does nothing.
   *
   * @param providerName The provider name.
   */
  public void appendProvider(String providerName) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    appendProvider(nativeHandle, providerName);
  }

  /**
   * Set a provider option.
   *
   * @param providerName The provider name.
   * @param optionKey The key of the option to set.
   * @param optionValue The value of the option to set.
   */
  public void setProviderOption(String providerName, String optionKey, String optionValue) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    setProviderOption(nativeHandle, providerName, optionKey, optionValue);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyConfig(nativeHandle);
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

  private native long createConfig(String modelPath) throws GenAIException;

  private native void destroyConfig(long configHandle);

  private native void clearProviders(long configHandle);

  private native void appendProvider(long configHandle, String provider_name);

  private native void setProviderOption(
      long configHandle, String providerName, String optionKey, String optionValue);
}
