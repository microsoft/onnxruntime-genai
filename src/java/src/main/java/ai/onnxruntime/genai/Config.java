/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * Use Config to set multiple ORT execution providers. The EP used will be chosen based on the
 * insertion order.
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
   * Add the provider at the end of the list of providers in the given config if it doesn't already exist.
   * If it already exists, does nothing.
   *
   * @param provider_name The provider name.
   */
  public void appendProvider(String provider_name) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    appendProvider(nativeHandle, provider_name);
  }

  /**
   * Set a provider option.
   *
   * @param provider_name The provider name.
   * @param option_key The key of the option to set.
   * @param option_value The value of the option to set.
   */
  public void setProviderOption(String provider_name, String option_key, String option_value) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    setProviderOption(nativeHandle, provider_name, option_key, option_value);
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
      long configHandle, String provider_name, String option_key, String option_value);
}
