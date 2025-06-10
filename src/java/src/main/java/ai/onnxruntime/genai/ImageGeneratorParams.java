/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** Represents the parameters used for generating images with a diffusion model. */
public final class ImageGeneratorParams implements AutoCloseable {
  private long nativeHandle = 0;

  /**
   * Creates ImageGeneratorParams from the given model.
   *
   * @param model The model to use for image generation.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public ImageGeneratorParams(Model model) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalStateException("model has been freed and is invalid");
    }

    nativeHandle = createImageGeneratorParams(model.nativeHandle());
  }

  /**
   * Sets the prompt and optional negative prompt for image generation.
   *
   * @param prompt The text prompt describing the desired image.
   * @param negativePrompt Optional text describing what to avoid in the image. Can be null.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setPrompts(String prompt, String negativePrompt) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    String[] prompts = new String[] {prompt};
    String[] negativePrompts = negativePrompt != null ? new String[] {negativePrompt} : null;
    setPrompts(nativeHandle, prompts, negativePrompts, 1);
  }

  /**
   * Sets multiple prompts and optional negative prompts for batch image generation.
   *
   * @param prompts Array of text prompts describing the desired images.
   * @param negativePrompts Optional array of texts describing what to avoid in each image. Can be
   *     null.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setPrompts(String[] prompts, String[] negativePrompts) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (negativePrompts != null && prompts.length != negativePrompts.length) {
      throw new IllegalArgumentException(
          "prompts and negativePrompts arrays must be the same length");
    }

    setPrompts(nativeHandle, prompts, negativePrompts, prompts.length);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyImageGeneratorParams(nativeHandle);
      nativeHandle = 0;
    }
  }

  /**
   * Returns the native handle for this object.
   *
   * @return The native handle.
   */
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

  private native long createImageGeneratorParams(long modelHandle) throws GenAIException;

  private native void destroyImageGeneratorParams(long nativeHandle);

  private native void setPrompts(
      long nativeHandle, String[] prompts, String[] negativePrompts, long promptCount)
      throws GenAIException;
}
