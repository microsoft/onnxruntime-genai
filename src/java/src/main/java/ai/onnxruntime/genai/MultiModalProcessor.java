/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * The MultiModalProcessor class is responsible for converting text/images into a NamedTensors list
 * that can be fed into a Generator class instance.
 */
public class MultiModalProcessor implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct a MultiModalProcessor for a given model.
   *
   * @param model The model to be used.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public MultiModalProcessor(Model model) throws GenAIException {
    assert (model.nativeHandle() != 0); // internal code should never pass an invalid model

    nativeHandle = createMultiModalProcessor(model.nativeHandle());
  }

  /**
   * Processes a string and image into a NamedTensor.
   *
   * @param prompt Text to encode as token ids.
   * @param images image input.
   * @return NamedTensors object.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public NamedTensors processImages(String prompt, Images images) throws GenAIException {
    long imagesHandle = (images == null) ? 0 : images.nativeHandle();
    long namedTensorsHandle = processorProcessImages(nativeHandle, prompt, imagesHandle);

    return new NamedTensors(namedTensorsHandle);
  }

  /**
   * Decodes a sequence of token ids into text.
   *
   * @param sequence Collection of token ids to decode to text.
   * @return The text representation of the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String decode(int[] sequence) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return processorDecode(nativeHandle, sequence);
  }

  /**
   * Creates a TokenizerStream object for streaming tokenization. This is used with Generator class
   * to provide each token as it is generated.
   *
   * @return The new TokenizerStream instance.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public TokenizerStream createStream() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    return new TokenizerStream(createTokenizerStreamFromProcessor(nativeHandle));
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyMultiModalProcessor(nativeHandle);
      nativeHandle = 0;
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createMultiModalProcessor(long modelHandle) throws GenAIException;

  private native void destroyMultiModalProcessor(long tokenizerHandle);

  private native long processorProcessImages(long processorHandle, String prompt, long imagesHandle)
      throws GenAIException;

  private native String processorDecode(long processorHandle, int[] sequence) throws GenAIException;

  private native long createTokenizerStreamFromProcessor(long processorHandle)
      throws GenAIException;
}
