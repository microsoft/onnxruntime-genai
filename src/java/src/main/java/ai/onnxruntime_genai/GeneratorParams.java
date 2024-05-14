package ai.onnxruntime_genai;

/**
 * The `GeneratorParams` class represents the parameters used for generating sequences with a model.
 * Set the prompt using setInput, and any other search options using setSearchOption.
 */
public class GeneratorParams implements AutoCloseable {
  private long nativeHandle = 0;

  public GeneratorParams(Model model) throws GenAIException {
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
   * @param sequences The encoded input prompt/s.
   */
  public void setInput(Sequences sequences) throws GenAIException {
    if (sequences.nativeHandle() == 0) {
      throw new IllegalStateException("sequences has been freed and is invalid");
    }

    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setInputSequences(nativeHandle, sequences.nativeHandle());
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyGeneratorParams(nativeHandle);
      nativeHandle = 0;
    }
  }

  protected long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createGeneratorParams(long modelHandle);

  private native void destroyGeneratorParams(long nativeHandle);

  private native void setSearchOptionNumber(long nativeHandle, String optionName, double value);

  private native void setSearchOptionBool(long nativeHandle, String optionName, boolean value);

  private native void setInputSequences(long nativeHandle, long sequencesHandle);
}
