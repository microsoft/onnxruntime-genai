package ai.onnxruntime_genai;

public class GeneratorParams implements AutoCloseable {
    private long nativeHandle = 0;

    public GeneratorParams(Model model) {
        nativeHandle = createGeneratorParams(model.nativeHandle());
    }

    public void SetSearchOption(String searchOption, double value) {
        setSearchOptionNumber(nativeHandle, value);
    }

    public void SetSearchOption(String searchOption, boolean value) {
        setSearchOptionBool(nativeHandle, value);
    }

    @Override
    public void close() throws Exception {
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
    private native void setSearchOptionNumber(long nativeHandle, double value);
    private native void setSearchOptionBool(long nativeHandle, boolean value);

}
