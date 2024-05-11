package ai.onnxruntime_genai;

public class GeneratorParams implements AutoCloseable {
    private long nativeHandle = 0;

    protected GeneratorParams(long modelHandle) {
        nativeHandle = createGeneratorParams(modelHandle);
    }

    public void SetSearchOption(String searchOption, double value) {
        setSearchOptionNumber(nativeHandle, searchOption, value);
    }

    public void SetSearchOption(String searchOption, boolean value) {
        setSearchOptionBool(nativeHandle, searchOption, value);
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
    private native void setSearchOptionNumber(long nativeHandle, String optionName, double value);
    private native void setSearchOptionBool(long nativeHandle, String optionName, boolean value);

}
