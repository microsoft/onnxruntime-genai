/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

public class Model implements AutoCloseable {
    private long modelHandle;
    private long generatorParamsHandle;
    private GeneratorParams generatorParams;

    public Model(String modelPath) throws GenAIException
    {
        modelHandle = createModel(modelPath);

    }

    public GeneratorParams GetParams() {
        return generatorParams;
    }

    public Sequences Generate(GeneratorParams generatorParams) throws GenAIException {
        return new Sequences(generate(modelHandle, generatorParams.handle()));
    }

    @Override
    public void close() throws GenAIException {
        if (modelHandle != 0) {
            releaseModel(modelHandle);
            modelHandle = 0;
        }
    }

    protected long nativeHandle() {
        return modelHandle;
    }

    /*
    private native long loadModel(String modelPath);
    private native long createTokenizer(long nativeModel);
    private native void releaseTokenizer(long nativeTokenizer);
    private native String run(long nativeModel, long nativeTokenizer, String prompt, boolean useCallback);
    */

    static {
        try {
            GenAI.init();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
        }
    }

    private native long createModel(String modelPath);
    private native void releaseModel(long modelHandle);

    private native long generate(long modelHandle, long generatorParamsHandle);
}