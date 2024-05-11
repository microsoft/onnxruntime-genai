/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

public class Model implements AutoCloseable {
    private long modelHandle;
    private GeneratorParams generatorParams;

    public Model(String modelPath) throws GenAIException
    {
        modelHandle = createModel(modelPath);
        generatorParams = new GeneratorParams(modelHandle);
    }

    public GeneratorParams GetParams() {
        return generatorParams;
    }

    public Sequences Generate(GeneratorParams generatorParams) throws GenAIException {
        long sequencesHandle = generate(modelHandle, generatorParams.nativeHandle());
        return new Sequences(sequencesHandle);
    }

    @Override
    public void close() throws GenAIException {
        if (modelHandle != 0) {
            destroyModel(modelHandle);
            modelHandle = 0;
        }
    }

    protected long nativeHandle() {
        return modelHandle;
    }

    static {
        try {
            GenAI.init();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
        }
    }

    private native long createModel(String modelPath);
    private native void destroyModel(long modelHandle);

    private native long generate(long modelHandle, long generatorParamsHandle);
}