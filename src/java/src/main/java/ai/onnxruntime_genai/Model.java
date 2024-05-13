/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

public class Model implements AutoCloseable {
    private long modelHandle;

    public Model(String modelPath) throws GenAIException {
        modelHandle = createModel(modelPath);
    }

    /**
     * Run the model to generate output sequences.
     * Generation is limited to the "max_length" value (default:300) in the generator parameters.
     * Use a Tokenizer to decode the generated sequences.
     * 
     * @param generatorParams The generator parameters.
     * @return The generated sequences.
     */
    public Sequences generate(GeneratorParams generatorParams) throws GenAIException {
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
