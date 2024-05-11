/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

public class Generator implements AutoCloseable {
    private long nativeHandle = 0;

    public Generator(Model model, GeneratorParams generatorParams) {
        nativeHandle = createGenerator(model.nativeHandle(), generatorParams.nativeHandle());
    }

    public boolean isDone() {
        return isDone(nativeHandle);
    }

    public void computeLogits() {
        computeLogits(nativeHandle);
    }

    public void GenerateNextToken() {
        generateNextToken(nativeHandle);
    }

    public int[] GetSequence(long index) {
        return getSequence(nativeHandle);
    }

    @Override
    public void close() throws Exception {
        if (nativeHandle != 0) {
            releaseGenerator(nativeHandle);
            nativeHandle = 0;
        }
    }

    private native long createGenerator(long modelHandle, long generatorParamsHandle);

    private native void releaseGenerator(long nativeHandle);

    private native boolean isDone(long nativeHandle);

    private native void computeLogits(long nativeHandle);

    private native void generateNextToken(long nativeHandle);

    private native int[] getSequence(long nativeHandle);
}
