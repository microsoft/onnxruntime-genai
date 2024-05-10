/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

import java.nio.IntBuffer;
import java.io.IOException;

public class Generator implements AutoCloseable {
    private long nativeHandle;

    public Generator(Model model, GeneratorParams generatorParams)
    {
        // Result.VerifySuccess(NativeMethods.OgaCreateGenerator(model.Handle, generatorParams.Handle, out _generatorHandle));
    }

    public boolean isDone()
    {
        // return NativeMethods.OgaGenerator_IsDone(_generatorHandle);
        return false;
    }

    public void computeLogits()
    {
        // Result.VerifySuccess(NativeMethods.OgaGenerator_ComputeLogits(_generatorHandle));
    }

    public void GenerateNextToken()
    {
        // Result.VerifySuccess(NativeMethods.OgaGenerator_GenerateNextToken(_generatorHandle));
    }

    public IntBuffer GetSequence(long index)
    {
        return getSequence(nativeHandle);
    }

    @Override
    public void close() throws Exception {

    }

    private native IntBuffer getSequence(long nativeHandle);
}