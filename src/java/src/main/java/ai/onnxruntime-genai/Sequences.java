/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime-genai;

import java.nio.Buffer;

public class GeneratorParams {

}

public class Generator {
    private long _generatorHandle;

    public Generator(Model model, GeneratorParams generatorParams)
    {
        // Result.VerifySuccess(NativeMethods.OgaCreateGenerator(model.Handle, generatorParams.Handle, out _generatorHandle));
    }

    public boolean IsDone()
    {
        // return NativeMethods.OgaGenerator_IsDone(_generatorHandle);
        return false;
    }

    public void ComputeLogits()
    {
        // Result.VerifySuccess(NativeMethods.OgaGenerator_ComputeLogits(_generatorHandle));
    }

    public void GenerateNextToken()
    {
        // Result.VerifySuccess(NativeMethods.OgaGenerator_GenerateNextToken(_generatorHandle));
    }

    public Buffer GetSequence(long index)
    {
//        ulong sequenceLength = NativeMethods.OgaGenerator_GetSequenceCount(_generatorHandle, (UIntPtr)index).ToUInt64();
//        IntPtr sequencePtr = NativeMethods.OgaGenerator_GetSequenceData(_generatorHandle, (UIntPtr)index);
//        unsafe
//        {
//            return new ReadOnlySpan<int>(sequencePtr.ToPointer(), (int)sequenceLength);
//        }
        return new Buffer();
    }

}