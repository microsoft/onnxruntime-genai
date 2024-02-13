// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Generator : IDisposable
    {
        private IntPtr _generatorHandle;

        public Generator(Model model, GeneratorParams generatorParams)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateGenerator(model.Handle, generatorParams.Handle, out _generatorHandle));
        }

        public bool IsDone()
        {
            return NativeMethods.OgaGenerator_IsDone(_generatorHandle);
        }

        public void ComputeLogits()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_ComputeLogits(_generatorHandle));
        }

        public void GenerateNextTokenTop()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GenerateNextToken_Top(_generatorHandle));
        }

        public int[] GetSequence(int index)
        {
            IntPtr nullPtr = IntPtr.Zero;
            UIntPtr sequenceLength = UIntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaGenerator_GetSequence(_generatorHandle, index, nullPtr, out sequenceLength));
            int[] sequence = new int[sequenceLength.ToUInt64()];
            GCHandle handle = GCHandle.Alloc(sequence, GCHandleType.Pinned);
            try
            {
                IntPtr sequencePtr = handle.AddrOfPinnedObject();
                Result.VerifySuccess(NativeMethods.OgaGenerator_GetSequence(_generatorHandle, index, sequencePtr, out sequenceLength));
            }
            finally
            {
                if (handle.IsAllocated)
                {
                    handle.Free();
                }
            }

            return sequence;
        }

        ~Generator()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_generatorHandle != IntPtr.Zero)
            {
                NativeMethods.OgaDestroyGenerator(_generatorHandle);
                _generatorHandle = IntPtr.Zero;
            }
        }
    }
}
