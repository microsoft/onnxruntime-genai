// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public enum DeviceType : long
    {
        Auto = 0,
        CPU = 1,
        CUDA = 2
    }

    public class Model : IDisposable
    {
        private IntPtr _modelHandle;

        public Model(string modelPath, DeviceType deviceType)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateModel(Utils.ToUtf8(modelPath), deviceType, out _modelHandle));
        }

        internal IntPtr Handle { get { return _modelHandle; } }

        public int[][] Generate(GeneratorParams generatorParams)
        {
            IntPtr nativeSequences = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaGenerate(_modelHandle, generatorParams.Handle, out nativeSequences));
            try
            {
                ulong batchSize = NativeMethods.OgaSequencesCount(nativeSequences).ToUInt64();
                int[][] sequences = new int[batchSize][];

                for (ulong sequenceIndex = 0; sequenceIndex < batchSize; sequenceIndex++)
                {
                    ulong sequenceLength = NativeMethods.OgaSequencesGetSequenceCount(nativeSequences, (UIntPtr)sequenceIndex).ToUInt64();
                    sequences[sequenceIndex] = new int[sequenceLength];
                    IntPtr sequencePtr = NativeMethods.OgaSequencesGetSequenceData(nativeSequences, (UIntPtr)sequenceIndex);
                    Marshal.Copy(sequencePtr, sequences[sequenceIndex], 0, sequences[sequenceIndex].Length);
                }

                return sequences;
            }
            finally
            {
                NativeMethods.OgaDestroySequences(nativeSequences);
            }
        }

        ~Model()
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
            if (_modelHandle != IntPtr.Zero)
            {
                NativeMethods.OgaDestroyModel(_modelHandle);
                _modelHandle = IntPtr.Zero;
            }
        }
    }
}
