// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Generator : IDisposable
    {
        private IntPtr _generatorHandle;
        private bool _disposed = false;

        public Generator(Model model, GeneratorParams generatorParams)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateGenerator(model.Handle, generatorParams.Handle, out _generatorHandle));
        }

        public bool IsDone()
        {
            return NativeMethods.OgaGenerator_IsDone(_generatorHandle) != 0;
        }

        public void AppendTokens(ReadOnlySpan<int> tokens)
        {
            unsafe
            {
                fixed (int* tokenIDsPtr = tokens)
                {
                    Result.VerifySuccess(NativeMethods.OgaGenerator_AppendTokens(_generatorHandle, tokenIDsPtr, (UIntPtr)tokens.Length));
                }
            }
        }

        public void AppendTokenSequences(Sequences sequences)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_AppendTokenSequences(_generatorHandle, sequences.Handle));
        }

        public void GenerateNextToken()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GenerateNextToken(_generatorHandle));
        }

        public void RewindTo(ulong index)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_RewindTo(_generatorHandle, (UIntPtr)index));
        }

        public ReadOnlySpan<int> GetSequence(ulong index)
        {
            ulong sequenceLength = NativeMethods.OgaGenerator_GetSequenceCount(_generatorHandle, (UIntPtr)index).ToUInt64();
            IntPtr sequencePtr = NativeMethods.OgaGenerator_GetSequenceData(_generatorHandle, (UIntPtr)index);
            unsafe
            {
                return new ReadOnlySpan<int>(sequencePtr.ToPointer(), (int)sequenceLength);
            }
        }

        /// <summary>
        /// Fetches and returns the output tensor with the given name.
        /// Throw on error
        /// </summary>
        /// <param name="outputName"></param>
        /// <returns>a disposable instance of Tensor</returns>
        public Tensor GetOutput(string outputName)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GetOutput(_generatorHandle,
                                                                      StringUtils.ToUtf8(outputName),
                                                                      out IntPtr outputTensor));
            return new Tensor(outputTensor);
        }

        /// <summary>
        /// Activates one of the loaded adapters.
        /// Throws on error.
        /// </summary>
        /// <param name="adapters">Adapters container</param>
        /// <param name="adapterName">adapter name that was previously loaded</param>
        public void SetActiveAdapter(Adapters adapters, string adapterName)
        {
            Result.VerifySuccess(NativeMethods.OgaSetActiveAdapter(_generatorHandle,
                                                                   adapters.Handle,
                                                                   StringUtils.ToUtf8(adapterName)));
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
            if (_disposed)
            {
                return;
            }
            NativeMethods.OgaDestroyGenerator(_generatorHandle);
            _generatorHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
