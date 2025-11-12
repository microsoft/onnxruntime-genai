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

        public void SetModelInput(string name, Tensor value)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_SetModelInput(_generatorHandle, StringUtils.ToUtf8(name), value.Handle));
        }

        public void SetInputs(NamedTensors namedTensors)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_SetInputs(_generatorHandle, namedTensors.Handle));
        }

        public void AppendTokens(ReadOnlySpan<int> inputIDs)
        {
            unsafe
            {
                fixed (int* inputIDsPtr = inputIDs)
                {
                    Result.VerifySuccess(NativeMethods.OgaGenerator_AppendTokens(_generatorHandle, inputIDsPtr, (UIntPtr)inputIDs.Length));
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

        /// <summary>
        /// Rewinds the generator to the given newLength.
        /// Throw on error
        /// </summary>
        /// <param name="newLength"></param>
        public void RewindTo(ulong newLength)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_RewindTo(_generatorHandle, (UIntPtr)newLength));
        }

        public ReadOnlySpan<int> GetNextTokens()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GetNextTokens(_generatorHandle, out IntPtr tokenIds, out UIntPtr tokenCount));
            unsafe
            {
                return new ReadOnlySpan<int>(tokenIds.ToPointer(), (int)tokenCount.ToUInt64());
            }
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
        /// Fetches and returns the input tensor with the given name.
        /// Throw on error
        /// </summary>
        /// <param name="inputName"></param>
        /// <returns>a disposable instance of Tensor</returns>
        public Tensor GetInput(string inputName)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GetInput(_generatorHandle,
                                                                     StringUtils.ToUtf8(inputName),
                                                                     out IntPtr inputTensor));
            return new Tensor(inputTensor);
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
