// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// The Generator class generates output using a model and generator parameters.
    /// </summary>
    public class Generator : IDisposable
    {
        private IntPtr _generatorHandle;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a Generator object with the given model and generator parameters.
        /// <param name="model">The model to use.</param>
        /// <param name="generatorParams">The generator parameters.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Generator(Model model, GeneratorParams generatorParams)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateGenerator(model.Handle, generatorParams.Handle, out _generatorHandle));
        }

        /// <summary>
        /// Checks if the generation process is done.
        /// </summary>
        /// <returns>
        /// True if the generation process is done, false otherwise.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public bool IsDone()
        {
            return NativeMethods.OgaGenerator_IsDone(_generatorHandle);
        }

        /// <summary>
        /// Computes the logits for the next token in the sequence.
        /// </summary>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void ComputeLogits()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_ComputeLogits(_generatorHandle));
        }

        /// <summary>
        /// Generates the next token in the sequence.
        /// </summary>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void GenerateNextToken()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GenerateNextToken(_generatorHandle));
        }

        /// <summary>
        /// Retrieves a sequence of token ids for the specified sequence index.
        /// </summary>
        /// <param name="index">The index of the sequence.</param>
        /// <returns>
        /// A ReadOnlySpan of integers with the sequence token ids.
        /// </returns>
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
        /// </summary>
        /// <param name="outputName"></param>
        /// <returns>A disposable instance of Tensor</returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Tensor GetOutput(string outputName)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GetOutput(_generatorHandle,
                                                                      StringUtils.ToUtf8(outputName),
                                                                      out IntPtr outputTensor));
            return new Tensor(outputTensor);
        }

        /// <summary>
        /// Activates one of the loaded adapters.
        /// </summary>
        /// <param name="adapters">Adapters container</param>
        /// <param name="adapterName">adapter name that was previously loaded</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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
