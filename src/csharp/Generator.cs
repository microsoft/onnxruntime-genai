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
        public bool IsDone()
        {
            return NativeMethods.OgaGenerator_IsDone(_generatorHandle) != 0;
        }

        /// <summary>
        /// Appends tokens to the generator.
        /// </summary>
        /// <param name="inputIDs">The tokens to append.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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

        /// <summary>
        /// Appends token sequences to the generator.
        /// </summary>
        /// <param name="sequences">The sequences to append.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void AppendTokenSequences(Sequences sequences)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_AppendTokenSequences(_generatorHandle, sequences.Handle));
        }

        /// <summary>
        /// Computes the logits from the model based on the input ids and the past state. The computed
        /// logits are stored in the generator.
        /// </summary>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void GenerateNextToken()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GenerateNextToken(_generatorHandle));
        }

        /// <summary>
        /// Rewinds the generator to the given length. This is useful when the user wants to rewind the
        /// generator to a specific length and continue generating from that point.
        /// </summary>
        /// <param name="newLength">The desired length in tokens after rewinding.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void RewindTo(ulong newLength)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_RewindTo(_generatorHandle, (UIntPtr)newLength));
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
        /// Returns a copy of the model output identified by the given name as a Tensor.
        /// </summary>
        /// <param name="name">The name of the output needed.</param>
        /// <returns>A disposable instance of Tensor</returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Tensor GetOutput(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GetOutput(_generatorHandle,
                                                                      StringUtils.ToUtf8(name),
                                                                      out IntPtr outputTensor));
            return new Tensor(outputTensor);
        }

        /// <summary>
        /// Sets the adapter with the given adapter name as active.
        /// </summary>
        /// <param name="adapters">The adapters container.</param>
        /// <param name="adapterName">The adapter name that was previously loaded.</param>
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
