// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    ///<summary>
    /// Represents the parameters used for generating sequences with a model.
    ///</summary>
    public class GeneratorParams : IDisposable
    {
        private IntPtr _generatorParamsHandle;
        private bool _disposed = false;

        /// <summary>
        /// Construct a GeneratorParams.
        /// </summary>
        /// <param name="model">The model to use.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public GeneratorParams(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateGeneratorParams(model.Handle, out _generatorParamsHandle));
        }

        internal IntPtr Handle { get { return _generatorParamsHandle; } }

        /// <summary>
        /// Set seach option with double value.
        /// </summary>
        /// <param name="searchOption">The option name</param>
        /// <param name="value">The option value</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void SetSearchOption(string searchOption, double value)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetSearchNumber(_generatorParamsHandle, StringUtils.ToUtf8(searchOption), value));
        }

        /// <summary>
        /// Set seach option with boolean value.
        /// </summary>
        /// <param name="searchOption">The option name</param>
        /// <param name="value">The option value</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void SetSearchOption(string searchOption, bool value)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetSearchBool(_generatorParamsHandle, StringUtils.ToUtf8(searchOption), value));
        }

        /// <summary>
        /// Try graph capture.
        /// </summary>
        /// <param name="maxBatchSize">The max batch size</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void TryGraphCaptureWithMaxBatchSize(int maxBatchSize)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(_generatorParamsHandle, maxBatchSize));
        }

        /// <summary>
        /// Sets the prompt/s for model execution.
        /// </summary>
        /// <param name="inputIDs">The encoded input ids.</param>
        /// <param name="sequenceLength">The sequence length.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void SetInputIDs(ReadOnlySpan<int> inputIDs, ulong sequenceLength, ulong batchSize)
        {
            unsafe
            {
                fixed (int* inputIDsPtr = inputIDs)
                {
                    Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetInputIDs(_generatorParamsHandle, inputIDsPtr, (UIntPtr)inputIDs.Length, (UIntPtr)sequenceLength, (UIntPtr)batchSize));
                }
            }
        }

        /// <summary>
        ///  Sets the sequences for model execution.
        /// </summary>
        /// <param name="sequences">The Sequences containing the encoded prompt.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void SetInputSequences(Sequences sequences)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetInputSequences(_generatorParamsHandle, sequences.Handle));
        }

        /// <summary>
        /// Add a Tensor as a model input.
        /// </summary>
        /// <param name="name">The name of the model input the tensor will provide.</param>
        /// <param name="value">The tensor value.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void SetModelInput(string name, Tensor value)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetModelInput(_generatorParamsHandle, StringUtils.ToUtf8(name), value.Handle));
        }

        /// <summary>
        /// Add a NamedTensors as a model input.
        /// </summary>
        /// <param name="namedTensors">The NamedTensors value.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void SetInputs(NamedTensors namedTensors)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetInputs(_generatorParamsHandle, namedTensors.Handle));
        }

        ~GeneratorParams()
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
            NativeMethods.OgaDestroyGeneratorParams(_generatorParamsHandle);
            _generatorParamsHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
