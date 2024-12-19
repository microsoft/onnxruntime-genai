// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// A container of adapters.
    /// </summary>
    public class Adapters : SafeHandle
    {
        /// <summary>
        /// Constructs an Adapters object with the given model.
        /// </summary>
        /// <param name="model">Reference to a loaded model</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Adapters(Model model) : base(IntPtr.Zero, true)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateAdapters(model.Handle, out handle));
        }

        /// <summary>
        /// Loads the model adapter from the given adapter file path and adapter name.
        /// </summary>
        /// <param name="adapterPath">The path of the adapter.</param>
        /// <param name="adapterName">A unique user supplied adapter identifier.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void LoadAdapter(string adapterFilePath, string adapterName)
        {
            Result.VerifySuccess(NativeMethods.OgaLoadAdapter(handle,
                StringUtils.ToUtf8(adapterFilePath), StringUtils.ToUtf8(adapterName)));
        }

        /// <summary>
        /// Unloads the adapter with the given identifier from the previosly loaded adapters. If the
        /// adapter is not found, or if it cannot be unloaded (when it is in use), an error is returned.
        /// </summary>
        /// <param name="adapterName"></param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void UnloadAdapter(string adapterName)
        {
            Result.VerifySuccess(NativeMethods.OgaUnloadAdapter(handle, StringUtils.ToUtf8(adapterName)));
        }

        internal IntPtr Handle { get { return handle; } }

        /// <summary>
        /// Implement SafeHandle override.
        /// </summary>
        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            NativeMethods.OgaDestroyAdapters(handle);
            handle = IntPtr.Zero;
            return true;
        }
    }
}