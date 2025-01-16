// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// A TokenizerStream is used to convert individual tokens when using Generator.GenerateNextToken.
    /// </summary>
    public class TokenizerStream : IDisposable
    {
        private IntPtr _tokenizerStreamHandle;
        private bool _disposed = false;

        internal TokenizerStream(IntPtr tokenizerStreamHandle)
        {
            _tokenizerStreamHandle = tokenizerStreamHandle;
        }

        internal IntPtr Handle { get { return _tokenizerStreamHandle; } }

        /// <summary>
        /// Decode one token.
        /// </summary>
        /// <param name="token">The token.</param>
        /// <returns>
        /// The decoded result.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public string Decode(int token)
        {
            IntPtr decodedStr = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamDecode(_tokenizerStreamHandle, token, out decodedStr));
            return StringUtils.FromUtf8(decodedStr);
        }

        ~TokenizerStream()
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
            NativeMethods.OgaDestroyTokenizerStream(_tokenizerStreamHandle);
            _tokenizerStreamHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
