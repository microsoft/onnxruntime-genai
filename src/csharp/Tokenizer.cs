// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// The Tokenizer class is responsible for converting between text and token ids.
    /// <summary>
    public class Tokenizer : IDisposable
    {
        private IntPtr _tokenizerHandle;
        private bool _disposed = false;

        /// <summary>
        /// Creates a Tokenizer from the given model.
        /// </summary>
        /// <param name="model">The model to use.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Tokenizer(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateTokenizer(model.Handle, out _tokenizerHandle));
        }

        /// <summary>
        /// Encodes an array of strings into a sequence of token ids for each input.
        /// </summary>
        /// <param name="strings">The collection of strings to encode as token ids.</param>
        /// <returns>
        /// A Sequences object with one sequence per input string.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Sequences EncodeBatch(string[] strings)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateSequences(out IntPtr nativeSequences));
            try
            {
                foreach (string str in strings)
                {
                    Result.VerifySuccess(NativeMethods.OgaTokenizerEncode(_tokenizerHandle, StringUtils.ToUtf8(str), nativeSequences));
                }

                return new Sequences(nativeSequences);
            }
            catch
            {
                NativeMethods.OgaDestroySequences(nativeSequences);
                throw;
            }
        }

        /// <summary>
        /// Decodes a batch of sequences of token ids into text.
        /// </summary>
        /// <param name="sequences"> A Sequences object with one or more sequences of token ids.</param>
        /// <returns>
        /// An array of strings with the text representation of each sequence.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public string[] DecodeBatch(Sequences sequences)
        {
            string[] result = new string[sequences.NumSequences];
            for (ulong i = 0; i < sequences.NumSequences; i++)
            {
                result[i] = Decode(sequences[i]);
            }

            return result;
        }

        /// <summary>
        /// Encodes an array of strings into a sequence of token ids for each input.
        /// </summary>
        /// <param name="str">The text to encode as token ids.</param>
        /// <returns>
        /// A Sequences object with a single sequence.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Sequences Encode(string str)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateSequences(out IntPtr nativeSequences));
            try
            {
                Result.VerifySuccess(NativeMethods.OgaTokenizerEncode(_tokenizerHandle, StringUtils.ToUtf8(str), nativeSequences));
                return new Sequences(nativeSequences);
            }
            catch
            {
                NativeMethods.OgaDestroySequences(nativeSequences);
                throw;
            }
        }

        /// <summary>
        /// Decodes a sequence of token ids into text.
        /// </summary>
        /// <param name="sequence">The token ids.</param>
        /// <returns>
        /// The text representation of the sequence.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public string Decode(ReadOnlySpan<int> sequence)
        {
            IntPtr outStr = IntPtr.Zero;
            unsafe
            {
                fixed (int* sequencePtr = sequence)
                {
                    Result.VerifySuccess(NativeMethods.OgaTokenizerDecode(_tokenizerHandle, sequencePtr, (UIntPtr)sequence.Length, out outStr));
                }
            }
            try
            {
                return StringUtils.FromUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
        }

        /// <summary>
        /// Creates a TokenizerStream object for streaming tokenization. This is used with Generator class
        /// to provide each token as it is generated.
        /// </summary>
        /// <returns>
        /// The new TokenizerStream instance.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public TokenizerStream CreateStream()
        {
            IntPtr tokenizerStreamHandle = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaCreateTokenizerStream(_tokenizerHandle, out tokenizerStreamHandle));
            return new TokenizerStream(tokenizerStreamHandle);
        }


        ~Tokenizer()
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
            NativeMethods.OgaDestroyTokenizer(_tokenizerHandle);
            _tokenizerHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
