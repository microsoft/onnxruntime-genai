// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Tokenizer : IDisposable
    {
        private IntPtr _tokenizerHandle;
        private bool _disposed = false;

        public Tokenizer(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateTokenizer(model.Handle, out _tokenizerHandle));
        }

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

        public string[] DecodeBatch(Sequences sequences)
        {
            string[] result = new string[sequences.NumSequences];
            for (ulong i = 0; i < sequences.NumSequences; i++)
            {
                result[i] = Decode(sequences[i]);
            }

            return result;
        }

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
