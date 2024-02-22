// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

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
            IntPtr nativeSequences = IntPtr.Zero;
            IntPtr stringArray = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaCreateStringArray(out stringArray));
            try
            {
                foreach (string s in strings)
                {
                    Result.VerifySuccess(NativeMethods.OgaStringArrayAddString(stringArray, Utils.ToUtf8(s)));
                }
                Result.VerifySuccess(NativeMethods.OgaTokenizerEncodeBatch(_tokenizerHandle, stringArray, out nativeSequences));
            }
            finally
            {
                NativeMethods.OgaDestroyStringArray(stringArray);
            }

            return new Sequences(nativeSequences);
        }

        public string[] DecodeBatch(Sequences sequences)
        {
            IntPtr stringArray = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaTokenizerDecodeBatch(_tokenizerHandle, sequences.Handle, out stringArray));
            try
            {
                ulong numStrings = NativeMethods.OgaStringArrayGetCount(stringArray).ToUInt64();
                string[] result = new string[numStrings];
                for (ulong i = 0; i < numStrings; i++)
                {
                    IntPtr outStr = IntPtr.Zero;
                    Result.VerifySuccess(NativeMethods.OgaStringArrayGetString(stringArray, (UIntPtr)i, out outStr));
                    result[i] = Utils.FromUtf8(outStr);
                }
                return result;
            }
            finally
            {
                NativeMethods.OgaDestroyStringArray(stringArray);
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
                return Utils.FromUtf8(outStr);
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
