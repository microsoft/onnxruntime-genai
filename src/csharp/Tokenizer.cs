// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Text;

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
            IntPtr stringArray = IntPtr.Zero;
            UIntPtr[] stringLengths = new UIntPtr[strings.Length];
            for (int i = 0; i < strings.Length; i++)
            {
                stringLengths[i] = (UIntPtr)UTF8Encoding.UTF8.GetByteCount(strings[i]);
            }

            unsafe
            {
                fixed (UIntPtr* stringLengthsPtr = stringLengths)
                {
                    Result.VerifySuccess(NativeMethods.OgaCreateAllocatedStrings((UIntPtr)strings.Length, stringLengthsPtr, out stringArray));
                }
            }

            try
            {
                for (ulong i = 0; i < (ulong)strings.Length; i++)
                {
                    Result.VerifySuccess(NativeMethods.OgaStringsGetBuffer(stringArray, (UIntPtr)i, out IntPtr buffer));
                    Utils.ToNativeBuffer(strings[i], buffer, (int)stringLengths[i]);
                }
                Result.VerifySuccess(NativeMethods.OgaTokenizerEncodeBatchStrings(_tokenizerHandle, stringArray, out IntPtr nativeSequences));

                return new Sequences(nativeSequences);
            }
            finally
            {
                NativeMethods.OgaDestroyStrings(stringArray);
            }
        }

        public string[] DecodeBatch(Sequences sequences)
        {
            IntPtr stringArray = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaTokenizerDecodeBatchStrings(_tokenizerHandle, sequences.Handle, out stringArray));
            try
            {
                ulong numStrings = NativeMethods.OgaStringsGetCount(stringArray).ToUInt64();
                string[] result = new string[numStrings];
                for (ulong i = 0; i < numStrings; i++)
                {
                    IntPtr outStr = IntPtr.Zero;
                    Result.VerifySuccess(NativeMethods.OgaStringsGetString(stringArray, (UIntPtr)i, out outStr));
                    result[i] = Utils.FromUtf8(outStr);
                }
                return result;
            }
            finally
            {
                NativeMethods.OgaDestroyStrings(stringArray);
            }
        }

        public Sequences Encode(string str)
        {
            IntPtr nativeSequences = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaTokenizerEncode(_tokenizerHandle, Utils.ToUtf8(str), out nativeSequences));
            return new Sequences(nativeSequences);
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
