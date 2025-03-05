// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class Tokenizer : Tokenizers.Tokenizer, IDisposable
    {
        private IntPtr _tokenizerHandle;

        public Tokenizer(Model model)
        {
            if (model is null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            Result.VerifySuccess(NativeMethods.OgaCreateTokenizer(model.Handle, out _tokenizerHandle));
        }

        public Sequences EncodeBatch(string[] strings)
        {
            if (strings is null)
            {
                throw new ArgumentNullException(nameof(strings));
            }

            Result.VerifySuccess(NativeMethods.OgaCreateSequences(out IntPtr nativeSequences));
            try
            {
                foreach (string str in strings)
                {
                    Result.VerifySuccess(NativeMethods.OgaTokenizerEncode(_tokenizerHandle, StringUtils.ToNullTerminatedUtf8(str), nativeSequences));
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
            if (sequences is null)
            {
                throw new ArgumentNullException(nameof(sequences));
            }

            string[] result = new string[sequences.NumSequences];
            for (ulong i = 0; i < sequences.NumSequences; i++)
            {
                result[i] = Decode(sequences[i]);
            }

            return result;
        }

        public Sequences Encode(string str)
        {
            if (str is null)
            {
                throw new ArgumentNullException(nameof(str));
            }

            return Encode(str.AsSpan());
        }

        public Sequences Encode(ReadOnlySpan<char> str)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateSequences(out IntPtr nativeSequences));
            try
            {
                Result.VerifySuccess(NativeMethods.OgaTokenizerEncode(_tokenizerHandle, StringUtils.ToNullTerminatedUtf8(str), nativeSequences));
                return new Sequences(nativeSequences);
            }
            catch
            {
                NativeMethods.OgaDestroySequences(nativeSequences);
                throw;
            }
        }

        public unsafe string Decode(ReadOnlySpan<int> sequence)
        {
            IntPtr outStr;

            fixed (int* sequencePtr = sequence)
            {
                Result.VerifySuccess(NativeMethods.OgaTokenizerDecode(_tokenizerHandle, sequencePtr, (UIntPtr)sequence.Length, out outStr));
            }

            try
            {
                return StringUtils.FromNullTerminatedUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
        }

        public string ApplyChatTemplate(string template_str, string messages, bool add_generation_prompt)
        {
            IntPtr outStr = IntPtr.Zero;
            try
            {
                Result.VerifySuccess(NativeMethods.OgaTokenizerApplyChatTemplate(_tokenizerHandle, StringUtils.ToNullTerminatedUtf8(template_str), StringUtils.ToNullTerminatedUtf8(messages), add_generation_prompt, out outStr));
                return StringUtils.FromNullTerminatedUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
        }

        public TokenizerStream CreateStream()
        {
            Result.VerifySuccess(NativeMethods.OgaCreateTokenizerStream(_tokenizerHandle, out nint tokenizerStreamHandle));
            return new TokenizerStream(tokenizerStreamHandle);
        }

        ~Tokenizer()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_tokenizerHandle != IntPtr.Zero)
            {
                NativeMethods.OgaDestroyTokenizer(_tokenizerHandle);
                _tokenizerHandle = IntPtr.Zero;
            }

            GC.SuppressFinalize(this);
        }

        #region Base Tokenizer Overrides
        private static int GetMaxTokenCount(EncodeSettings settings)
        {
            if (settings.MaxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(settings.MaxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            return settings.MaxTokenCount;
        }

        /// <inheritdoc />
        protected override int CountTokens(string text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            Debug.Assert(text is null || textSpan.SequenceEqual(text.AsSpan()));

            using Sequences sequences = Encode(textSpan);
            Debug.Assert(sequences.NumSequences == 1);

            return Math.Min(GetMaxTokenCount(settings), sequences[0].Length);
        }

        /// <inheritdoc />
        protected override EncodeResults<EncodedToken> EncodeToTokens(string text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            Debug.Assert(text is null || textSpan.SequenceEqual(text.AsSpan()));

            int maxTokenCount = GetMaxTokenCount(settings);

            using Sequences sequences = Encode(textSpan);
            if (sequences.NumSequences != 1)
            {
                throw new InvalidOperationException("Expected exactly one sequence.");
            }

            ReadOnlySpan<int> sequence = sequences[0];
            if (sequence.Length > maxTokenCount)
            {
                sequence = sequence.Slice(0, maxTokenCount);
            }

            // Only the token IDs are returned. The Sequences doesn't contain offset information about each token.
            EncodedToken[] tokens = new EncodedToken[sequence.Length];
            for (int i = 0; i < sequence.Length; i++)
            {
                tokens[i] = new EncodedToken(sequence[i], string.Empty, default);
            }

            return new EncodeResults<EncodedToken>() { Tokens = tokens };
        }

        /// <inheritdoc />
        protected override EncodeResults<int> EncodeToIds(string text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            Debug.Assert(text is null || textSpan.SequenceEqual(text.AsSpan()));

            int maxTokenCount = GetMaxTokenCount(settings);

            using Sequences sequences = Encode(textSpan);
            Debug.Assert(sequences.NumSequences == 1);

            ReadOnlySpan<int> sequence = sequences[0];
            if (sequence.Length > maxTokenCount)
            {
                sequence = sequence.Slice(0, maxTokenCount);
            }

            return new EncodeResults<int>() { Tokens = sequence.ToArray() };
        }

        /// <inheritdoc />
        public override string Decode(IEnumerable<int> ids)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            return Decode(ids as int[] ?? ids.ToArray());
        }

        /// <inheritdoc />
        public override unsafe OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            IntPtr outStr;

            int[] idsArray = ids as int[] ?? ids.ToArray();
            fixed (int* sequencePtr = idsArray)
            {
                try
                {
                    Result.VerifySuccess(NativeMethods.OgaTokenizerDecode(_tokenizerHandle, sequencePtr, (UIntPtr)idsArray.Length, out outStr));
                }
                catch
                {
                    idsConsumed = charsWritten = 0;
                    return OperationStatus.InvalidData;
                }
            }

            try
            {
                fixed (char* pDest = destination)
                {
                    charsWritten = Encoding.UTF8.GetChars((byte*)outStr, StringUtils.GetNullTerminatedUtf8Length(outStr), pDest, destination.Length);
                    idsConsumed = idsArray.Length;
                }
            }
            catch (ArgumentException)
            {
                idsConsumed = charsWritten = 0;
                return OperationStatus.DestinationTooSmall;
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }

            return OperationStatus.Done;
        }
#endregion
    }
}
