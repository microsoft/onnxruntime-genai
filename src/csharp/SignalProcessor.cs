// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public static class SignalProcessor
    {
        // Numeric ElementType values matching OgaElementType in the C API:
        // enum OgaElementType { undefined=0, float32=1, ..., int64=7, ... }
        private const int ET_Float32 = 1;
        private const int ET_Int64 = 7;

        /// <summary>
        /// Thin wrapper around the native OgaSplitSignalSegments.
        /// All arguments are OgaTensor handles (IntPtr).
        /// </summary>
        public static void SplitSignalSegments(
            IntPtr inputTensor,
            IntPtr srTensor,
            IntPtr frameMsTensor,
            IntPtr hopMsTensor,
            IntPtr energyThresholdDbTensor,
            IntPtr outputTensor)
        {
            int err = NativeMethods.OgaSplitSignalSegments(
                inputTensor,
                srTensor,
                frameMsTensor,
                hopMsTensor,
                energyThresholdDbTensor,
                outputTensor);

            if (err != 0)
                throw new InvalidOperationException($"OgaSplitSignalSegments failed with error code {err}");
        }

        /// <summary>
        /// Thin wrapper around the native OgaMergeSignalSegments.
        /// All arguments are OgaTensor handles (IntPtr).
        /// </summary>
        public static void MergeSignalSegments(
            IntPtr segmentsTensor,
            IntPtr mergeGapMsTensor,
            IntPtr outputTensor)
        {
            int err = NativeMethods.OgaMergeSignalSegments(
                segmentsTensor,
                mergeGapMsTensor,
                outputTensor);

            if (err != 0)
                throw new InvalidOperationException($"OgaMergeSignalSegments failed with error code {err}");
        }


        /// <summary>
        /// Create a tensor view over a managed float[] using OgaCreateTensorFromBuffer.
        /// </summary>
        public static unsafe IntPtr CreateFloatTensorFromArray(float[] data, long[] shape)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (shape == null) throw new ArgumentNullException(nameof(shape));

            IntPtr tensor;
            fixed (float* p = data)
            {
                Result.VerifySuccess(
                    NativeMethods.OgaCreateTensorFromBuffer(
                        (IntPtr)p,
                        shape,
                        (UIntPtr)shape.Length,
                        (ElementType)ET_Float32,
                        out tensor));
            }
            return tensor;
        }

        /// <summary>
        /// Create a tensor view over a managed long[] using OgaCreateTensorFromBuffer.
        /// </summary>
        public static unsafe IntPtr CreateInt64TensorFromArray(long[] data, long[] shape)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (shape == null) throw new ArgumentNullException(nameof(shape));

            IntPtr tensor;
            fixed (long* p = data)
            {
                Result.VerifySuccess(
                    NativeMethods.OgaCreateTensorFromBuffer(
                        (IntPtr)p,
                        shape,
                        (UIntPtr)shape.Length,
                        (ElementType)ET_Int64,
                        out tensor));
            }
            return tensor;
        }

        /// <summary>
        /// Create an output tensor that points at a caller-owned long[] buffer.
        /// </summary>
        public static unsafe IntPtr CreateOutputInt64Tensor(long[] backingBuffer, long rows, long cols)
        {
            if (backingBuffer == null) throw new ArgumentNullException(nameof(backingBuffer));
            if (rows * cols > backingBuffer.LongLength)
                throw new ArgumentException("backingBuffer too small for requested shape");

            IntPtr tensor;
            long[] shape = new long[] { rows, cols };
            fixed (long* p = backingBuffer)
            {
                Result.VerifySuccess(
                    NativeMethods.OgaCreateTensorFromBuffer(
                        (IntPtr)p,
                        shape,
                        (UIntPtr)shape.Length,
                        (ElementType)ET_Int64,
                        out tensor));
            }
            return tensor;
        }

        /// <summary>
        /// Runs STFT over the input signal and finds the areas of high energy with start/end timestamps in ms.
        /// </summary>
        public static (double Start, double End)[] SplitAndMergeSegments(
            float[] inputSignal,
            int sampleRate,
            int frameMs,
            int hopMs,
            float energyThresholdDb,
            int mergeGapMs)
        {
            if (inputSignal == null || inputSignal.Length == 0)
                throw new ArgumentException("Input array cannot be null or empty", nameof(inputSignal));

            const int MaxSegs = 128;
            long[] splitBacking = new long[MaxSegs * 2];
            long[] mergedBacking = new long[MaxSegs * 2];

            IntPtr input = IntPtr.Zero, sr = IntPtr.Zero, frame = IntPtr.Zero, hop = IntPtr.Zero,
                   thr = IntPtr.Zero, splitOut = IntPtr.Zero, mergeGap = IntPtr.Zero, mergedOut = IntPtr.Zero;

            try
            {
                long[] inputShape = new long[] { 1, inputSignal.Length };

                input = CreateFloatTensorFromArray(inputSignal, inputShape);
                sr = CreateInt64TensorFromArray(new long[] { sampleRate }, new long[] { 1 });
                frame = CreateInt64TensorFromArray(new long[] { frameMs }, new long[] { 1 });
                hop = CreateInt64TensorFromArray(new long[] { hopMs }, new long[] { 1 });
                thr = CreateFloatTensorFromArray(new float[] { energyThresholdDb }, new long[] { 1 });

                splitOut = CreateOutputInt64Tensor(splitBacking, MaxSegs, 2);
                mergedOut = CreateOutputInt64Tensor(mergedBacking, MaxSegs, 2);

                SplitSignalSegments(input, sr, frame, hop, thr, splitOut);

                long[] splitShapeBuf = new long[2];
                Result.VerifySuccess(NativeMethods.OgaTensorGetShape(splitOut, splitShapeBuf, (UIntPtr)splitShapeBuf.Length));
                long splitRows = splitShapeBuf[0];
                long splitCols = splitShapeBuf[1];

                mergeGap = CreateInt64TensorFromArray(new long[] { mergeGapMs }, new long[] { 1 });
                MergeSignalSegments(splitOut, mergeGap, mergedOut);

                long[] mergedShapeBuf = new long[2];
                Result.VerifySuccess(NativeMethods.OgaTensorGetShape(mergedOut, mergedShapeBuf, (UIntPtr)mergedShapeBuf.Length));
                long mergedRowsDbg = mergedShapeBuf[0];
                long mergedColsDbg = mergedShapeBuf[1];

                long[] shapeBuf = new long[2];
                Result.VerifySuccess(NativeMethods.OgaTensorGetShape(mergedOut, shapeBuf, (UIntPtr)shapeBuf.Length));
                long mergedRows = shapeBuf[0];
                long mergedCols = shapeBuf[1];
                if (mergedCols != 2)
                    throw new InvalidOperationException($"Expected merged output with 2 columns, got {mergedCols}");

                // Convert to array of start/end tuples.
                var result = new List<(double Start, double End)>();
                for (int i = 0; i < mergedRows; ++i)
                {
                    long start = mergedBacking[i * 2 + 0];
                    long end = mergedBacking[i * 2 + 1];
                    if (start == 0 && end == 0) continue;
                    result.Add((start, end));
                }

                return result.ToArray();
            }
            finally
            {
                if (input != IntPtr.Zero) NativeMethods.OgaDestroyTensor(input);
                if (sr != IntPtr.Zero) NativeMethods.OgaDestroyTensor(sr);
                if (frame != IntPtr.Zero) NativeMethods.OgaDestroyTensor(frame);
                if (hop != IntPtr.Zero) NativeMethods.OgaDestroyTensor(hop);
                if (thr != IntPtr.Zero) NativeMethods.OgaDestroyTensor(thr);
                if (splitOut != IntPtr.Zero) NativeMethods.OgaDestroyTensor(splitOut);
                if (mergeGap != IntPtr.Zero) NativeMethods.OgaDestroyTensor(mergeGap);
                if (mergedOut != IntPtr.Zero) NativeMethods.OgaDestroyTensor(mergedOut);
            }
        }
    }
}
