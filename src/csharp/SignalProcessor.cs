// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public static class SignalProcessor
    {
        private const int ET_Float32 = 1;
        private const int ET_Int64 = 7;

        // Track pinned buffers for tensors created from managed arrays
        private static readonly ConcurrentDictionary<IntPtr, GCHandle> _tensorPins = new();

        #region Native wrappers

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

        #endregion

        #region Tensor creation helpers (fixed)

        /// <summary>
        /// Create a tensor view over a managed float[].
        /// The underlying buffer is pinned for the lifetime of the tensor.
        /// </summary>
        public static IntPtr CreateFloatTensorFromArray(float[] data, long[] shape)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (shape == null) throw new ArgumentNullException(nameof(shape));

            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);

            IntPtr tensor;
            var status = NativeMethods.OgaCreateTensorFromBuffer(
                handle.AddrOfPinnedObject(),
                shape,
                (UIntPtr)shape.Length,
                (ElementType)ET_Float32,
                out tensor);

            if (status.ToInt64() != 0)
            {
                handle.Free();
                throw new InvalidOperationException($"OgaCreateTensorFromBuffer(float) failed with {status.ToInt64()}");
            }

            if (!_tensorPins.TryAdd(tensor, handle))
            {
                handle.Free();
                throw new InvalidOperationException("Failed to track pinned buffer for float tensor.");
            }

            return tensor;
        }

        /// <summary>
        /// Create a tensor view over a managed long[].
        /// The underlying buffer is pinned for the lifetime of the tensor.
        /// </summary>
        public static IntPtr CreateInt64TensorFromArray(long[] data, long[] shape)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (shape == null) throw new ArgumentNullException(nameof(shape));

            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);

            IntPtr tensor;
            var status = NativeMethods.OgaCreateTensorFromBuffer(
                handle.AddrOfPinnedObject(),
                shape,
                (UIntPtr)shape.Length,
                (ElementType)ET_Int64,
                out tensor);

            if (status.ToInt64() != 0)
            {
                handle.Free();
                throw new InvalidOperationException($"OgaCreateTensorFromBuffer(int64) failed with {status.ToInt64()}");
            }

            if (!_tensorPins.TryAdd(tensor, handle))
            {
                handle.Free();
                throw new InvalidOperationException("Failed to track pinned buffer for int64 tensor.");
            }

            return tensor;
        }

        /// <summary>
        /// Create an output tensor that points at a caller-owned long[] buffer.
        /// The buffer is pinned for the lifetime of the tensor.
        /// </summary>
        public static IntPtr CreateOutputInt64Tensor(long[] backingBuffer, long rows, long cols)
        {
            if (backingBuffer == null) throw new ArgumentNullException(nameof(backingBuffer));
            if (rows * cols > backingBuffer.LongLength)
                throw new ArgumentException("backingBuffer too small for requested shape");

            long[] shape = new long[] { rows, cols };

            var handle = GCHandle.Alloc(backingBuffer, GCHandleType.Pinned);

            IntPtr tensor;
            var status = NativeMethods.OgaCreateTensorFromBuffer(
                handle.AddrOfPinnedObject(),
                shape,
                (UIntPtr)shape.Length,
                (ElementType)ET_Int64,
                out tensor);

            if (status.ToInt64() != 0)
            {
                handle.Free();
                throw new InvalidOperationException($"OgaCreateTensorFromBuffer(output int64) failed with {status.ToInt64()}");
            }

            if (!_tensorPins.TryAdd(tensor, handle))
            {
                handle.Free();
                throw new InvalidOperationException("Failed to track pinned buffer for output tensor.");
            }

            return tensor;
        }

        /// <summary>
        /// Destroy a tensor and release its pinned managed buffer (if any).
        /// </summary>
        private static void SafeDestroyTensor(IntPtr tensor)
        {
            if (tensor == IntPtr.Zero)
                return;

            NativeMethods.OgaDestroyTensor(tensor);

            if (_tensorPins.TryRemove(tensor, out var handle))
            {
                if (handle.IsAllocated)
                    handle.Free();
            }
        }

        #endregion

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

            const int MaxSegs = 1024;

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
                long mergedRows = mergedShapeBuf[0];
                long mergedCols = mergedShapeBuf[1];

                if (mergedCols != 2)
                    throw new InvalidOperationException($"Expected merged output with 2 columns, got {mergedCols}");

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
                SafeDestroyTensor(input);
                SafeDestroyTensor(sr);
                SafeDestroyTensor(frame);
                SafeDestroyTensor(hop);
                SafeDestroyTensor(thr);
                SafeDestroyTensor(splitOut);
                SafeDestroyTensor(mergeGap);
                SafeDestroyTensor(mergedOut);
            }
        }
    }
}
