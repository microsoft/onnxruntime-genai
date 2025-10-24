// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public static class SignalProcessor
    {
        // Numeric ElementType values matching OgaElementType in the C API:
        // enum OgaElementType { undefined=0, float32=1, ..., int64=7, ... }
        private const int ET_Float32 = 1;
        private const int ET_Int64   = 7;

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
        /// Caller must pin the array for the duration of native usage or use it only for the call.
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
        /// The native op will write segment indices (start,end) pairs into it.
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

#if DEBUG
        /// <summary>
        /// Minimal smoketest that allocates tiny inputs and calls the native split op.
        /// Note: this preallocates a fixed-size output buffer (e.g., 64x2). Adjust as needed.
        /// </summary>
        public static void TestSplitSignalSegments()
        {
            Console.WriteLine("=== SignalProcessor.TestSplitSignalSegments 22===");

            // dummy audio (1xN)
            float[] audio = new float[] { 0.1f, 0.2f, 0.0f, 0.3f, 0.0f, 0.0f, 0.25f, 0.0f };
            long[] inputShape = new long[] { 1, audio.Length };

            // parameters (1,)
            long[] srVal    = new long[] { 16000 };
            long[] frameVal = new long[] { 25 };
            long[] hopVal   = new long[] { 10 };
            float[] thrVal  = new float[] { -40.0f };

            // prealloc output: up to 64 segments (rows) x 2 (start,end)
            const int MaxSegs = 64;
            long[] outputBacking = new long[MaxSegs * 2];

            IntPtr input = IntPtr.Zero, sr = IntPtr.Zero, frame = IntPtr.Zero, hop = IntPtr.Zero, thr = IntPtr.Zero, output = IntPtr.Zero;

            try
            {
                input = CreateFloatTensorFromArray(audio, inputShape);
                sr    = CreateInt64TensorFromArray(srVal,    new long[] { 1 });
                frame = CreateInt64TensorFromArray(frameVal, new long[] { 1 });
                hop   = CreateInt64TensorFromArray(hopVal,   new long[] { 1 });

                // energy threshold tensor (float, shape [1])
                {
                    long[] s = new long[] { 1 };
                    unsafe
                    {
                        fixed (float* p = thrVal)
                        {
                            Result.VerifySuccess(
                                NativeMethods.OgaCreateTensorFromBuffer(
                                    (IntPtr)p, s, (UIntPtr)s.Length, (ElementType)ET_Float32, out thr));
                        }
                    }
                }

                output = CreateOutputInt64Tensor(outputBacking, MaxSegs, 2);

            Console.WriteLine("=== SignalProcessor.TestSplitSignalSegments 23===");

                // call native
                SplitSignalSegments(input, sr, frame, hop, thr, output);
            Console.WriteLine("=== SignalProcessor.TestSplitSignalSegments 24===");

                // Read back the real shape to know how many segments were produced
                Result.VerifySuccess(NativeMethods.OgaTensorGetShapeRank(output, out UIntPtr rank));
                if ((ulong)rank != 2UL) throw new InvalidOperationException("Output rank != 2");
                long[] shapeBuf = new long[2];
                Result.VerifySuccess(NativeMethods.OgaTensorGetShape(output, shapeBuf, (UIntPtr)shapeBuf.Length));

                long rows = shapeBuf[0];
                long cols = shapeBuf[1];
                Console.WriteLine($"Output shape: [{rows}, {cols}]");

                // Print first few segment pairs from backing buffer
                long count = Math.Min(rows, 8);
                for (int i = 0; i < count; ++i)
                {
                    long start = outputBacking[i * 2 + 0];
                    long end   = outputBacking[i * 2 + 1];
                    Console.WriteLine($"seg[{i}] = ({start}, {end})");
                }

                Console.WriteLine("✅ TestSplitSignalSegments completed.");
            }
            finally
            {
                if (input != IntPtr.Zero)  NativeMethods.OgaDestroyTensor(input);
                if (sr != IntPtr.Zero)     NativeMethods.OgaDestroyTensor(sr);
                if (frame != IntPtr.Zero)  NativeMethods.OgaDestroyTensor(frame);
                if (hop != IntPtr.Zero)    NativeMethods.OgaDestroyTensor(hop);
                if (thr != IntPtr.Zero)    NativeMethods.OgaDestroyTensor(thr);
                if (output != IntPtr.Zero) NativeMethods.OgaDestroyTensor(output);
            }
        }















public static (double Start, double End)[] SplitAndMergeSegments(
    float[] audio,
    int sampleRate,
    int frameMs,
    int hopMs,
    float energyThresholdDb,
    int mergeGapMs,
    bool returnInMilliseconds = false)
{
    if (audio == null || audio.Length == 0)
        throw new ArgumentException("Audio array cannot be null or empty", nameof(audio));

    const int MaxSegs = 1024;
    long[] splitBacking = new long[MaxSegs * 2];
    long[] mergedBacking = new long[MaxSegs * 2];

    // ✅ all tensor handles declared here
    IntPtr input = IntPtr.Zero, sr = IntPtr.Zero, frame = IntPtr.Zero, hop = IntPtr.Zero,
           thr = IntPtr.Zero, splitOut = IntPtr.Zero, mergeGap = IntPtr.Zero, mergedOut = IntPtr.Zero;

    try
    {
        long[] inputShape = new long[] { 1, audio.Length };

        // --- Build tensors ---
        input = CreateFloatTensorFromArray(audio, inputShape);
        sr    = CreateInt64TensorFromArray(new long[] { sampleRate }, new long[] { 1 });
        frame = CreateInt64TensorFromArray(new long[] { frameMs }, new long[] { 1 });
        hop   = CreateInt64TensorFromArray(new long[] { hopMs }, new long[] { 1 });
        thr   = CreateFloatTensorFromArray(new float[] { energyThresholdDb }, new long[] { 1 });

        // Output buffers
        splitOut  = CreateOutputInt64Tensor(splitBacking, MaxSegs, 2);
        mergedOut = CreateOutputInt64Tensor(mergedBacking, MaxSegs, 2);

        // --- 1. Split ---
        SplitSignalSegments(input, sr, frame, hop, thr, splitOut);
        
        // --- 🔍 Debug print split-out tensor ---
Console.WriteLine("=== Debug: Split output preview ===");
long[] splitShapeBuf = new long[2];
Result.VerifySuccess(NativeMethods.OgaTensorGetShape(splitOut, splitShapeBuf, (UIntPtr)splitShapeBuf.Length));
long splitRows = splitShapeBuf[0];
long splitCols = splitShapeBuf[1];
Console.WriteLine($"Split output shape: [{splitRows}, {splitCols}]");

for (int i = 0; i < Math.Min(splitRows, 8); ++i)
{
    long start = splitBacking[i * 2 + 0];
    long end   = splitBacking[i * 2 + 1];
    Console.WriteLine($"  split[{i}] = ({start}, {end})");
}
Console.WriteLine("==============================\n");
        // --- 2. Merge ---
        mergeGap = CreateInt64TensorFromArray(new long[] { mergeGapMs }, new long[] { 1 });
        MergeSignalSegments(splitOut, mergeGap, mergedOut);

        // --- 🔍 Debug print merged-out tensor ---
Console.WriteLine("=== Debug: Merge output preview ===");
long[] mergedShapeBuf = new long[2];
Result.VerifySuccess(NativeMethods.OgaTensorGetShape(mergedOut, mergedShapeBuf, (UIntPtr)mergedShapeBuf.Length));
long mergedRowsDbg = mergedShapeBuf[0];
long mergedColsDbg = mergedShapeBuf[1];
Console.WriteLine($"Merged output shape: [{mergedRowsDbg}, {mergedColsDbg}]");

for (int i = 0; i < Math.Min(mergedRowsDbg, 8); ++i)
{
    long start = mergedBacking[i * 2 + 0];
    long end   = mergedBacking[i * 2 + 1];
    Console.WriteLine($"  merged[{i}] = ({start}, {end})");
}
Console.WriteLine("==============================\n");

        // --- Read merged shape ---
        long[] shapeBuf = new long[2];
        Result.VerifySuccess(NativeMethods.OgaTensorGetShape(mergedOut, shapeBuf, (UIntPtr)shapeBuf.Length));
        long mergedRows = shapeBuf[0];
        long mergedCols = shapeBuf[1];
        if (mergedCols != 2)
            throw new InvalidOperationException($"Expected merged output with 2 columns, got {mergedCols}");

        // --- Convert to managed tuples ---
        var result = new (double Start, double End)[mergedRows];
        for (int i = 0; i < mergedRows; ++i)
        {
            long start = mergedBacking[i * 2 + 0];
            long end   = mergedBacking[i * 2 + 1];
            if (returnInMilliseconds)
            {
                double startMs = (start * 1000.0) / sampleRate;
                double endMs   = (end   * 1000.0) / sampleRate;
                result[i] = (startMs, endMs);
            }
            else
            {
                result[i] = (start, end);
            }
        }

        return result;
    }
    finally
    {
        if (input     != IntPtr.Zero) NativeMethods.OgaDestroyTensor(input);
        if (sr        != IntPtr.Zero) NativeMethods.OgaDestroyTensor(sr);
        if (frame     != IntPtr.Zero) NativeMethods.OgaDestroyTensor(frame);
        if (hop       != IntPtr.Zero) NativeMethods.OgaDestroyTensor(hop);
        if (thr       != IntPtr.Zero) NativeMethods.OgaDestroyTensor(thr);
        if (splitOut  != IntPtr.Zero) NativeMethods.OgaDestroyTensor(splitOut);
        if (mergeGap  != IntPtr.Zero) NativeMethods.OgaDestroyTensor(mergeGap);
        if (mergedOut != IntPtr.Zero) NativeMethods.OgaDestroyTensor(mergedOut);
    }
}







        
#endif
    }
}
