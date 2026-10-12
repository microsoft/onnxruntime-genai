using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>Native half-open interval in absolute acoustic frames.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTokenMetadataAcousticFrameInterval
    {
        internal long Start;
        internal long Stop;
    }

    /// <summary>Native token with an optional acoustic interval stored by value.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTokenMetadataInput
    {
        internal int TokenId;
        internal int HasTokenAcousticFrameInterval;
        internal NativeTokenMetadataAcousticFrameInterval TokenAcousticFrameInterval;
        internal int HasSpeakerEvidence;
        internal NativeTokenMetadataSpeakerEvidence SpeakerEvidence;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTokenMetadataSpeakerEvidence
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        internal double[] StartProbabilitySums;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        internal double[] StopProbabilitySums;
        internal long StartFrameCount;
        internal long StopFrameCount;
    }

    /// <summary>Native per-call text and optional timestamp payload borrowed from the stream.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTokenMetadataOutput
    {
        internal IntPtr Text;
        internal IntPtr TimestampMetadata;
    }

    /// <summary>Native word and segment arrays completed by one stream operation.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTokenMetadataTimestamp
    {
        internal IntPtr Words;
        internal UIntPtr WordCount;
        internal IntPtr Segments;
        internal UIntPtr SegmentCount;
    }

    /// <summary>Native completed word or segment with acoustic bounds and times.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTokenMetadataTimestampRecord
    {
        internal IntPtr Text;
        internal long StartFrame;
        internal long StopFrame;
        internal double StartTime;
        internal double StopTime;
        internal int HasTimestamp;
        internal IntPtr SpeakerLabel;
    }
}