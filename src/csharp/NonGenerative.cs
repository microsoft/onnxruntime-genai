// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.Win32.SafeHandles;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    internal abstract class OgaSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        protected OgaSafeHandle() : base(true) { }
        internal void Initialize(IntPtr value) { SetHandle(value); }
    }

    internal sealed class DirectoryTokenizerHandle : OgaSafeHandle
    {
        protected override bool ReleaseHandle() { NativeMethods.OgaDestroyDirectoryTokenizer(handle); return true; }
    }

    internal sealed class RankingSessionHandle : OgaSafeHandle
    {
        protected override bool ReleaseHandle() { NativeMethods.OgaDestroyRankingSession(handle); return true; }
    }

    internal sealed class DecisionSessionHandle : OgaSafeHandle
    {
        protected override bool ReleaseHandle() { NativeMethods.OgaDestroyDecisionSession(handle); return true; }
    }

    internal sealed class ComponentSessionHandle : OgaSafeHandle
    {
        protected override bool ReleaseHandle() { NativeMethods.OgaDestroyComponentSession(handle); return true; }
    }

    internal static class SafeHandleAccess
    {
        internal static T Use<T>(OgaSafeHandle handle, Func<IntPtr, T> operation)
        {
            bool added = false;
            try
            {
                handle.DangerousAddRef(ref added);
                if (handle.IsClosed || handle.IsInvalid)
                    throw new ObjectDisposedException(handle.GetType().Name);
                return operation(handle.DangerousGetHandle());
            }
            finally
            {
                if (added) handle.DangerousRelease();
            }
        }

        internal static void Use(OgaSafeHandle handle, Action<IntPtr> operation)
        {
            Use(handle, value => { operation(value); return 0; });
        }
    }

    internal sealed class NativeStringArray : IDisposable
    {
        private readonly IntPtr[] _strings;
        internal IntPtr Pointer { get; private set; }
        internal UIntPtr Count { get { return (UIntPtr)(uint)_strings.Length; } }

        internal NativeStringArray(IEnumerable<string> values)
        {
            var items = values == null ? new List<string>() : new List<string>(values);
            _strings = new IntPtr[items.Count];
            try
            {
                for (int i = 0; i < items.Count; ++i)
                {
                    if (items[i] == null) throw new ArgumentException("String lists cannot contain null.", nameof(values));
                    byte[] utf8 = StringUtils.ToUtf8(items[i]);
                    _strings[i] = Marshal.AllocHGlobal(utf8.Length);
                    Marshal.Copy(utf8, 0, _strings[i], utf8.Length);
                }
                if (_strings.Length != 0)
                {
                    Pointer = Marshal.AllocHGlobal(IntPtr.Size * _strings.Length);
                    Marshal.Copy(_strings, 0, Pointer, _strings.Length);
                }
            }
            catch { Dispose(); throw; }
        }

        public void Dispose()
        {
            if (Pointer != IntPtr.Zero) { Marshal.FreeHGlobal(Pointer); Pointer = IntPtr.Zero; }
            foreach (IntPtr value in _strings) if (value != IntPtr.Zero) Marshal.FreeHGlobal(value);
        }
    }

    internal sealed class NativeStructuredValue : IDisposable
    {
        private const int MaxDepth = 128;

        private sealed class ReferenceComparer : IEqualityComparer<object>
        {
            internal static readonly ReferenceComparer Instance = new ReferenceComparer();
            public new bool Equals(object left, object right) { return ReferenceEquals(left, right); }
            public int GetHashCode(object value) { return RuntimeHelpers.GetHashCode(value); }
        }

        internal IntPtr Handle { get; private set; }
        private NativeStructuredValue(IntPtr handle) { Handle = handle; }

        internal static NativeStructuredValue Create(object value)
        {
            return Create(value, 0, new HashSet<object>(ReferenceComparer.Instance));
        }

        internal static void Validate(object value)
        {
            Validate(value, 0, new HashSet<object>(ReferenceComparer.Instance));
        }

        private static void EnterContainer(object value, int depth, HashSet<object> active)
        {
            if (depth >= MaxDepth)
                throw new ArgumentException("Structured value exceeds the maximum nesting depth of 128.");
            if (!active.Add(value))
                throw new ArgumentException("Structured value contains a reference cycle.");
        }

        private static void Validate(object value, int depth, HashSet<object> active)
        {
            if (value == null || value is bool || value is string ||
                value is float || value is double || value is decimal)
                return;
            if (IsInteger(value))
            {
                ValidateIntegerRange(value);
                return;
            }
            if (value is IDictionary dictionary)
            {
                EnterContainer(value, depth, active);
                try
                {
                    foreach (DictionaryEntry item in dictionary)
                    {
                        if (!(item.Key is string)) throw new ArgumentException("Structured object keys must be strings.");
                        Validate(item.Value, depth + 1, active);
                    }
                }
                finally { active.Remove(value); }
                return;
            }
            if (value is IEnumerable enumerable)
            {
                EnterContainer(value, depth, active);
                try
                {
                    foreach (object item in enumerable) Validate(item, depth + 1, active);
                }
                finally { active.Remove(value); }
                return;
            }
            throw new ArgumentException("Structured values support null, primitive numbers, strings, dictionaries, and lists.", nameof(value));
        }

        private static NativeStructuredValue Create(object value, int depth, HashSet<object> active)
        {
            IntPtr handle;
            if (value == null)
                Result.VerifySuccess(NativeMethods.OgaCreateStructuredValueNull(out handle));
            else if (value is bool)
                Result.VerifySuccess(NativeMethods.OgaCreateStructuredValueBool((bool)value, out handle));
            else if (value is string)
                Result.VerifySuccess(NativeMethods.OgaCreateStructuredValueString(StringUtils.ToUtf8((string)value), out handle));
            else if (value is float || value is double || value is decimal)
                Result.VerifySuccess(NativeMethods.OgaCreateStructuredValueDouble(Convert.ToDouble(value), out handle));
            else if (IsInteger(value))
            {
                ValidateIntegerRange(value);
                Result.VerifySuccess(NativeMethods.OgaCreateStructuredValueInt64(Convert.ToInt64(value), out handle));
            }
            else if (value is IDictionary dictionary)
            {
                EnterContainer(value, depth, active);
                Result.VerifySuccess(NativeMethods.OgaCreateStructuredValueObject(out handle));
                try
                {
                    foreach (DictionaryEntry item in dictionary)
                    {
                        if (!(item.Key is string)) throw new ArgumentException("Structured object keys must be strings.");
                        using (NativeStructuredValue child = Create(item.Value, depth + 1, active))
                            Result.VerifySuccess(NativeMethods.OgaStructuredValueObjectAppend(handle, StringUtils.ToUtf8((string)item.Key), child.Handle));
                    }
                }
                catch { NativeMethods.OgaDestroyStructuredValue(handle); throw; }
                finally { active.Remove(value); }
            }
            else if (value is IEnumerable enumerable)
            {
                EnterContainer(value, depth, active);
                Result.VerifySuccess(NativeMethods.OgaCreateStructuredValueArray(out handle));
                try
                {
                    foreach (object item in enumerable)
                        using (NativeStructuredValue child = Create(item, depth + 1, active))
                            Result.VerifySuccess(NativeMethods.OgaStructuredValueArrayAppend(handle, child.Handle));
                }
                catch { NativeMethods.OgaDestroyStructuredValue(handle); throw; }
                finally { active.Remove(value); }
            }
            else
                throw new ArgumentException("Structured values support null, primitive numbers, strings, dictionaries, and lists.", nameof(value));
            return new NativeStructuredValue(handle);
        }

        private static bool IsInteger(object value)
        {
            return value is byte || value is sbyte || value is short || value is ushort ||
                   value is int || value is uint || value is long || value is ulong;
        }

        private static void ValidateIntegerRange(object value)
        {
            if (value is ulong unsigned && unsigned > long.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(value), "Structured integers must fit signed int64.");
        }

        internal static object Read(IntPtr value)
        {
            Result.VerifySuccess(NativeMethods.OgaStructuredValueGetType(value, out int type));
            switch (type)
            {
                case 0: return null;
                case 1: Result.VerifySuccess(NativeMethods.OgaStructuredValueGetBool(value, out bool boolean)); return boolean;
                case 2: Result.VerifySuccess(NativeMethods.OgaStructuredValueGetInt64(value, out long integer)); return integer;
                case 3: Result.VerifySuccess(NativeMethods.OgaStructuredValueGetDouble(value, out double number)); return number;
                case 4: Result.VerifySuccess(NativeMethods.OgaStructuredValueGetString(value, out IntPtr text)); return StringUtils.FromUtf8(text);
                case 5:
                    Result.VerifySuccess(NativeMethods.OgaStructuredValueGetCount(value, out UIntPtr arrayCount));
                    var array = new List<object>();
                    for (int i = 0; i < CheckedCount(arrayCount); ++i)
                    {
                        Result.VerifySuccess(NativeMethods.OgaStructuredValueGetArrayItem(value, (UIntPtr)(uint)i, out IntPtr item));
                        array.Add(Read(item));
                    }
                    return array;
                case 6:
                    Result.VerifySuccess(NativeMethods.OgaStructuredValueGetCount(value, out UIntPtr objectCount));
                    var obj = new Dictionary<string, object>();
                    for (int i = 0; i < CheckedCount(objectCount); ++i)
                    {
                        Result.VerifySuccess(NativeMethods.OgaStructuredValueGetObjectItem(value, (UIntPtr)(uint)i, out IntPtr key, out IntPtr item));
                        obj.Add(StringUtils.FromUtf8(key), Read(item));
                    }
                    return obj;
                default: throw new OnnxRuntimeGenAIException("Native runtime returned an unknown structured value type.");
            }
        }

        internal static int CheckedCount(UIntPtr value) { return checked((int)value.ToUInt64()); }
        public void Dispose() { if (Handle != IntPtr.Zero) { NativeMethods.OgaDestroyStructuredValue(Handle); Handle = IntPtr.Zero; } }
    }

    /// <summary>Tokenizer loaded directly from a non-generative package directory.</summary>
    public sealed class DirectoryTokenizer : IDisposable
    {
        private readonly DirectoryTokenizerHandle _handle = new DirectoryTokenizerHandle();
        public DirectoryTokenizer(string packagePath)
        {
            if (packagePath == null) throw new ArgumentNullException(nameof(packagePath));
            Result.VerifySuccess(NativeMethods.OgaCreateDirectoryTokenizer(StringUtils.ToUtf8(packagePath), out IntPtr value));
            _handle.Initialize(value);
        }
        public int[] Encode(string text)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));
            return SafeHandleAccess.Use(_handle, native =>
            {
                Result.VerifySuccess(NativeMethods.OgaDirectoryTokenizerEncode(native, StringUtils.ToUtf8(text), out IntPtr ids));
                try
                {
                    Result.VerifySuccess(NativeMethods.OgaTokenIdsGetData(ids, out IntPtr data, out UIntPtr count));
                    int[] result = new int[NativeStructuredValue.CheckedCount(count)];
                    Marshal.Copy(data, result, 0, result.Length);
                    return result;
                }
                finally { NativeMethods.OgaDestroyTokenIds(ids); }
            });
        }
        public int PadTokenId { get { return SafeHandleAccess.Use(_handle, native => { Result.VerifySuccess(NativeMethods.OgaDirectoryTokenizerGetPadTokenId(native, out int id)); return id; }); } }
        public void Dispose() { _handle.Dispose(); }
    }

    public sealed class StructuredQuestion
    {
        public string Type { get; private set; }
        public object Instructions { get; private set; }
        public object Criteria { get; private set; }
        public StructuredQuestion(string type, object instructions, object criteria = null)
        {
            if (string.IsNullOrEmpty(type)) throw new ArgumentException("Question type is required.", nameof(type));
            NativeStructuredValue.Validate(instructions);
            NativeStructuredValue.Validate(criteria);
            Type = type; Instructions = instructions; Criteria = criteria;
        }
    }

    public sealed class StructuredRequest
    {
        public object State { get; set; }
        public IDictionary<string, StructuredQuestion> Questions { get; private set; }
        public float? Temperature { get; set; }
        public StructuredRequest(object state, IDictionary<string, StructuredQuestion> questions)
        {
            NativeStructuredValue.Validate(state);
            State = state;
            Questions = questions ?? throw new ArgumentNullException(nameof(questions));
        }
    }

    public sealed class FreeFormRankRequest
    {
        public object State { get; set; }
        public object Instructions { get; set; }
        public IDictionary<string, object> Candidates { get; private set; }
        public float? Temperature { get; set; }
        public FreeFormRankRequest(object state, object instructions, IDictionary<string, object> candidates)
        {
            NativeStructuredValue.Validate(state);
            NativeStructuredValue.Validate(instructions);
            if (candidates != null)
                foreach (KeyValuePair<string, object> candidate in candidates)
                    NativeStructuredValue.Validate(candidate.Value);
            State = state; Instructions = instructions;
            Candidates = candidates ?? throw new ArgumentNullException(nameof(candidates));
        }
    }

    public sealed class ModelAnswer
    {
        public string Id { get; internal set; }
        public string Type { get; internal set; }
        public double? Noul { get; internal set; }
        public string Choice { get; internal set; }
        public double? Score { get; internal set; }
        public double? Confidence { get; internal set; }
        public IDictionary<string, double> Probabilities { get; internal set; }
        public IDictionary<string, string> Legend { get; internal set; }
    }

    public sealed class ModelResult
    {
        public string Model { get; internal set; }
        public IList<ModelAnswer> Answers { get; internal set; }
    }

    public sealed class RankedItem
    {
        public ulong Rank { get; internal set; }
        public string Key { get; internal set; }
        public object Value { get; internal set; }
        public double Probability { get; internal set; }
    }

    public sealed class RankingResult
    {
        public string Model { get; internal set; }
        public IList<RankedItem> Items { get; internal set; }
    }

    public sealed class NonGenerativeCacheStats
    {
        public ulong Hits { get; internal set; }
        public ulong Misses { get; internal set; }
        public ulong Evictions { get; internal set; }
        public ulong Entries { get; internal set; }
        public ulong Bytes { get; internal set; }
        public ulong EntryCapacity { get; internal set; }
        public ulong ByteCapacity { get; internal set; }
    }

    public sealed class KevPrefixReuseStats
    {
        public ulong PrefixRuns { get; internal set; }
        public ulong BranchRuns { get; internal set; }
        public ulong FallbackRuns { get; internal set; }
    }

    internal static class NonGenerativeMarshal
    {
        internal static IntPtr BuildRequest(StructuredRequest request)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            Result.VerifySuccess(NativeMethods.OgaCreateStructuredRequest(out IntPtr native));
            try
            {
                using (NativeStructuredValue state = NativeStructuredValue.Create(request.State))
                    Result.VerifySuccess(NativeMethods.OgaStructuredRequestSetState(native, state.Handle));
                foreach (KeyValuePair<string, StructuredQuestion> pair in request.Questions)
                {
                    if (string.IsNullOrEmpty(pair.Key) || pair.Value == null) throw new ArgumentException("Question ids and values are required.");
                    using (NativeStructuredValue instructions = NativeStructuredValue.Create(pair.Value.Instructions))
                    using (NativeStructuredValue criteria = NativeStructuredValue.Create(pair.Value.Criteria))
                    {
                        Result.VerifySuccess(NativeMethods.OgaCreateQuestion(StringUtils.ToUtf8(pair.Value.Type), instructions.Handle, criteria.Handle, out IntPtr question));
                        try { Result.VerifySuccess(NativeMethods.OgaStructuredRequestAddQuestion(native, StringUtils.ToUtf8(pair.Key), question)); }
                        finally { NativeMethods.OgaDestroyQuestion(question); }
                    }
                }
                if (request.Temperature.HasValue)
                    Result.VerifySuccess(NativeMethods.OgaStructuredRequestSetTemperature(native, request.Temperature.Value));
                return native;
            }
            catch { NativeMethods.OgaDestroyStructuredRequest(native); throw; }
        }

        internal static IntPtr BuildRankRequest(FreeFormRankRequest request)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            Result.VerifySuccess(NativeMethods.OgaCreateFreeFormRankRequest(out IntPtr native));
            try
            {
                using (NativeStructuredValue state = NativeStructuredValue.Create(request.State))
                    Result.VerifySuccess(NativeMethods.OgaFreeFormRankRequestSetState(native, state.Handle));
                using (NativeStructuredValue instructions = NativeStructuredValue.Create(request.Instructions))
                    Result.VerifySuccess(NativeMethods.OgaFreeFormRankRequestSetInstructions(native, instructions.Handle));
                foreach (KeyValuePair<string, object> pair in request.Candidates)
                    using (NativeStructuredValue value = NativeStructuredValue.Create(pair.Value))
                        Result.VerifySuccess(NativeMethods.OgaFreeFormRankRequestAddCandidate(native, StringUtils.ToUtf8(pair.Key), value.Handle));
                if (request.Temperature.HasValue)
                    Result.VerifySuccess(NativeMethods.OgaFreeFormRankRequestSetTemperature(native, request.Temperature.Value));
                return native;
            }
            catch { NativeMethods.OgaDestroyFreeFormRankRequest(native); throw; }
        }

        internal static ModelResult ReadModelResult(IntPtr result)
        {
            Result.VerifySuccess(NativeMethods.OgaModelResultGetModel(result, out IntPtr model));
            Result.VerifySuccess(NativeMethods.OgaModelResultGetAnswerCount(result, out UIntPtr count));
            var answers = new List<ModelAnswer>();
            for (int i = 0; i < NativeStructuredValue.CheckedCount(count); ++i)
            {
                UIntPtr index = (UIntPtr)(uint)i;
                Result.VerifySuccess(NativeMethods.OgaModelResultGetAnswerId(result, index, out IntPtr id));
                Result.VerifySuccess(NativeMethods.OgaModelResultGetAnswerType(result, index, out IntPtr type));
                var answer = new ModelAnswer {
                    Id = StringUtils.FromUtf8(id), Type = StringUtils.FromUtf8(type),
                    Probabilities = new Dictionary<string, double>(), Legend = new Dictionary<string, string>()
                };
                Result.VerifySuccess(NativeMethods.OgaModelResultGetAnswerNoul(result, index, out double number, out bool present));
                if (present) answer.Noul = number;
                Result.VerifySuccess(NativeMethods.OgaModelResultGetAnswerChoice(result, index, out IntPtr choice, out present));
                if (present) answer.Choice = StringUtils.FromUtf8(choice);
                Result.VerifySuccess(NativeMethods.OgaModelResultGetAnswerScore(result, index, out number, out present));
                if (present) answer.Score = number;
                Result.VerifySuccess(NativeMethods.OgaModelResultGetAnswerConfidence(result, index, out number, out present));
                if (present) answer.Confidence = number;
                Result.VerifySuccess(NativeMethods.OgaModelResultGetProbabilityCount(result, index, out UIntPtr pc));
                for (int p = 0; p < NativeStructuredValue.CheckedCount(pc); ++p)
                {
                    Result.VerifySuccess(NativeMethods.OgaModelResultGetProbability(result, index, (UIntPtr)(uint)p, out IntPtr key, out double value));
                    answer.Probabilities.Add(StringUtils.FromUtf8(key), value);
                }
                Result.VerifySuccess(NativeMethods.OgaModelResultGetLegendCount(result, index, out UIntPtr lc));
                for (int l = 0; l < NativeStructuredValue.CheckedCount(lc); ++l)
                {
                    Result.VerifySuccess(NativeMethods.OgaModelResultGetLegend(result, index, (UIntPtr)(uint)l, out IntPtr key, out IntPtr value));
                    answer.Legend.Add(StringUtils.FromUtf8(key), StringUtils.FromUtf8(value));
                }
                answers.Add(answer);
            }
            return new ModelResult { Model = StringUtils.FromUtf8(model), Answers = answers };
        }
    }

    public sealed class RankingSession : IDisposable
    {
        private readonly RankingSessionHandle _handle = new RankingSessionHandle();
        public RankingSession(string packagePath, IEnumerable<string> providers = null)
        {
            if (packagePath == null) throw new ArgumentNullException(nameof(packagePath));
            using (var p = new NativeStringArray(providers))
            {
                Result.VerifySuccess(NativeMethods.OgaCreateRankingSession(StringUtils.ToUtf8(packagePath), p.Pointer, p.Count, out IntPtr value));
                _handle.Initialize(value);
            }
        }
        public ModelResult Run(StructuredRequest request)
        {
            IntPtr nativeRequest = NonGenerativeMarshal.BuildRequest(request);
            try { return SafeHandleAccess.Use(_handle, native => { Result.VerifySuccess(NativeMethods.OgaRankingSessionRun(native, nativeRequest, out IntPtr result)); try { return NonGenerativeMarshal.ReadModelResult(result); } finally { NativeMethods.OgaDestroyModelResult(result); } }); }
            finally { NativeMethods.OgaDestroyStructuredRequest(nativeRequest); }
        }
        public RankingResult Rank(FreeFormRankRequest request)
        {
            IntPtr nativeRequest = NonGenerativeMarshal.BuildRankRequest(request);
            try
            {
                return SafeHandleAccess.Use(_handle, native =>
                {
                    Result.VerifySuccess(NativeMethods.OgaRankingSessionRank(native, nativeRequest, out IntPtr result));
                    try
                    {
                        Result.VerifySuccess(NativeMethods.OgaRankingResultGetModel(result, out IntPtr model));
                        Result.VerifySuccess(NativeMethods.OgaRankingResultGetCount(result, out UIntPtr count));
                        var items = new List<RankedItem>();
                        for (int i = 0; i < NativeStructuredValue.CheckedCount(count); ++i)
                        {
                            UIntPtr index = (UIntPtr)(uint)i;
                            Result.VerifySuccess(NativeMethods.OgaRankingResultGetRank(result, index, out UIntPtr rank));
                            Result.VerifySuccess(NativeMethods.OgaRankingResultGetKey(result, index, out IntPtr key));
                            Result.VerifySuccess(NativeMethods.OgaRankingResultGetValue(result, index, out IntPtr value));
                            Result.VerifySuccess(NativeMethods.OgaRankingResultGetProbability(result, index, out double probability));
                            items.Add(new RankedItem { Rank = rank.ToUInt64(), Key = StringUtils.FromUtf8(key), Value = NativeStructuredValue.Read(value), Probability = probability });
                        }
                        return new RankingResult { Model = StringUtils.FromUtf8(model), Items = items };
                    }
                    finally { NativeMethods.OgaDestroyRankingResult(result); }
                });
            }
            finally { NativeMethods.OgaDestroyFreeFormRankRequest(nativeRequest); }
        }
        public void SetCacheCapacity(ulong entries, ulong bytes) { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaRankingSessionSetCacheCapacity(native, (UIntPtr)entries, (UIntPtr)bytes))); }
        public NonGenerativeCacheStats CacheStats { get { return SafeHandleAccess.Use(_handle, native => { Result.VerifySuccess(NativeMethods.OgaRankingSessionGetCacheStats(native, out NativeMethods.NonGenerativeCacheStats s)); return ConvertStats(s); }); } }
        public void ClearCache() { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaRankingSessionClearCache(native))); }
        public void InvalidateCache() { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaRankingSessionInvalidateCache(native))); }
        internal static NonGenerativeCacheStats ConvertStats(NativeMethods.NonGenerativeCacheStats s) { return new NonGenerativeCacheStats { Hits = s.Hits, Misses = s.Misses, Evictions = s.Evictions, Entries = s.Entries.ToUInt64(), Bytes = s.Bytes.ToUInt64(), EntryCapacity = s.EntryCapacity.ToUInt64(), ByteCapacity = s.ByteCapacity.ToUInt64() }; }
        public void Dispose() { _handle.Dispose(); }
    }

    public sealed class DecisionSession : IDisposable
    {
        private readonly DecisionSessionHandle _handle = new DecisionSessionHandle();
        private readonly object _prefixReuseLock = new object();
        public DecisionSession(string packagePath, IEnumerable<string> providers = null)
        {
            if (packagePath == null) throw new ArgumentNullException(nameof(packagePath));
            using (var p = new NativeStringArray(providers))
            {
                Result.VerifySuccess(NativeMethods.OgaCreateDecisionSession(StringUtils.ToUtf8(packagePath), p.Pointer, p.Count, out IntPtr value));
                _handle.Initialize(value);
            }
        }
        private ModelResult Execute(StructuredRequest request, bool decide)
        {
            IntPtr nativeRequest = NonGenerativeMarshal.BuildRequest(request);
            try
            {
                return SafeHandleAccess.Use(_handle, native =>
                {
                    IntPtr status = decide
                        ? NativeMethods.OgaDecisionSessionDecide(native, nativeRequest, out IntPtr result)
                        : NativeMethods.OgaDecisionSessionRun(native, nativeRequest, out result);
                    Result.VerifySuccess(status);
                    try { return NonGenerativeMarshal.ReadModelResult(result); } finally { NativeMethods.OgaDestroyModelResult(result); }
                });
            }
            finally { NativeMethods.OgaDestroyStructuredRequest(nativeRequest); }
        }
        public ModelResult Run(StructuredRequest request) { return Execute(request, false); }
        public ModelResult Decide(StructuredRequest request) { return Execute(request, true); }
        public void SetCacheCapacity(ulong entries, ulong bytes) { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaDecisionSessionSetCacheCapacity(native, (UIntPtr)entries, (UIntPtr)bytes))); }
        public NonGenerativeCacheStats CacheStats { get { return SafeHandleAccess.Use(_handle, native => { Result.VerifySuccess(NativeMethods.OgaDecisionSessionGetCacheStats(native, out NativeMethods.NonGenerativeCacheStats s)); return RankingSession.ConvertStats(s); }); } }
        public bool PrefixReuseEnabled
        {
            get { lock (_prefixReuseLock) { return SafeHandleAccess.Use(_handle, native => { Result.VerifySuccess(NativeMethods.OgaDecisionSessionGetPrefixReuseEnabled(native, out bool enabled)); return enabled; }); } }
            set { lock (_prefixReuseLock) { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaDecisionSessionSetPrefixReuseEnabled(native, value))); } }
        }
        public string PrefixReuseStatus
        {
            get
            {
                lock (_prefixReuseLock)
                {
                    return SafeHandleAccess.Use(_handle, native =>
                    {
                        Result.VerifySuccess(NativeMethods.OgaDecisionSessionCopyPrefixReuseStatus(
                            native, IntPtr.Zero, UIntPtr.Zero, out UIntPtr required));
                        int size = checked((int)required.ToUInt64());
                        IntPtr buffer = Marshal.AllocHGlobal(size);
                        try
                        {
                            Result.VerifySuccess(NativeMethods.OgaDecisionSessionCopyPrefixReuseStatus(
                                native, buffer, required, out UIntPtr copied));
                            if (copied != required) throw new OnnxRuntimeGenAIException("Prefix reuse status size changed during copy.");
                            return StringUtils.FromUtf8(buffer);
                        }
                        finally { Marshal.FreeHGlobal(buffer); }
                    });
                }
            }
        }
        public void SetPrefixCacheCapacity(ulong entries, ulong bytes) { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaDecisionSessionSetPrefixCacheCapacity(native, (UIntPtr)entries, (UIntPtr)bytes))); }
        public NonGenerativeCacheStats PrefixCacheStats { get { return SafeHandleAccess.Use(_handle, native => { Result.VerifySuccess(NativeMethods.OgaDecisionSessionGetPrefixCacheStats(native, out NativeMethods.NonGenerativeCacheStats s)); return RankingSession.ConvertStats(s); }); } }
        public KevPrefixReuseStats PrefixReuseStats { get { return SafeHandleAccess.Use(_handle, native => { Result.VerifySuccess(NativeMethods.OgaDecisionSessionGetPrefixReuseStats(native, out NativeMethods.KevPrefixReuseStats s)); return new KevPrefixReuseStats { PrefixRuns = s.PrefixRuns, BranchRuns = s.BranchRuns, FallbackRuns = s.FallbackRuns }; }); } }
        public void ClearCache() { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaDecisionSessionClearCache(native))); }
        public void InvalidateCache() { SafeHandleAccess.Use(_handle, native => Result.VerifySuccess(NativeMethods.OgaDecisionSessionInvalidateCache(native))); }
        public void Dispose() { _handle.Dispose(); }
    }

    public sealed class ComponentTensor
    {
        public byte[] Data { get; private set; }
        public long[] Shape { get; private set; }
        public ElementType Type { get; private set; }
        public ComponentTensor(byte[] data, long[] shape, ElementType type)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Type = type;
        }
    }

    public sealed class ComponentInputInfo
    {
        public ElementType Type { get; internal set; }
        public IList<long> Shape { get; internal set; }
        public IList<string> SymbolicShape { get; internal set; }
    }

    /// <summary>Low-level, uncached execution of a named exported component.</summary>
    public sealed class ComponentSession : IDisposable
    {
        private readonly ComponentSessionHandle _handle = new ComponentSessionHandle();
        public ComponentSession(string packagePath, string component, IEnumerable<string> providers = null)
        {
            if (packagePath == null) throw new ArgumentNullException(nameof(packagePath));
            if (string.IsNullOrEmpty(component)) throw new ArgumentException("Component name is required.", nameof(component));
            using (var p = new NativeStringArray(providers))
            {
                Result.VerifySuccess(NativeMethods.OgaCreateComponentSession(StringUtils.ToUtf8(packagePath), StringUtils.ToUtf8(component), p.Pointer, p.Count, out IntPtr value));
                _handle.Initialize(value);
            }
        }
        private IList<string> Names(bool input)
        {
            return SafeHandleAccess.Use(_handle, h =>
            {
                UIntPtr count;
                if (input) Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputCount(h, out count));
                else Result.VerifySuccess(NativeMethods.OgaComponentSessionGetOutputCount(h, out count));
                var names = new List<string>();
                for (int i = 0; i < NativeStructuredValue.CheckedCount(count); ++i)
                {
                    IntPtr name;
                    if (input) Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputName(h, (UIntPtr)(uint)i, out name));
                    else Result.VerifySuccess(NativeMethods.OgaComponentSessionGetOutputName(h, (UIntPtr)(uint)i, out name));
                    names.Add(StringUtils.FromUtf8(name));
                }
                return names;
            });
        }
        public IList<string> InputNames { get { return Names(true); } }
        public IList<string> OutputNames { get { return Names(false); } }
        public IDictionary<string, ComponentInputInfo> InputInfo
        {
            get
            {
                return SafeHandleAccess.Use(_handle, h =>
                {
                    Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputCount(h, out UIntPtr count));
                    var result = new Dictionary<string, ComponentInputInfo>();
                    for (int i = 0; i < NativeStructuredValue.CheckedCount(count); ++i)
                    {
                        UIntPtr index = (UIntPtr)(uint)i;
                        Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputName(h, index, out IntPtr name));
                        Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputType(h, index, out ElementType type));
                        Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputShapeRank(h, index, out UIntPtr rank));
                        var shape = new List<long>();
                        var symbols = new List<string>();
                        for (int d = 0; d < NativeStructuredValue.CheckedCount(rank); ++d)
                        {
                            Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputShapeDimension(h, index, (UIntPtr)(uint)d, out long dimension));
                            Result.VerifySuccess(NativeMethods.OgaComponentSessionGetInputSymbolicDimension(h, index, (UIntPtr)(uint)d, out IntPtr symbol));
                            shape.Add(dimension);
                            symbols.Add(symbol == IntPtr.Zero ? null : StringUtils.FromUtf8(symbol));
                        }
                        result.Add(StringUtils.FromUtf8(name), new ComponentInputInfo { Type = type, Shape = shape, SymbolicShape = symbols });
                    }
                    return result;
                });
            }
        }
        public unsafe IDictionary<string, ComponentTensor> Run(IDictionary<string, ComponentTensor> inputs, IEnumerable<string> outputNames)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));
            return SafeHandleAccess.Use(_handle, native =>
            {
                Result.VerifySuccess(NativeMethods.OgaCreateComponentInputs(out IntPtr nativeInputs));
                try
                {
                    foreach (KeyValuePair<string, ComponentTensor> pair in inputs)
                        fixed (byte* data = pair.Value.Data)
                            Result.VerifySuccess(NativeMethods.OgaComponentInputsAdd(nativeInputs, StringUtils.ToUtf8(pair.Key), data, (UIntPtr)(uint)pair.Value.Data.Length, pair.Value.Shape, (UIntPtr)(uint)pair.Value.Shape.Length, pair.Value.Type));
                    using (var names = new NativeStringArray(outputNames))
                    {
                        Result.VerifySuccess(NativeMethods.OgaComponentSessionRun(native, nativeInputs, names.Pointer, names.Count, out IntPtr tensors));
                        try
                        {
                            Result.VerifySuccess(NativeMethods.OgaComponentTensorsGetCount(tensors, out UIntPtr count));
                            var result = new Dictionary<string, ComponentTensor>();
                            for (int i = 0; i < NativeStructuredValue.CheckedCount(count); ++i)
                            {
                                UIntPtr index = (UIntPtr)(uint)i;
                                Result.VerifySuccess(NativeMethods.OgaComponentTensorsGetName(tensors, index, out IntPtr name));
                                Result.VerifySuccess(NativeMethods.OgaComponentTensorsGetType(tensors, index, out ElementType type));
                                Result.VerifySuccess(NativeMethods.OgaComponentTensorsGetShapeRank(tensors, index, out UIntPtr rank));
                                long[] shape = new long[NativeStructuredValue.CheckedCount(rank)];
                                for (int d = 0; d < shape.Length; ++d) Result.VerifySuccess(NativeMethods.OgaComponentTensorsGetShapeDimension(tensors, index, (UIntPtr)(uint)d, out shape[d]));
                                Result.VerifySuccess(NativeMethods.OgaComponentTensorsGetData(tensors, index, out IntPtr data, out UIntPtr bytes));
                                byte[] copied = new byte[NativeStructuredValue.CheckedCount(bytes)];
                                Marshal.Copy(data, copied, 0, copied.Length);
                                result.Add(StringUtils.FromUtf8(name), new ComponentTensor(copied, shape, type));
                            }
                            return result;
                        }
                        finally { NativeMethods.OgaDestroyComponentTensors(tensors); }
                    }
                }
                finally { NativeMethods.OgaDestroyComponentInputs(nativeInputs); }
            });
        }
        public void Dispose() { _handle.Dispose(); }
    }
}
