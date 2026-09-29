// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    public class NonGenerativeTests
    {
        [Fact]
        public void RequestObjectsValidateRequiredFields()
        {
            Assert.Throws<ArgumentException>(() => new StructuredQuestion("", "prompt"));
            Assert.Throws<ArgumentNullException>(() => new StructuredRequest(null, null));
            Assert.Throws<ArgumentNullException>(() => new FreeFormRankRequest(null, null, null));
            Assert.Throws<ArgumentOutOfRangeException>(() => new StructuredRequest(
                ulong.MaxValue, new Dictionary<string, StructuredQuestion>()));

            var cycle = new Hashtable();
            cycle["self"] = cycle;
            Assert.Throws<ArgumentException>(() => new StructuredRequest(
                cycle, new Dictionary<string, StructuredQuestion>()));

            IList root = new ArrayList();
            IList current = root;
            for (int i = 0; i < 128; ++i)
            {
                var child = new ArrayList();
                current.Add(child);
                current = child;
            }
            Assert.Throws<ArgumentException>(() => new StructuredRequest(
                root, new Dictionary<string, StructuredQuestion>()));
        }

        [Fact]
        public void ComponentTensorRetainsStructuredMetadata()
        {
            var tensor = new ComponentTensor(new byte[] { 1, 2, 3, 4 }, new long[] { 1 }, ElementType.float32);
            Assert.Equal(ElementType.float32, tensor.Type);
            Assert.Equal(new long[] { 1 }, tensor.Shape);
            Assert.Equal(4, tensor.Data.Length);
        }

        [Fact]
        public void OptInClmPackageCoversLifecycleConversionAndCache()
        {
            string path = Environment.GetEnvironmentVariable("ORTGENAI_TEST_CLM_PACKAGE");
            if (string.IsNullOrEmpty(path)) return;

            var questions = new Dictionary<string, StructuredQuestion>
            {
                ["q"] = new StructuredQuestion("noul", "Is this suitable?")
            };
            using (var session = new RankingSession(path, new[] { "cpu" }))
            {
                session.SetCacheCapacity(2, 1024 * 1024);
                ModelResult result = session.Run(new StructuredRequest(
                    new Dictionary<string, object> { ["state"] = "rain" }, questions));
                Assert.NotNull(result.Model);
                Assert.Single(result.Answers);
                RankingResult ranked = session.Rank(new FreeFormRankRequest(
                    new Dictionary<string, object> { ["state"] = "rain" },
                    "Choose the best activity",
                    new Dictionary<string, object>
                    {
                        ["inside"] = new List<object> { "museum", true },
                        ["outside"] = "picnic"
                    }));
                Assert.Equal(2, ranked.Items.Count);
                Assert.Equal((ulong)2, session.CacheStats.EntryCapacity);
                session.ClearCache();
                Assert.Equal((ulong)0, session.CacheStats.Entries);
                session.InvalidateCache();
            }

            var concurrent = new RankingSession(path, new[] { "cpu" });
            var start = new ManualResetEventSlim(false);
            Task operation = Task.Run(() =>
            {
                start.Wait();
                try { concurrent.Run(new StructuredRequest("rain", questions)); }
                catch (ObjectDisposedException) { }
            });
            Task close = Task.Run(() => { start.Wait(); concurrent.Dispose(); });
            start.Set();
            Task.WaitAll(operation, close);
            concurrent.Dispose();

            string component = Environment.GetEnvironmentVariable("ORTGENAI_TEST_COMPONENT");
            if (!string.IsNullOrEmpty(component))
            {
                var componentSession = new ComponentSession(path, component, new[] { "cpu" });
                start.Reset();
                Task metadata = Task.Run(() =>
                {
                    start.Wait();
                    try
                    {
                        Assert.NotEmpty(componentSession.InputNames);
                        Assert.NotNull(componentSession.InputInfo);
                    }
                    catch (ObjectDisposedException) { }
                });
                Task componentClose = Task.Run(() => { start.Wait(); componentSession.Dispose(); });
                start.Set();
                Task.WaitAll(metadata, componentClose);
                componentSession.Dispose();
            }
        }

        [Fact]
        public void OptInKevPackageCoversDecisionConversion()
        {
            string path = Environment.GetEnvironmentVariable("ORTGENAI_TEST_KEV_PACKAGE");
            if (string.IsNullOrEmpty(path)) return;

            using (var session = new DecisionSession(path))
            {
                Assert.True(session.PrefixReuseEnabled);
                Assert.False(string.IsNullOrEmpty(session.PrefixReuseStatus));
                session.PrefixReuseEnabled = false;
                Assert.False(session.PrefixReuseEnabled);
                session.PrefixReuseEnabled = true;
                session.SetPrefixCacheCapacity(1, 1024 * 1024);
                Assert.Equal((ulong)1, session.PrefixCacheStats.EntryCapacity);

                ModelResult result = session.Decide(new StructuredRequest(
                    new Dictionary<string, object> { ["weather"] = "rain" },
                    new Dictionary<string, StructuredQuestion>
                    {
                        ["q"] = new StructuredQuestion("noul", "Take an umbrella?")
                    }));
                Assert.NotEmpty(result.Answers);
                Assert.True(session.PrefixReuseStats.PrefixRuns +
                            session.PrefixReuseStats.BranchRuns +
                            session.PrefixReuseStats.FallbackRuns > 0);
                session.ClearCache();
                Assert.Equal((ulong)0, session.PrefixCacheStats.Entries);

                var mutated = new Hashtable();
                var request = new StructuredRequest(
                    mutated,
                    new Dictionary<string, StructuredQuestion>
                    {
                        ["q"] = new StructuredQuestion("noul", "Take an umbrella?")
                    });
                mutated["self"] = mutated;
                Assert.Throws<ArgumentException>(() => session.Decide(request));

                var mutableState = new Dictionary<string, object>();
                var unsignedRequest = new StructuredRequest(
                    mutableState,
                    new Dictionary<string, StructuredQuestion>
                    {
                        ["q"] = new StructuredQuestion("noul", "Take an umbrella?")
                    });
                mutableState["overflow"] = ulong.MaxValue;
                Assert.Throws<ArgumentOutOfRangeException>(() => session.Decide(unsignedRequest));
            }
        }
    }
}
