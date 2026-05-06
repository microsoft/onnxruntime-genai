// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "json.h"

#include <gtest/gtest.h>

namespace Generators::test {

namespace {

// Round-trip helper: parse text into a Document and serialize it back.
std::string RoundTrip(std::string_view text) {
  return JSON::SerializeDocument(JSON::ParseDocument(text));
}

// Apply RFC 7386 merge patch via the public API and serialize the result.
std::string MergeAndSerialize(std::string_view target, std::string_view patch) {
  JSON::Document t = JSON::ParseDocument(target);
  JSON::Document p = JSON::ParseDocument(patch);
  JSON::MergePatch(t, p);
  return JSON::SerializeDocument(t);
}

}  // namespace

// ---------------------------------------------------------------------------
// DOM parse / serialize round-trip basics.
// ---------------------------------------------------------------------------

TEST(JsonDomTest, RoundTripScalars) {
  EXPECT_EQ(RoundTrip("null"), "null");
  EXPECT_EQ(RoundTrip("true"), "true");
  EXPECT_EQ(RoundTrip("false"), "false");
  EXPECT_EQ(RoundTrip("42"), "42");
  EXPECT_EQ(RoundTrip("-7"), "-7");
  EXPECT_EQ(RoundTrip("\"hello\""), "\"hello\"");
}

TEST(JsonDomTest, RoundTripObjectAndArray) {
  // Object keys are serialized in sorted order (std::map). Both inputs below
  // therefore round-trip to the same canonical string.
  EXPECT_EQ(RoundTrip("{\"a\":1,\"b\":\"x\"}"), "{\"a\":1,\"b\":\"x\"}");
  EXPECT_EQ(RoundTrip("{\"b\":\"x\",\"a\":1}"), "{\"a\":1,\"b\":\"x\"}");
  EXPECT_EQ(RoundTrip("[1,2,3]"), "[1,2,3]");
  EXPECT_EQ(RoundTrip("{\"a\":[1,{\"b\":2}]}"), "{\"a\":[1,{\"b\":2}]}");
}

TEST(JsonDomTest, RoundTripEmpty) {
  EXPECT_EQ(RoundTrip("{}"), "{}");
  EXPECT_EQ(RoundTrip("[]"), "[]");
  EXPECT_EQ(RoundTrip("{\"a\":[],\"b\":{}}"), "{\"a\":[],\"b\":{}}");
}

TEST(JsonDomTest, IntegralDoublesEmittedWithoutDecimal) {
  // Integral-valued doubles round-trip without a decimal point so that token
  // ids / hidden_size etc. round-trip byte-stable through a no-op merge.
  JSON::Document d(static_cast<double>(199999));
  EXPECT_EQ(JSON::SerializeDocument(d), "199999");
}

// ---------------------------------------------------------------------------
// RFC 7386 conformance examples.
// https://datatracker.ietf.org/doc/html/rfc7386 §3
// ---------------------------------------------------------------------------

TEST(JsonMergePatchTest, RfcExample_ReplaceScalar) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"b"})", R"({"a":"c"})"),
            R"({"a":"c"})");
}

TEST(JsonMergePatchTest, RfcExample_AddKey) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"b"})", R"({"b":"c"})"),
            R"({"a":"b","b":"c"})");
}

TEST(JsonMergePatchTest, RfcExample_DeleteKey) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"b"})", R"({"a":null})"),
            R"({})");
}

TEST(JsonMergePatchTest, RfcExample_DeleteOneOfMany) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"b","b":"c"})", R"({"a":null})"),
            R"({"b":"c"})");
}

TEST(JsonMergePatchTest, RfcExample_ReplaceArrayWithScalar) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":["b"]})", R"({"a":"c"})"),
            R"({"a":"c"})");
}

TEST(JsonMergePatchTest, RfcExample_ReplaceScalarWithArray) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"c"})", R"({"a":["b"]})"),
            R"({"a":["b"]})");
}

TEST(JsonMergePatchTest, RfcExample_RecurseObjectAndDelete) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":{"b":"c"}})",
                              R"({"a":{"b":"d","c":null}})"),
            R"({"a":{"b":"d"}})");
}

TEST(JsonMergePatchTest, RfcExample_ArrayOfObjectReplacesWholesale) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":[{"b":"c"}]})", R"({"a":[1]})"),
            R"({"a":[1]})");
}

TEST(JsonMergePatchTest, RfcExample_ArrayPatchReplacesArray) {
  EXPECT_EQ(MergeAndSerialize(R"(["a","b"])", R"(["c","d"])"),
            R"(["c","d"])");
}

TEST(JsonMergePatchTest, RfcExample_ArrayPatchReplacesObject) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"b"})", R"(["c"])"),
            R"(["c"])");
}

TEST(JsonMergePatchTest, RfcExample_NullPatchReplacesObject) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"foo"})", R"(null)"),
            R"(null)");
}

TEST(JsonMergePatchTest, RfcExample_StringPatchReplacesObject) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"foo"})", R"("bar")"),
            R"("bar")");
}

TEST(JsonMergePatchTest, RfcExample_NullInTargetSurvivesAddition) {
  // Pre-existing null in the target is not removed by an unrelated patch.
  EXPECT_EQ(MergeAndSerialize(R"({"e":null})", R"({"a":1})"),
            R"({"a":1,"e":null})");
}

TEST(JsonMergePatchTest, RfcExample_MergeIntoArrayResetsToObject) {
  // RFC: when the target is not an object and the patch is an object, the
  // target is treated as an empty object.
  EXPECT_EQ(MergeAndSerialize(R"([1,2])", R"({"a":"b","c":null})"),
            R"({"a":"b"})");
}

TEST(JsonMergePatchTest, RfcExample_DeepNestedAdd) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":{"b":{"c":1}}})",
                              R"({"a":{"b":{"d":2}}})"),
            R"({"a":{"b":{"c":1,"d":2}}})");
}

// ---------------------------------------------------------------------------
// GenAI-relevant scenarios — sanity-check the shapes the package overlay
// design will rely on (per Appendix A of the v4 design).
// ---------------------------------------------------------------------------

TEST(JsonMergePatchTest, GenAi_OverrideContextLengthAndIONames) {
  constexpr const char* base = R"({
    "model": {
      "type": "phi3",
      "context_length": 131072,
      "decoder": {
        "inputs": {"position_ids": "position_ids"}
      }
    }
  })";
  constexpr const char* overlay = R"({
    "model": {
      "context_length": 4096,
      "decoder": {
        "inputs": {"position_ids": "pos_ids_qnn"}
      }
    }
  })";
  JSON::Document t = JSON::ParseDocument(base);
  JSON::Document p = JSON::ParseDocument(overlay);
  JSON::MergePatch(t, p);
  EXPECT_EQ(t.AsObject().at("model").AsObject().at("context_length").AsNumber(),
            4096);
  EXPECT_EQ(t.AsObject()
                .at("model")
                .AsObject()
                .at("decoder")
                .AsObject()
                .at("inputs")
                .AsObject()
                .at("position_ids")
                .AsString(),
            "pos_ids_qnn");
  // type stayed put.
  EXPECT_EQ(t.AsObject().at("model").AsObject().at("type").AsString(), "phi3");
}

TEST(JsonMergePatchTest, GenAi_PipelineArrayReplacedWholesale) {
  // Pipeline arrays are replaced wholesale (RFC 7386). This is the documented
  // contract for v4 multi-file QNN-style variants.
  constexpr const char* base = R"({
    "model": {"decoder": {"pipeline": [{"old_stage": {"filename": "old.onnx"}}]}}
  })";
  constexpr const char* overlay = R"({
    "model": {"decoder": {"pipeline": [
      {"embedding": {"filename": "emb.onnx"}},
      {"prompt": {"filename": "ctx.onnx"}}
    ]}}
  })";
  JSON::Document t = JSON::ParseDocument(base);
  JSON::Document p = JSON::ParseDocument(overlay);
  JSON::MergePatch(t, p);
  const auto& pipeline = t.AsObject()
                             .at("model")
                             .AsObject()
                             .at("decoder")
                             .AsObject()
                             .at("pipeline")
                             .AsArray();
  ASSERT_EQ(pipeline.size(), 2u);
  EXPECT_TRUE(pipeline[0].AsObject().count("embedding"));
  EXPECT_TRUE(pipeline[1].AsObject().count("prompt"));
}

TEST(JsonMergePatchTest, GenAi_NullDeletesOptionalField) {
  // Producers should be able to suppress a base field by setting it to null
  // in the overlay (e.g. drop a default eos_token_id list).
  constexpr const char* base = R"({"model": {"eos_token_id": [1, 2]}})";
  constexpr const char* overlay = R"({"model": {"eos_token_id": null}})";
  EXPECT_EQ(MergeAndSerialize(base, overlay), R"({"model":{}})");
}

// ---------------------------------------------------------------------------
// Edge cases beyond the RFC examples.
// ---------------------------------------------------------------------------

TEST(JsonMergePatchTest, EmptyPatchObjectIsNoOp) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":1})", R"({})"), R"({"a":1})");
}

TEST(JsonMergePatchTest, EmptyPatchObjectOntoScalarBecomesEmptyObject) {
  // Per RFC: when target is non-object and patch is object, target is treated
  // as an empty object before applying the patch.
  EXPECT_EQ(MergeAndSerialize(R"(42)", R"({})"), R"({})");
}

TEST(JsonMergePatchTest, DeleteOfMissingKeyIsNoOp) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":1})", R"({"missing":null})"),
            R"({"a":1})");
}

TEST(JsonMergePatchTest, ObjectOverlayOntoScalarLeafForcesObjectSemantics) {
  EXPECT_EQ(MergeAndSerialize(R"({"a":"old"})", R"({"a":{"b":1}})"),
            R"({"a":{"b":1}})");
}

TEST(JsonDomTest, TrailingWhitespaceParses) {
  EXPECT_EQ(JSON::SerializeDocument(JSON::ParseDocument("  {\"a\":1}  ")),
            R"({"a":1})");
}

TEST(JsonDomTest, BareLiteralAtEndOfInputParses) {
  // Regression: the streaming parser used to silently drop a top-level
  // true / false / null literal when it appeared at the very end of the
  // input buffer (Skip() bounds check was off-by-one).
  EXPECT_EQ(JSON::SerializeDocument(JSON::ParseDocument("true")), "true");
  EXPECT_EQ(JSON::SerializeDocument(JSON::ParseDocument("false")), "false");
  EXPECT_EQ(JSON::SerializeDocument(JSON::ParseDocument("null")), "null");
}

}  // namespace Generators::test
