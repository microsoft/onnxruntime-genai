#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <ort_genai_c.h>
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif
struct Deleters {
  void operator()(OgaResult* p) {
    OgaDestroyResult(p);
  }
  void operator()(OgaBuffer* p) {
    OgaDestroyBuffer(p);
  }
  void operator()(OgaSequences* p) {
    OgaDestroySequences(p);
  }
  void operator()(OgaModel* p) {
    OgaDestroyModel(p);
  }
  void operator()(OgaTokenizer* p) {
    OgaDestroyTokenizer(p);
  }
  void operator()(OgaTokenizerStream* p) {
    OgaDestroyTokenizerStream(p);
  }
  void operator()(OgaGeneratorParams* p) {
    OgaDestroyGeneratorParams(p);
  }
  void operator()(OgaGenerator* p) {
    OgaDestroyGenerator(p);
  }
};

using OgaResultPtr = std::unique_ptr<OgaResult, Deleters>;
using OgaBufferPtr = std::unique_ptr<OgaBuffer, Deleters>;
using OgaSequencesPtr = std::unique_ptr<OgaSequences, Deleters>;
using OgaModelPtr = std::unique_ptr<OgaModel, Deleters>;
using OgaTokenizerPtr = std::unique_ptr<OgaTokenizer, Deleters>;
using OgaTokenizerStreamPtr = std::unique_ptr<OgaTokenizerStream, Deleters>;
using OgaGeneratorParamsPtr = std::unique_ptr<OgaGeneratorParams, Deleters>;
using OgaGeneratorPtr = std::unique_ptr<OgaGenerator, Deleters>;

void CheckResult(OgaResult* result) {
  if (!result)
    return;

  OgaResultPtr result_ptr{result};
  throw std::runtime_error(OgaResultGetError(result));
}

TEST(CAPITests, TokenizerCAPI) {
#if TEST_PHI2
  OgaModel* model;
  CheckResult(OgaCreateModel(MODEL_PATH "phi-2", OgaDeviceTypeCPU, &model));
  OgaModelPtr model_ptr{model};

  OgaTokenizer* tokenizer;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));
  OgaTokenizerPtr tokenizer_ptr{tokenizer};

  // Encode single decode single
  {
    const char* input_string = "She sells sea shells by the sea shore.";
    OgaSequences* input_sequences;
    CheckResult(OgaCreateSequences(&input_sequences));
    CheckResult(OgaTokenizerEncode(tokenizer, input_string, input_sequences));
    OgaSequencesPtr input_sequences_ptr{input_sequences};

    std::span<const int32_t> sequence{OgaSequencesGetSequenceData(input_sequences, 0), OgaSequencesGetSequenceCount(input_sequences, 0)};
    const char* out_string;
    CheckResult(OgaTokenizerDecode(tokenizer, sequence.data(), sequence.size(), &out_string));
    ASSERT_STREQ(input_string, out_string);
  }

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  OgaSequences* sequences;
  CheckResult(OgaCreateSequences(&sequences));
  OgaSequencesPtr sequences_ptr{sequences};

  // Encode all strings
  {
    for (auto &string : input_strings)
      CheckResult(OgaTokenizerEncode(tokenizer, string, sequences));
  }

  // Decode one at a time
  for (size_t i = 0; i < OgaSequencesCount(sequences); i++) {
    std::span<const int32_t> sequence{OgaSequencesGetSequenceData(sequences, i), OgaSequencesGetSequenceCount(sequences, i)};
    const char* out_string;
    CheckResult(OgaTokenizerDecode(tokenizer, sequence.data(), sequence.size(), &out_string));
    std::cout << "Decoded string:" << out_string << std::endl;
    if (strcmp(input_strings[i], out_string) != 0)
      throw std::runtime_error("Token decoding mismatch");
    OgaDestroyString(out_string);
  }

  // Stream Decode one at a time
  for (size_t i = 0; i < OgaSequencesCount(sequences); i++) {
    OgaTokenizerStream* tokenizer_stream;
    CheckResult(OgaCreateTokenizerStream(tokenizer, &tokenizer_stream));
    OgaTokenizerStreamPtr tokenizer_stream_ptr{tokenizer_stream};

    std::span<const int32_t> sequence{OgaSequencesGetSequenceData(sequences, i), OgaSequencesGetSequenceCount(sequences, i)};
    std::string stream_result;
    for (auto& token : sequence) {
      const char* chunk;
      CheckResult(OgaTokenizerStreamDecode(tokenizer_stream, token, &chunk));
      stream_result += std::string(chunk);
    }
    std::cout << "Stream decoded string:" << stream_result << std::endl;
    if (strcmp(input_strings[i], stream_result.c_str()) != 0)
      throw std::runtime_error("Stream token decoding mismatch");
  }
#endif
}

TEST(CAPITests, EndToEndPhiBatch) {
#if TEST_PHI2
  OgaModel* model;
  CheckResult(OgaCreateModel(MODEL_PATH "phi-2", OgaDeviceTypeCPU, &model));
  OgaModelPtr model_ptr{model};

  OgaTokenizer* tokenizer;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));
  OgaTokenizerPtr tokenizer_ptr{tokenizer};

  OgaSequences* input_sequences;
  CheckResult(OgaCreateSequences(&input_sequences));
  OgaSequencesPtr sequences_ptr{input_sequences};

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  for (auto& string : input_strings)
    CheckResult(OgaTokenizerEncode(tokenizer, string, input_sequences));

  OgaGeneratorParams* params;
  CheckResult(OgaCreateGeneratorParams(model, &params));
  OgaGeneratorParamsPtr params_ptr{params};
  CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 20));
  CheckResult(OgaGeneratorParamsSetInputSequences(params, input_sequences));

  OgaSequences* output_sequences;
  CheckResult(OgaGenerate(model, params, &output_sequences));
  OgaSequencesPtr output_sequences_ptr{output_sequences};

  // Decode The Batch
  for (size_t i = 0; i < OgaSequencesCount(output_sequences); i++) {
    std::span<const int32_t> sequence{OgaSequencesGetSequenceData(output_sequences, i), OgaSequencesGetSequenceCount(output_sequences, i)};

    const char* out_string;
    CheckResult(OgaTokenizerDecode(tokenizer, sequence.data(), sequence.size(), &out_string));
    std::cout << "Decoded string:" << out_string << std::endl;
    OgaDestroyString(out_string);
  }
#endif
}

TEST(CAPITests, GreedySearchGptFp32CAPI) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto sequence_length = input_ids_shape[1];
  auto batch_size = input_ids_shape[0];
  int max_length = 10;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  OgaModel* model;
  CheckResult(OgaCreateModel(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32", OgaDeviceTypeCPU, &model));
  OgaModelPtr model_ptr{model};

  OgaGeneratorParams* params;
  CheckResult(OgaCreateGeneratorParams(model, &params));
  OgaGeneratorParamsPtr params_ptr{params};
  CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", max_length));
  CheckResult(OgaGeneratorParamsSetInputIDs(params, input_ids.data(), input_ids.size(), sequence_length, batch_size));

  OgaGenerator* generator;
  CheckResult(OgaCreateGenerator(model, params, &generator));
  OgaGeneratorPtr generator_ptr{generator};

  while (!OgaGenerator_IsDone(generator)) {
    CheckResult(OgaGenerator_ComputeLogits(generator));
    CheckResult(OgaGenerator_GenerateNextToken_Top(generator));
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    size_t token_count = OgaGenerator_GetSequenceLength(generator, i);
    const int32_t* data = OgaGenerator_GetSequence(generator, i);
    std::vector<int32_t> sequence(data, data + token_count);

    auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
  }

  // Test high level API
  OgaSequences* sequences;
  CheckResult(OgaGenerate(model, params, &sequences));
  OgaSequencesPtr sequences_ptr{sequences};

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    std::span<const int32_t> sequence{OgaSequencesGetSequenceData(sequences, i), OgaSequencesGetSequenceCount(sequences, i)};

    auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
  }
}
