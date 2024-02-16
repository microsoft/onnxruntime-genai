#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <ort_genai_c.h>

// Our working directory is generators/build so one up puts us in the root directory:
#define MODEL_PATH "../../test/test_models/"

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
  OgaModel* model;
  CheckResult(OgaCreateModel(MODEL_PATH "../examples/phi2/model", OgaDeviceTypeCPU, &model));
  OgaModelPtr model_ptr{model};

  OgaTokenizer* tokenizer;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));
  OgaTokenizerPtr tokenizer_ptr{tokenizer};

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  OgaSequences* sequences;
  CheckResult(OgaTokenizerEncodeBatch(tokenizer, input_strings, std::size(input_strings), &sequences));
  OgaSequencesPtr sequences_ptr{sequences};

  // Decode Batch
  {
    const char* const* out_strings;
    CheckResult(OgaTokenizerDecodeBatch(tokenizer, sequences, &out_strings));
    for (size_t i = 0; i < OgaSequencesCount(sequences); i++) {
      std::cout << "Decoded string:" << out_strings[i] << std::endl;
      if (strcmp(input_strings[i], out_strings[i]) != 0)
        throw std::runtime_error("Batch Token decoding mismatch");
    }
    OgaTokenizerDestroyStrings(out_strings, OgaSequencesCount(sequences));
  }

  // Decode Single
  for (size_t i = 0; i < OgaSequencesCount(sequences); i++) {
    std::span<const int32_t> sequence{OgaSequencesGetSequenceData(sequences, i), OgaSequencesGetSequenceCount(sequences, i)};
    const char* out_string;
    CheckResult(OgaTokenizerDecode(tokenizer, sequence.data(), sequence.size(), &out_string));
    std::cout << "Decoded string:" << out_string << std::endl;
    if (strcmp(input_strings[i], out_string) != 0)
      throw std::runtime_error("Token decoding mismatch");
    OgaDestroyString(out_string);
  }

  // Stream Decode
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
  CheckResult(OgaGeneratorParamsSetMaxLength(params, max_length));
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
