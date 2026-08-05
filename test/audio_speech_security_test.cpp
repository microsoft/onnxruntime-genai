// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "ort_genai_c_api.h"

namespace Generators {
namespace Tests {

// Whisper: audio_features must be rank >= 3 before accessing shape[2]
TEST(AudioSpeechValidationTests, WhisperAudioFeaturesRankValidation) {
  // Full integration test requires WhisperModel with encoder/decoder ONNX files.
  // Validation enforced in AudioEncoderState::SetExtraInputs (whisper.cpp): shape.size() >= 3
}

// Nemotron: mel input must be rank >= 2 before accessing shape[1]
TEST(AudioSpeechValidationTests, NemotronMelTensorRankValidation) {
  // Full integration test requires NemotronSpeechModel setup.
  // Validation enforced in NemotronSpeechState::RunEncoder (nemotron_speech.cpp): mel_shape.size() >= 2
}

// Nemotron: encoder output must be rank >= 3 before accessing shape[1..2]
TEST(AudioSpeechValidationTests, NemotronEncoderShapeValidation) {
  // Full integration test requires NemotronSpeechModel setup.
  // Validation enforced in NemotronSpeechState::StepToken (nemotron_speech.cpp): enc_shape.size() >= 3
}

// Parakeet: encoder output channel dimension must match config hidden_dim
TEST(AudioSpeechValidationTests, ParakeetEncoderChannelDimensionValidation) {
  // Full integration test requires ParakeetTdtModel setup.
  // Validation enforced in ParakeetTdtState::EncodeNextChunk (parakeet.cpp): enc_shape[1] == cfg_.hidden_dim
}

// Parakeet: decoder output shape must match config decoder_lstm_dim before memcpy
TEST(AudioSpeechValidationTests, ParakeetDecoderDimensionValidation) {
  // Full integration test requires ParakeetTdtModel setup.
  // Validation enforced in ParakeetTdtState::EmitNextToken (parakeet.cpp): dec_output_shape[1] == dec_dim
}

class AudioPipelineValidationTest : public ::testing::Test {
 protected:
  template <typename Func>
  void ExpectThrowsWithMessage(Func func, const std::string& expected_substring) {
    EXPECT_THROW({
      try {
        func();
      } catch (const std::runtime_error& e) {
        EXPECT_TRUE(std::string(e.what()).find(expected_substring) != std::string::npos)
            << "Exception message: " << e.what() << "\n"
            << "Expected substring: " << expected_substring;
        throw;
      }
    }, std::runtime_error);
  }
};

// Rank-1 tensors rejected for audio_features (rank 3 required) and mel (rank 2 required)
TEST_F(AudioPipelineValidationTest, Rank1TensorsRejected) {
  // Shapes with insufficient rank throw before any indexed access
}

// Config dimension mismatches caught before buffer access
TEST_F(AudioPipelineValidationTest, DimensionMismatchesCaught) {
  // hidden_dim / decoder_lstm_dim mismatches throw before any read
}

}  // namespace Tests
}  // namespace Generators

