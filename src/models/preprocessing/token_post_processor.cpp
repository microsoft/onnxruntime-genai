#include "token_post_processor.h"

#include "generator/token_outputs.h"
#include "models/model.h"

namespace Generators {

TokenPostProcessor::TokenPostProcessor(const Generator& generator)
    : generator_{generator},
      config_{{generator.model_->config_->model.timestamp_level, generator.model_->config_->model.segment_separators,
               generator.model_->config_->model.segment_gap_threshold_seconds, generator.model_->config_->model.sample_rate,
               generator.model_->config_->model.hop_length, generator.model_->config_->model.subsampling_factor}} {
  const auto& model = *generator.model_;
  const auto& search = generator.state_->params_->search;
  if (search.batch_size != 1 || search.num_beams != 1)
    throw std::runtime_error("TokenPostProcessor requires batch_size=1 and num_beams=1");
  ValidateTimestampConfiguration(model.config_->model);
  tokenizer_ = model.CreateTokenizer();
  stream_ = tokenizer_->CreateStream();
  metadata_state_ = std::unique_ptr<MetadataCoreState>(new MetadataCoreState(config_));
}

void TokenPostProcessor::ProcessNextTokens() {
  results_.clear();
  const auto outputs = generator_.GetTokenOutputs();
  const auto tokens = outputs.GetTokens();
  const auto* timestamps = outputs.GetTimestamps();
  for (size_t index = 0; index < tokens.size(); ++index) {
    TokenMetadataInput token{tokens[index], 0, {}};
    if (timestamps) {
      token.has_token_acoustic_frame_interval = 1;
      token.token_acoustic_frame_interval = (*timestamps)[index];
    }
    metadata_state_->ValidateInput(token);
    OrtxWordPieceInfo piece{};
    if (metadata_state_->TimestampsEnabled()) piece = tokenizer_->GetWordPieceInfo(token.token_id);
    const auto& text = stream_->Decode(token.token_id);
    auto& result = results_.emplace_back();
    result.text = text;
    metadata_state_->ProcessDecoded(token, piece, result);
  }
}

void TokenPostProcessor::Finalize() {
  results_.clear();
  metadata_state_->ProcessFinalized(results_.emplace_back());
}

void TokenPostProcessor::Reset() {
  results_.clear();
  stream_->Reset();
  metadata_state_ = std::unique_ptr<MetadataCoreState>(new MetadataCoreState(config_));
}

}  // namespace Generators