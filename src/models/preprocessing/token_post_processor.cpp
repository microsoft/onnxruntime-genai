#include "token_post_processor.h"

#include "generator/token_outputs.h"
#include "models/model.h"

namespace Generators {

TokenPostProcessor::TokenPostProcessor(const Model& model, const Generator& generator)
    : generator_{generator}, generator_lifetime_{generator.postprocessor_lifetime_} {
  if (generator.model_.get() != &model)
    throw std::runtime_error("TokenPostProcessor model must match the generator model");
  const auto& search = generator.state_->params_->search;
  if (search.batch_size != 1 || search.num_beams != 1)
    throw std::runtime_error("TokenPostProcessor requires batch_size=1 and num_beams=1");
  ValidateTimestampConfiguration(model.config_->model);
  const auto& source = model.config_->model;
  config_.timestamps = {source.timestamp_level, source.segment_separators, source.segment_gap_threshold_seconds,
                        source.sample_rate, source.hop_length, source.subsampling_factor};
  tokenizer_ = model.CreateTokenizer();
  stream_ = tokenizer_->CreateStream();
  metadata_state_ = std::unique_ptr<MetadataCoreState>(new MetadataCoreState(config_));
}

void TokenPostProcessor::CheckGenerator() const {
  if (generator_lifetime_.expired()) throw std::runtime_error("TokenPostProcessor generator has been destroyed");
}

void TokenPostProcessor::AppendResult(const MetadataOutput& source) {
  PostProcessorResult result;
  result.text = source.text;
  result.has_timestamps = source.timestampMetadata != nullptr;
  const auto copy = [](const TimestampRecordView* records, size_t count,
                       std::vector<PostProcessorTimestamp>& output) {
    for (size_t index = 0; index < count; ++index) {
      const auto& record = records[index];
      output.push_back({record.text, record.start_frame, record.stop_frame, record.start_time, record.stop_time});
    }
  };
  if (source.timestampMetadata) {
    const auto& timestamps = *source.timestampMetadata;
    copy(timestamps.words, timestamps.word_count, result.words);
    copy(timestamps.segments, timestamps.segment_count, result.segments);
  }
  results_.push_back(std::move(result));
}

void TokenPostProcessor::ProcessNextTokens() {
  CheckGenerator();
  results_.clear();
  const auto outputs = generator_.GetTokenOutputs();
  for (size_t index = 0; index < outputs.tokens_.size(); ++index) {
    const bool has_interval = !outputs.intervals_.empty();
    const TokenMetadataInput token{outputs.tokens_[index], has_interval ? 1 : 0,
                                   has_interval ? outputs.intervals_[index] : OgaTokenMetadataAcousticFrameInterval{}};
    metadata_state_->ValidateInput(token);
    const auto piece = tokenizer_->GetWordPieceInfo(token.token_id);
    const auto& text = stream_->Decode(token.token_id);
    AppendResult(metadata_state_->ProcessDecoded(token, text.c_str(), piece));
  }
}

void TokenPostProcessor::Finalize() {
  CheckGenerator();
  results_.clear();
  AppendResult(metadata_state_->ProcessFinalized());
}

void TokenPostProcessor::Reset() {
  CheckGenerator();
  results_.clear();
  stream_->Reset();
  metadata_state_ = std::unique_ptr<MetadataCoreState>(new MetadataCoreState(config_));
}

}  // namespace Generators