#include "asr_post_processor.h"

#include "generator/token_outputs.h"
#include "models/model.h"

namespace Generators {

namespace {
MetadataCoreConfig GetPostProcessorConfig(const Generator& generator) {
  const auto& model = *generator.model_;
  const auto& search = generator.state_->params_->search;
  if (search.batch_size != 1 || search.num_beams != 1)
    throw std::runtime_error("ASRPostProcessor requires batch_size=1 and num_beams=1");
  ValidateTimestampConfiguration(model.config_->model);
  const auto& source = model.config_->model;
  MetadataCoreConfig config;
  config.timestamps.level = source.timestamp_level;
  config.timestamps.segment_separators = source.segment_separators;
  if (source.timestamp_level != Config::TimestampLevel::Off) {
    config.seconds_per_frame = static_cast<double>(source.hop_length) * source.subsampling_factor / source.sample_rate;
    config.timestamps.segment_gap_threshold_frames = GetSegmentGapThresholdFrames(source);
  }
  return config;
}
}  // namespace

ASRPostProcessor::ASRPostProcessor(const Generator& generator)
    : PostProcessor{generator}, internal_{generator_.model_->CreateTokenizer(), GetPostProcessorConfig(generator_)} {}

void ASRPostProcessor::ProcessNext() {
  internal_.Process(generator_.GetTokenOutputs());
}

void ASRPostProcessor::Finalize() {
  internal_.Finalize();
}

void ASRPostProcessor::Reset() {
  internal_.Reset();
}

ASRPostProcessorInternal::ASRPostProcessorInternal(std::shared_ptr<Tokenizer> tokenizer, MetadataCoreConfig config)
    : tokenizer_{std::move(tokenizer)}, config_{std::move(config)} {
  stream_ = tokenizer_->CreateStream();
  metadata_state_ = std::unique_ptr<MetadataCoreState>(new MetadataCoreState(config_));
}

void ASRPostProcessorInternal::Process(const TokenOutputs& outputs) {
  results_.clear();
  const auto tokens = outputs.GetTokens();
  const auto* timestamps = outputs.GetAcousticFrameIntervals();
  if (timestamps && timestamps->size() != tokens.size())
    throw std::runtime_error("Token timing count does not match emitted tokens");
  for (size_t index = 0; index < tokens.size(); ++index) {
    TokenMetadataInput token{tokens[index], std::nullopt};
    if (timestamps) {
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

void ASRPostProcessorInternal::Finalize() {
  results_.clear();
  metadata_state_->ProcessFinalized(results_.emplace_back());
}

void ASRPostProcessorInternal::Reset() {
  results_.clear();
  stream_->Reset();
  metadata_state_ = std::unique_ptr<MetadataCoreState>(new MetadataCoreState(config_));
}

}  // namespace Generators