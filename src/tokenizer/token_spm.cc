#include "token_spm.h"

#include <fstream>
#include <iostream>

using namespace tfm;

SpmTokenizer::SpmTokenizer() {}

SpmTokenizer::~SpmTokenizer() {}

TfmStatus SpmTokenizer::Onload() {
  sentencepiece::ModelProto model_proto;
  std::string model_file = GetDataDir() + "/tokenizer.model";

  // load the protobuf's object model_proto from model_file

  std::fstream model_stream(model_file.c_str(),
                            std::ios::in | std::ios::binary);
  if (!model_proto.ParseFromIstream(&model_stream)) {
    const char* err_msg = "Invalid model file";
    if (model_stream.bad()) {
      err_msg = "Failed to read model file";
    }
    
    return {kTfmErrorInvalidFile, std::string(err_msg) + ": " + model_file};
  }

  spm_processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = spm_processor_->Load(model_proto);
  if (!status.ok()) {
    spm_processor_.reset();
    return {kTfmErrorInvalidFile, "Invalid model file"};
  }

  return {};
}

TfmStatus SpmTokenizer::Encode(std::string_view input,
                               std::vector<tfmTokenId_t> &ids) const {
  std::vector<int> ids_int;
  auto status = spm_processor_->Encode(input.data(), &ids_int);
  if (!status.ok()) {
    return {kTfmErrorInvalidArgument, status.error_message()};
  }

  auto config = GetConfig();
  if (config->add_bos_token_) {
    ids_int.insert(ids_int.begin(), spm_processor_->bos_id());
  }

  if (config->add_eos_token_) {
    ids_int.push_back(spm_processor_->eos_id());
  }

  std::transform(ids_int.begin(), ids_int.end(), std::back_inserter(ids),
                 [](int i) { return static_cast<tfmTokenId_t>(i); });
  return TfmStatus::OK();
}

TfmStatus SpmTokenizer::Decode(const span<tfmTokenId_t const> &ids,
                               std::string &text) const {
  std::vector<int> ids_int(ids.data(), ids.data() + ids.size());
  auto status = spm_processor_->Decode(ids_int, &text);
  if (!status.ok()) {
    return {kTfmErrorInvalidArgument, "Invalid input ids"};
  }

  return TfmStatus::OK();
}
