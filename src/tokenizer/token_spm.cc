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
    if (model_stream.fail()) {
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
                               std::vector<tfmTokenId_t>& ids) const {
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

TfmStatus SpmTokenizer::Decode(const span<tfmTokenId_t const>& ids,
                               std::string& text) const {
  std::vector<int> ids_int(ids.data(), ids.data() + ids.size());
  auto status = spm_processor_->Decode(ids_int, &text);
  if (!status.ok()) {
    return {kTfmErrorInvalidArgument, "Invalid input ids"};
  }

  return TfmStatus::OK();
}

static void ReplaceAll(std::string& s, const std::string& search, const std::string& replace) {
  std::string result;
  for (size_t pos = 0;; pos += search.length()) {
    auto new_pos = s.find(search, pos);
    if (new_pos == std::string::npos) {
      result += s.substr(pos, s.size() - pos);
      break;
    }
    result += s.substr(pos, new_pos - pos) + replace;
    pos = new_pos;
  }
  s = std::move(result);
}

TfmStatus SpmTokenizer::Id2Token(tfmTokenId_t id, std::string& token, DecoderState** state) const {
  std::unique_ptr<SpmDeocerState> state_ptr;
  auto spm_state = static_cast<SpmDeocerState*>(*state);
  if (spm_state == nullptr) {
    state_ptr = std::make_unique<SpmDeocerState>();
    spm_state = state_ptr.get();
  }

  auto piece = spm_processor_->IdToPiece(id);
  if (piece.empty() || spm_processor_->IsControl(id)) {
    token = "";
    spm_state->last_control_char_ = true;
  } else if (spm_processor_->IsByte(id)) {
    auto buf = piece.substr(3, 2);  // something like <0x20>
    token = {static_cast<char>(strtol(buf.c_str(), NULL, 16))};
  } else {
    token = piece;
    ReplaceAll(token, "\xe2\x96\x81", " ");
  }

  if (!token.empty() && token[0] == ' ' && spm_state->last_control_char_) {
    token = token.substr(1);
    spm_state->last_control_char_ = false;
  }

  if (*state == nullptr) {
    *state = state_ptr.release();
  }
  return {};
}
