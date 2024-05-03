#include "token_bpe.h"
#include "token_rwkv.h"

#include "../filesystem.h"
#include <memory>

namespace tfm {

static std::string_view GetModelName(const std::string_view& tok_cls) {
  constexpr std::string_view tok = "Tokenizer";
  if (tok_cls.size() > tok.length()) {
    if (tok_cls.substr(tok_cls.size() - tok.length()) == tok) {
      return tok_cls.substr(0, tok_cls.size() - tok.length());
    }
  }

  return tok_cls;
}

Tokenizer::Tokenizer() : TfmObjectImpl(tfmObjectKind_t::kTfmKindTokenizer) {}

TfmStatus CreateBPETokenizer(const std::string& tokenizer_path,
                             const std::string& tokenizer_type,
                             std::unique_ptr<Tokenizer>& token_ptr) {
  auto token_cfg = std::make_unique<TokenConfig>();
  auto config_file = tokenizer_path + "/tokenizer_config.json";
  auto status = token_cfg->LoadJson(config_file);

  std::string type = tokenizer_type;
  if (type.empty()) {
    if (BPETokenizer::IsSupportedModel(GetModelName(token_cfg->tokenizer_class_))) {
      type = "BPE";
    } /* else if (fs::exists(tokenizer_path + "/tokenizer.model")) {
      // if 'tokenizer.model exists in the tokenizer_path, then it is a sentencepiece model
      type = "SPM";
    } */ else {
      status = TfmStatus(kTfmErrorInvalidArgument, "Cannot determine the tokenizer type from tokenizer_path argument");
    }
  }

  if (type == "BPE") {
    token_ptr = std::make_unique<BPETokenizer>();
  } /* else if (type == "SPM") {
    token_ptr = std::make_unique<SpmTokenizer>();
  } */ else {
    status = TfmStatus(kTfmErrorInvalidArgument, "Unknown tokenizer_type, (BPE, RKWV) are supported.");
  }

  if (status.ok()) {
    status = token_ptr->LoadData(std::move(token_cfg), tokenizer_path);
    if (!status.ok()) {
      token_ptr.reset();
    }
  }

  return status;
}

std::unique_ptr<Tokenizer> CreateTokenizer(const std::string& tokenizer_path,
                                           const std::string& tokenizer_type,
                                           TfmStatus* status_return) {
  TfmStatus status;
  std::unique_ptr<Tokenizer> token_ptr;
  if (tokenizer_type == "RWKV") {
    token_ptr = std::make_unique<RwkvTokenizer>();
    status = token_ptr->LoadData({}, tokenizer_path);
  } else {
    status = CreateBPETokenizer(tokenizer_path, tokenizer_type, token_ptr);
  }

  if (status_return != nullptr) {
    *status_return = status;
  }

  return token_ptr;
}

TfmStatus Tokenizer::LoadData(std::unique_ptr<TokenConfig> token_cfg,
                              const std::string& tokenizer_dir) {
  TokenizerImpl& impl = *static_cast<TokenizerImpl*>(this);
  if (token_cfg != nullptr) {
    impl.BindConfig(std::move(token_cfg));
  }

  impl.SetDataDir(tokenizer_dir);
  return impl.OnLoad();
}

TfmStatus Tokenizer::Tokenize(const std::vector<std::string_view>& input,
                              std::vector<std::vector<tfmTokenId_t>>& t_ids) const {
  return BatchEncode(input, t_ids);
}

TfmStatus Tokenizer::Detokenize(const std::vector<span<tfmTokenId_t const>>& t_ids,
                                std::vector<std::string>& t_text) const {
  return BatchDecode(t_ids, t_text);
}

TfmStatus Tokenizer::Id2Token(tfmTokenId_t id, std::string& token, std::unique_ptr<DecoderState>& cache) const {
  DecoderState* state_ptr = cache.get();
  TfmStatus status = Id2Token(id, token, &state_ptr);
  if (status.ok()) {
    if (state_ptr != cache.get()) {
      cache.reset(state_ptr);
    }
  }

  return status;
}

TfmStatus TokenizerImpl::OnLoad() { return TfmStatus::OK(); };

TfmStatus TokenizerImpl::BatchEncode(
    const std::vector<std::string_view>& input,
    std::vector<std::vector<tfmTokenId_t>>& t_ids) const {
  for (const auto& s : input) {
    std::vector<tfmTokenId_t> ids;
    TfmStatus status = Encode(s, ids);
    if (!status.ok()) {
      return status;
    }
    t_ids.emplace_back(ids);
  }

  return TfmStatus::OK();
}

TfmStatus TokenizerImpl::BatchDecode(const std::vector<span<tfmTokenId_t const>>& t_ids,
                                     std::vector<std::string>& t_text) const {
  for (const auto& s : t_ids) {
    std::string text;
    TfmStatus status = Decode(s, text);
    if (!status.ok()) {
      return status;
    }
    t_text.emplace_back(text);
  }
  return TfmStatus::OK();
}

TfmStatus TokenizerImpl::Id2Token(tfmTokenId_t /* id */, std::string& /* token */, DecoderState** /* state */) const {
  return {kTfmErrorNotImplemented, "Id2Token not implemented for this tokenizer type."};
}

}  // namespace tfm
