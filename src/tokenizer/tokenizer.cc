#include "token_bpe.h"
#include "token_spm.h"
#include "token_rwkv.h"

#include <memory>

namespace tfm {

Tokenizer::Tokenizer() : TfmObjectImpl(tfmObjectKind_t::kTfmKindTokenizer) {}

std::unique_ptr<Tokenizer> CreateTokenizer(const std::string& tokenizer_path,
                                           const std::string& tokenizer_type,
                                           TfmStatus* status_return) {
  TfmStatus status;
  std::unique_ptr<Tokenizer> token_ptr;
  if (tokenizer_type == "RWKV") {
    token_ptr = std::make_unique<RwkvTokenizer>();
    status = token_ptr->LoadData({}, tokenizer_path);

  } else if (tokenizer_type == "BPE") {
    auto token_cfg = std::make_unique<TokenConfig>();
    auto config_file = tokenizer_path + "/tokenizer_config.json";
    status = token_cfg->LoadJson(config_file);
    token_ptr = std::make_unique<BPETokenizer>();
    if (status.ok()) {
    } else {
      token_cfg.reset();
    }

  } else {
    auto token_cfg = std::make_unique<TokenConfig>();
    auto config_file = tokenizer_path + "/tokenizer_config.json";
    status = token_cfg->LoadJson(config_file);
    if (status.ok()) {
      if (token_cfg->tokenizer_class_ == "LlamaTokenizer") {
        token_ptr = std::make_unique<SpmTokenizer>();
      } else if (token_cfg->tokenizer_class_ == "GPT2Tokenizer" ||
                 token_cfg->tokenizer_class_ == "CLIPTokenizer" ||
                 token_cfg->tokenizer_class_ == "CodeGenTokenizer") {
        token_ptr = std::make_unique<BPETokenizer>();
      } else {
        status = TfmStatus(kTfmErrorInvalidArgument, "Invalid tokenizer class");
      }

    } else {
      token_cfg.reset();
    }

    if (status.ok()) {
      status = token_ptr->LoadData(std::move(token_cfg), tokenizer_path);
      if (!status.ok()) {
        token_ptr.reset();
      }
    }
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
  return impl.Onload();
}

TfmStatus Tokenizer::Tokenize(const std::vector<std::string_view>& input,
                              std::vector<std::vector<tfmTokenId_t>>& t_ids) const {
  return BatchEncode(input, t_ids);
}

TfmStatus Tokenizer::Detokenize(const std::vector<span<tfmTokenId_t const>>& t_ids,
                                std::vector<std::string>& t_text) const {
  return BatchDecode(t_ids, t_text);
}

TfmStatus TokenizerImpl::Onload() { return TfmStatus::OK(); };

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

}  // namespace tfm
