// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <filesystem>
#include <algorithm>

#include "tfmtok.h"


namespace tfm {
class TokenId2DArray : public TfmObjectImpl {
 public:
  TokenId2DArray() : TfmObjectImpl(tfmObjectKind_t::kTfmKindTokenId2DArray) {}
  ~TokenId2DArray() override = default;

  void SetTokenIds(std::vector<std::vector<tfmTokenId_t>>&& token_ids) {
    token_ids_ = token_ids;
  }

  [[nodiscard]] const std::vector<std::vector<tfmTokenId_t>>& token_ids() const {
    return token_ids_;
  }

 private:
  std::vector<std::vector<tfmTokenId_t>> token_ids_;
};

class StringArray : public TfmObjectImpl {
 public:
  StringArray() : TfmObjectImpl(tfmObjectKind_t::kTfmKindStringArray) {}
  ~StringArray() override = default;

  void SetStrings(std::vector<std::string>&& strings) {
    strings_ = strings;
  }

  [[nodiscard]] const std::vector<std::string>& strings() const {
    return strings_;
  }

 private:
  std::vector<std::string> strings_;
};

class DetokenizerCache : public TfmObjectImpl {
 public:
  DetokenizerCache() : TfmObjectImpl(tfmObjectKind_t::kTfmKindDetokenizerCache) {}
  ~DetokenizerCache() override = default;

  [[nodiscard]] TfmStatus Detokenize(const std::vector<span<tfmTokenId_t const>>& token_ids,
                                     std::vector<std::string>& output_text) const {
    return {};
  }
};

}  // namespace tfm

using namespace tfm;

thread_local std::string last_error_message;

TfmStatus TfmObjectImpl::IsInstanceOf(tfmObjectKind_t kind) const {
  if (tfm_kind_ != static_cast<int>(kind)) {
    return {tfmError_t::kTfmErrorInvalidArgument,
            "Object is not an instance of the requested type"};
  }
  return {};
}

struct ReturnableStatus {
  ReturnableStatus(TfmStatus&& status) : status_(status) {}
  ~ReturnableStatus() {
    if (!status_.ok()) {
      last_error_message = status_.error_message();
    }
  }
  TfmStatus status_;

  bool ok() const { return status_.ok(); }
  tfmError_t code() const { return status_.code(); }
};

int TFM_API_CALL TfmGetAPIVersion() {
  return API_VERSION;
}

const char* TfmGetLastErrorMessage() {
  return last_error_message.c_str();
}

tfmError_t TFM_API_CALL TfmCreate(tfmObjectKind_t kind, TfmObject** object, ...) {
  if (object == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  if (kind == tfmObjectKind_t::kTfmKindUnknown) {
    return kTfmErrorInvalidArgument;
  }

  if (kind == tfmObjectKind_t::kTfmKindDetokenizerCache) {
    *object = static_cast<TfmObject*>(new TfmDetokenizerCache());
  } /* else if (kind == tfmObjectKind_t::kTfmKindTokenizer) {
    *object = static_cast<TfmObject*>(new TfmTokenizer());
  } */

  return tfmError_t();
}

tfmError_t TFM_API_CALL TfmCreateTokenizer(TfmTokenizer** tokenizer,
                                           const char* tokenizer_path) {
  // test if the tokenizer_path is a valid directory
  if (tokenizer_path == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  if (!std::filesystem::is_directory(tokenizer_path)) {
    last_error_message = std::string("Cannot find the directory of ") + tokenizer_path;
    return kTfmErrorInvalidArgument;
  }

  TfmStatus status;
  auto ptr = tfm::CreateTokenizer(tokenizer_path, "", &status);
  if (status.ok()) {
    *tokenizer = static_cast<TfmTokenizer*>(ptr.release());
    return tfmError_t();
  }

  return status.code();
}

template <typename T>
void Dispose(T* object) {
  auto token_ptr = static_cast<T*>(object);
  std::unique_ptr<T> ptr(token_ptr);
  ptr.reset();
}

tfmError_t TFM_API_CALL TfmDispose(TfmObject** object) {
  if (object == nullptr || *object == nullptr) {
    return kTfmErrorInvalidArgument;
  }

  auto tfm_object = static_cast<TfmObjectImpl*>(*object);
  if (tfm_object->tfm_kind() == tfmObjectKind_t::kTfmKindUnknown) {
    return kTfmErrorInvalidArgument;
  }

  if (tfm_object->tfm_kind() == tfmObjectKind_t::kTfmKindStringArray) {
    Dispose(static_cast<tfm::StringArray*>(*object));
  } else if (tfm_object->tfm_kind() == tfmObjectKind_t::kTfmKindTokenId2DArray) {
    Dispose(static_cast<tfm::TokenId2DArray*>(*object));
  } else if (tfm_object->tfm_kind() == tfmObjectKind_t::kTfmKindDetokenizerCache) {
    Dispose(static_cast<tfm::DetokenizerCache*>(*object));
  } else if (tfm_object->tfm_kind() == tfmObjectKind_t::kTfmKindTokenizer) {
    Dispose(static_cast<tfm::Tokenizer*>(*object));
  }

  *object = nullptr;
  return tfmError_t();
}

tfmError_t TFM_API_CALL TfmTokenize(const TfmTokenizer* tokenizer,
                                    const char* input[], size_t batch_size, TfmTokenId2DArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  auto token_ptr = static_cast<const Tokenizer*>(tokenizer);
  ReturnableStatus status =
      token_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindTokenizer);
  if (!status.ok()) {
    return status.code();
  }

  std::vector<std::vector<tfmTokenId_t>> t_ids;
  std::vector<std::string_view> input_view;
  std::transform(input, input + batch_size, std::back_inserter(input_view),
                 [](const char* str) { return std::string_view(str); });

  status = token_ptr->Tokenize(input_view, t_ids);
  if (!status.ok()) {
    return status.code();
  }

  auto result = std::make_unique<tfm::TokenId2DArray>().release();
  result->SetTokenIds(std::move(t_ids));
  *output = static_cast<TfmTokenId2DArray*>(result);

  return tfmError_t();
}

tfmError_t TFM_API_CALL TfmDetokenize(const TfmTokenizer* tokenizer,
                                      const TfmTokenId2DArray* input, TfmStringArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    return kTfmErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const Tokenizer*>(tokenizer);
  ReturnableStatus status(token_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindTokenizer));
  if (!status.ok()) {
    return status.code();
  }

  auto input_2d = static_cast<const TokenId2DArray*>(input);
  status = input_2d->IsInstanceOf(tfmObjectKind_t::kTfmKindTokenId2DArray);
  if (!status.ok()) {
    return status.code();
  }

  std::vector<span<tfmTokenId_t const>> t_ids;
  std::transform(input_2d->token_ids().begin(), input_2d->token_ids().end(),
                 std::back_inserter(t_ids),
                 [](const std::vector<tfmTokenId_t>& vec) {
                   return span<tfmTokenId_t const>(vec.data(), vec.size());
                 });

  std::vector<std::string> output_text;
  status = token_ptr->Detokenize(t_ids, output_text);
  if (!status.ok()) {
    return status.code();
  }

  auto result = std::make_unique<tfm::StringArray>().release();
  result->SetStrings(std::move(output_text));
  *output = static_cast<TfmStringArray*>(result);

  return tfmError_t();
  ;
}

tfmError_t TFM_API_CALL TfmDetokenize1D(const TfmTokenizer* tokenizer,
                                        const tfmTokenId_t* input,
                                        size_t len,
                                        TfmStringArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const Tokenizer*>(tokenizer);
  ReturnableStatus status(token_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindTokenizer));
  if (!status.ok()) {
    return status.code();
  }

  std::vector<span<tfmTokenId_t const>> t_ids = {{input, len}};
  std::vector<std::string> output_text;
  status = token_ptr->Detokenize(t_ids, output_text);
  if (!status.ok()) {
    return status.code();
  }

  auto result = std::make_unique<tfm::StringArray>().release();
  result->SetStrings(std::move(output_text));
  *output = static_cast<TfmStringArray*>(result);

  return tfmError_t();
}

tfmError_t TFM_API_CALL TfmStringArrayGetBatch(const TfmStringArray* string_array, size_t* length) {
  if (string_array == nullptr || length == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const StringArray*>(string_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindStringArray));
  if (!status.ok()) {
    return status.code();
  }

  *length = token_ptr->strings().size();

  return tfmError_t();
}

tfmError_t TFM_API_CALL TfmStringArrayGetItem(const TfmStringArray* string_array, size_t index, const char** item) {
  if (string_array == nullptr || item == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const StringArray*>(string_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindStringArray));
  if (!status.ok()) {
    return status.code();
  }

  if (index >= token_ptr->strings().size()) {
    last_error_message = "the index is out of range";
    return kTfmErrorInvalidArgument;
  }

  *item = token_ptr->strings()[index].c_str();

  return tfmError_t();
}

tfmError_t TFM_API_CALL TfmTokenId2DArrayGetBatch(const TfmTokenId2DArray* token_id_2d_array, size_t* length) {
  if (token_id_2d_array == nullptr || length == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  const auto token_2d_ptr = static_cast<const TokenId2DArray*>(token_id_2d_array);
  ReturnableStatus status(token_2d_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindTokenId2DArray));
  if (!status.ok()) {
    return status.code();
  }

  *length = token_2d_ptr->token_ids().size();

  return tfmError_t();
}

tfmError_t TFM_API_CALL TfmTokenId2DArrayGetItem(const TfmTokenId2DArray* token_id_2d_array,
                                                 size_t index, const tfmTokenId_t** item, size_t* length) {
  if (token_id_2d_array == nullptr || item == nullptr || length == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const TokenId2DArray*>(token_id_2d_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindTokenId2DArray));
  if (!status.ok()) {
    return status.code();
  }

  if (index >= token_ptr->token_ids().size()) {
    last_error_message = "the index is out of range";
    return kTfmErrorInvalidArgument;
  }

  *item = token_ptr->token_ids()[index].data();
  *length = token_ptr->token_ids()[index].size();

  return tfmError_t();
}

tfmError_t TfmDetokenizeCached(const TfmDetokenizerCache* cache,
                               const TfmTokenId2DArray* input,
                               TfmStringArray** output) {
  if (cache == nullptr || input == nullptr || output == nullptr) {
    last_error_message = "Invalid argument";
    return kTfmErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const DetokenizerCache*>(cache);
  ReturnableStatus status(token_ptr->IsInstanceOf(tfmObjectKind_t::kTfmKindDetokenizerCache));
  if (!status.ok()) {
    return status.code();
  }

  // TODO: implement

  return kTfmErrorUnimplemented;
}
