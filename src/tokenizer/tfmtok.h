// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tfmtok_c.h"

#include <string>
#include <vector>
#include <string_view>
#include <memory>

class TfmStatus final {
 public:
  TfmStatus();
  ~TfmStatus();
  TfmStatus(tfmError_t code, std::string_view error_message);
  TfmStatus(const TfmStatus& s);
  TfmStatus& operator=(const TfmStatus& s);
  bool operator==(const TfmStatus& s) const;
  bool operator!=(const TfmStatus& s) const;
  [[nodiscard]] inline bool ok() const { return rep_ == nullptr; }

  void SetErrorMessage(const char* str);
  [[nodiscard]] const char* error_message() const;
  [[nodiscard]] const char* message() const { return error_message(); }
  [[nodiscard]] tfmError_t code() const;
  [[nodiscard]] std::string ToString() const;

 private:
  struct Rep;
  std::unique_ptr<Rep> rep_;

 public:
  static TfmStatus OK();
};

class TfmObjectImpl : public TfmObject {
 public:
  explicit TfmObjectImpl(tfmObjectKind_t kind = tfmObjectKind_t::kTfmKindUnknown) : TfmObject() {
    tfm_kind_ = static_cast<int>(kind);
  };
  virtual ~TfmObjectImpl() = default;

  [[nodiscard]] TfmStatus IsInstanceOf(tfmObjectKind_t kind) const;
  [[nodiscard]] tfmObjectKind_t tfm_kind() const {
    if (tfm_kind_ < static_cast<int>(tfmObjectKind_t::kTfmKindBegin) ||
        tfm_kind_ >= static_cast<int>(tfmObjectKind_t::kTfmKindEnd)) {
      return tfmObjectKind_t::kTfmKindUnknown;
    }
    return static_cast<tfmObjectKind_t>(tfm_kind_);
  }
};

namespace tfm {

class TokenConfig;

template <typename T>
class span {
 public:
  using value_type = std::remove_cv_t<T>;

  span(T* d, size_t s) : data_(d), size_(s) {}
  span(std::vector<value_type>& v) {
    data_ = v.data();
    size_ = v.size();
  }

  T* data() const { return data_; }
  [[nodiscard]] size_t size() const { return size_; }
  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

 private:
  T* data_;
  size_t size_;
};

/**
 * @brief The Tokenizer class is responsible for tokenizing and detokenizing text.
 */
class Tokenizer : public TfmObjectImpl {
 public:
  /**
   * @brief Loads the token configuration data and tokenizer directory.
   *
   * @param token_cfg A unique pointer to the token configuration.
   * @param tokenizer_dir The directory path of the tokenizer.
   * @return The status of the load operation.
   */
  TfmStatus LoadData(std::unique_ptr<TokenConfig> token_cfg, const std::string& tokenizer_dir);

  /**
   * @brief Tokenizes the input text.
   *
   * @param input The vector of input strings to be tokenized.
   * @param t_ids The vector of token IDs for each input string.
   * @return The result of the tokenization operation.
   */
  TfmStatus Tokenize(const std::vector<std::string_view>& input, std::vector<std::vector<tfmTokenId_t>>& t_ids) const;

  /**
   * @brief Detokenizes the token IDs.
   *
   * @param t_ids The vector of token IDs to be detokenized.
   * @param t_text The vector of detokenized text.
   * @return The result of the detokenization operation.
   */
  TfmStatus Detokenize(const std::vector<span<tfmTokenId_t const>>& t_ids, std::vector<std::string>& t_text) const;

  // the override function for all derived classes.
 protected:
  /**
   * @brief Default constructor for the Tokenizer class.
   */
  Tokenizer();

  /**
   * @brief Callback function called during loading.
   *
   * @return The status of the onload operation.
   */
  virtual TfmStatus Onload() = 0;

  /**
   * @brief Batch encodes the input text.
   *
   * @param input The vector of input strings to be encoded.
   * @param t_ids The vector of token IDs for each input string.
   * @return The status of the batch encoding operation.
   */
  virtual TfmStatus BatchEncode(const std::vector<std::string_view>& input, std::vector<std::vector<tfmTokenId_t>>& t_ids) const = 0;

  /**
   * @brief Batch decodes the token IDs.
   *
   * @param t_ids The vector of token IDs to be decoded.
   * @param t_text The vector of decoded text.
   * @return The status of the batch decoding operation.
   */
  virtual TfmStatus BatchDecode(const std::vector<span<tfmTokenId_t const>>& t_ids, std::vector<std::string>& t_text) const = 0;
};

/**
 * @brief This function creates a Tokenizer object based on the specified tokenizer path and type.
 *
 * @param tokenizer_path The path to the tokenizer.
 * @param tokenizer_type The type of the tokenizer, if empty, the type will be inferred from the tokenizer path.
 * @param status A pointer to a TfmStatus object to store the status of the tokenizer creation.
 * @return A unique pointer to a Tokenizer object.
 */
std::unique_ptr<Tokenizer>
CreateTokenizer(const std::string& tokenizer_path,
                const std::string& tokenizer_type = "",
                TfmStatus* status = nullptr);

}  // namespace tfm
