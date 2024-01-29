// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "tfmtok.h"

struct TfmStatus::Rep {
  tfmError_t code;
  std::string error_message;
};

TfmStatus TfmStatus::OK() { return {}; }

TfmStatus::TfmStatus() = default;
TfmStatus::~TfmStatus() = default;

TfmStatus::TfmStatus(tfmError_t code, std::string_view error_message)
    : rep_(new Rep) {
  rep_->code = code;
  rep_->error_message = std::string(error_message);
}

TfmStatus::TfmStatus(const TfmStatus& s)
    : rep_((s.rep_ == nullptr) ? nullptr : new Rep(*s.rep_)) {}

TfmStatus& TfmStatus::operator=(const TfmStatus& s) {
  if (rep_ != s.rep_)
    rep_.reset((s.rep_ == nullptr) ? nullptr : new Rep(*s.rep_));

  return *this;
}

bool TfmStatus::operator==(const TfmStatus& s) const { return (rep_ == s.rep_); }

bool TfmStatus::operator!=(const TfmStatus& s) const { return (rep_ != s.rep_); }

const char* TfmStatus::error_message() const {
  return ok() ? "" : rep_->error_message.c_str();
}

void TfmStatus::SetErrorMessage(const char* str) {
  if (rep_ == nullptr)
    rep_ = std::make_unique<Rep>();
  rep_->error_message = str;
}

tfmError_t TfmStatus::code() const { return ok() ? tfmError_t() : rep_->code; }

std::string TfmStatus::ToString() const {
  if (rep_ == nullptr)
    return "OK";

  std::string result;
  switch (code()) {
    case tfmError_t::kTfmOK:
      result = "Success";
      break;
    case tfmError_t::kTfmErrorInvalidArgument:
      result = "Invalid argument";
      break;
    case tfmError_t::kTfmErrorOutOfMemory:
      result = "Out of Memory";
      break;
    case tfmError_t::kTfmErrorInvalidFile:
      result = "Invalid data file";
      break;
    case tfmError_t::kTfmErrorNotFound:
      result = "Not found";
      break;
    case tfmError_t::kTfmErrorAlreadyExists:
      result = "Already exists";
      break;
    case tfmError_t::kTfmErrorOutOfRange:
      result = "Out of range";
      break;
    case tfmError_t::kTfmErrorUnimplemented:
      result = "Unimplemented";
      break;
    case tfmError_t::kTfmErrorInternal:
      result = "Internal";
      break;
    case tfmError_t::kTfmErrorUnknown:
      result = "Unknown";
      break;
    default:
      break;
  }

  result += ": ";
  result += rep_->error_message;
  return result;
}
