// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"
#include "oga_utils.h"

namespace OgaPy {

// Forward declaration
struct OgaSequences;

struct OgaRequest : OgaObject {
  explicit OgaRequest(::OgaRequest* p) : ptr_(p) {}
  ~OgaRequest() override { if (ptr_) OgaDestroyRequest(ptr_); }
  ::OgaRequest* get() const { return ptr_; }
  
  // Add tokens to the request
  void AddTokens(const OgaSequences* tokens) {
    OgaCheckResult(OgaRequestAddTokens(ptr_, tokens->get()));
  }
  
  // Set opaque data on the request
  void SetOpaqueData(void* opaque_data) {
    OgaCheckResult(OgaRequestSetOpaqueData(ptr_, opaque_data));
  }
  
  // Get opaque data from the request
  void* GetOpaqueData() const {
    void* opaque_data = nullptr;
    OgaCheckResult(OgaRequestGetOpaqueData(ptr_, &opaque_data));
    return opaque_data;
  }
  
  // Check if there are unseen tokens
  bool HasUnseenTokens() const {
    bool out = false;
    OgaCheckResult(OgaRequestHasUnseenTokens(ptr_, &out));
    return out;
  }
  
  // Get the next unseen token
  int32_t GetUnseenToken() {
    int32_t out = 0;
    OgaCheckResult(OgaRequestGetUnseenToken(ptr_, &out));
    return out;
  }
  
  // Check if the request is done
  bool IsDone() const {
    bool out = false;
    OgaCheckResult(OgaRequestIsDone(ptr_, &out));
    return out;
  }
  
private:
  ::OgaRequest* ptr_;
};

struct OgaEngine : OgaObject {
  explicit OgaEngine(::OgaEngine* p) : ptr_(p) {}
  ~OgaEngine() override { if (ptr_) OgaDestroyEngine(ptr_); }
  ::OgaEngine* get() const { return ptr_; }
  
  // Step the engine and get the next request
  OgaRequest* Step() {
    ::OgaRequest* request = nullptr;
    OgaCheckResult(OgaEngineStep(ptr_, &request));
    return request ? new OgaRequest(request) : nullptr;
  }
  
  // Check if there are pending requests
  bool HasPendingRequests() const {
    bool out = false;
    OgaCheckResult(OgaEngineHasPendingRequests(ptr_, &out));
    return out;
  }
  
  // Add a request to the engine
  void AddRequest(OgaRequest* request) {
    OgaCheckResult(OgaEngineAddRequest(ptr_, request->get()));
  }
  
  // Remove a request from the engine
  void RemoveRequest(OgaRequest* request) {
    OgaCheckResult(OgaEngineRemoveRequest(ptr_, request->get()));
  }
  
private:
  ::OgaEngine* ptr_;
};

} // namespace OgaPy
