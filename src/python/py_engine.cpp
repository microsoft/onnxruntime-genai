// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include "../generators.h"
#include "../search.h"
#include "../engine/engine.h"
#include "../engine/request.h"
#include "py_numpy.h"

namespace nb = nanobind;

namespace Generators {

// Forward declarations
struct PyGeneratorParams {
  std::shared_ptr<GeneratorParams> GetParams();
};

struct PyModel {
  std::shared_ptr<Model> GetModel();
};

// Python wrapper for Request
struct PyRequest {
  PyRequest(std::shared_ptr<Request> request) : request_(request) {}
  
  void AddTokens(nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> tokens) {
    auto tokens_data = tokens.data();
    auto tokens_size = tokens.size();
    request_->AddTokens(std::span<const int32_t>(tokens_data, tokens_size));
  }
  
  bool HasUnseenTokens() const { return request_->HasUnseenTokens(); }
  bool IsDone() const { return request_->IsDone(); }
  int32_t GetUnseenToken() { return request_->UnseenToken(); }
  
  void SetOpaqueData(nb::object opaque_data) {
    if (opaque_data.is_none()) {
      request_->SetOpaqueData(nullptr);
    } else {
      // Increment reference count since we're storing it
      opaque_data.inc_ref();
      request_->SetOpaqueData(opaque_data.ptr());
    }
  }
  
  nb::object GetOpaqueData() {
    auto opaque_data = request_->GetOpaqueData();
    if (!opaque_data)
      return nb::none();
    // Borrow reference from the stored Python object
    return nb::borrow<nb::object>(static_cast<PyObject*>(opaque_data));
  }
  
  std::shared_ptr<Request> request_;
};

// Python wrapper for Engine
struct PyEngine {
  PyEngine(std::shared_ptr<Model> model) : engine_(std::make_shared<Engine>(model)) {}
  
  void AddRequest(std::shared_ptr<PyRequest> request) {
    engine_->AddRequest(request->request_);
  }
  
  std::shared_ptr<PyRequest> Step() {
    auto request = engine_->Step();
    if (!request)
      return nullptr;
    return std::make_shared<PyRequest>(request);
  }
  
  void RemoveRequest(std::shared_ptr<PyRequest> request) {
    engine_->RemoveRequest(request->request_);
  }
  
  bool HasPendingRequests() const {
    return engine_->HasPendingRequests();
  }
  
  std::shared_ptr<Engine> engine_;
};

void BindEngine(nb::module_& m) {
  // Request class
  nb::class_<PyRequest>(m, "Request")
      .def("__init__", [](PyRequest* self, PyGeneratorParams& params) {
        auto generator_params = params.GetParams();
        new (self) PyRequest(std::make_shared<Request>(generator_params));
      }, nb::arg("params"))
      .def("add_tokens", &PyRequest::AddTokens, nb::arg("tokens"))
      .def("has_unseen_tokens", &PyRequest::HasUnseenTokens)
      .def("is_done", &PyRequest::IsDone)
      .def("get_unseen_token", &PyRequest::GetUnseenToken)
      .def("set_opaque_data", &PyRequest::SetOpaqueData, nb::arg("opaque_data"))
      .def("get_opaque_data", &PyRequest::GetOpaqueData);

  // Engine class
  nb::class_<PyEngine>(m, "Engine")
      .def("__init__", [](PyEngine* self, PyModel& model) {
        new (self) PyEngine(model.GetModel());
      }, nb::arg("model"))
      .def("add_request", &PyEngine::AddRequest, nb::arg("request"))
      .def("step", &PyEngine::Step)
      .def("remove_request", &PyEngine::RemoveRequest, nb::arg("request"))
      .def("has_pending_requests", &PyEngine::HasPendingRequests);
}

}  // namespace Generators
