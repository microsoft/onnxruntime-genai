// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/function.h>
#include <nanobind/ndarray.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#define OGA_USE_SPAN 1
#include "../models/onnxruntime_api.h"
#include "../ort_genai_c.h"
// Note: We use pure C API to avoid lifetime issues with non-copyable C++ wrapper classes
#include <iostream>
#include <numeric>
#include <functional>

namespace nb = nanobind;
using namespace nb::literals;

// Include all wrapper classes
#include "wrappers/oga_wrappers.h"

// If a parameter to a C++ function is an array of float16, this type will let nanobind::ndarray<Ort::Float16_t> map to numpy's float16 format
namespace nanobind::detail {
template <>
struct dtype_traits<Ort::Float16_t> {
  static constexpr dlpack::dtype value{ (uint8_t) dlpack::dtype_code::Float, 16, 1 };
  static constexpr auto name = const_name("float16");
};
} // namespace nanobind::detail

template <typename T, typename... Extra>
auto ToSpan(nb::ndarray<T, Extra...>& v) {
    if constexpr (std::is_const_v<T>) {
        return std::span<const T>(v.data(), v.size());
    } else {
        return std::span<T>(v.mutable_data(), v.size());
    }
}

template <typename T>
std::span<T> ToSpan(OgaTensor& v) {
  OgaElementType type;
  OgaPy::OgaCheckResult(OgaTensorGetType(&v, &type));
  assert(static_cast<ONNXTensorElementDataType>(type) == Ort::TypeToTensorType<T>);

  size_t rank;
  OgaPy::OgaCheckResult(OgaTensorGetShapeRank(&v, &rank));
  std::vector<int64_t> shape(rank);
  OgaPy::OgaCheckResult(OgaTensorGetShape(&v, shape.data(), rank));

  auto element_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  void* data;
  OgaPy::OgaCheckResult(OgaTensorGetData(&v, &data));
  return {reinterpret_cast<T*>(data), static_cast<size_t>(element_count)};
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::shape<-1>> ToPython(std::span<T> v) {
  return nb::ndarray<nb::numpy, T, nb::shape<-1>>(v.data(), {v.size()});
}

ONNXTensorElementDataType ToTensorType(const nb::dlpack::dtype& type) {
  if (type.code == (uint8_t)nb::dlpack::dtype_code::Int) {
    switch (type.bits) {
      case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
      case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
  } else if (type.code == (uint8_t)nb::dlpack::dtype_code::UInt) {
    switch (type.bits) {
      case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
      case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    }
  } else if (type.code == (uint8_t)nb::dlpack::dtype_code::Float) {
    switch (type.bits) {
      case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
  } else if (type.code == (uint8_t)nb::dlpack::dtype_code::Bool) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  }
  throw std::runtime_error("Unsupported numpy type");
}

nb::dlpack::dtype ToDlpackDtype(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return nb::dtype<bool>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return nb::dtype<int8_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return nb::dtype<uint8_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return nb::dtype<int16_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return nb::dtype<uint16_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return nb::dtype<int32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return nb::dtype<uint32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return nb::dtype<int64_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return nb::dtype<uint64_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return nb::dtype<Ort::Float16_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return nb::dtype<float>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return nb::dtype<double>();
        default: throw std::runtime_error("Unsupported onnx type");
    }
}

std::unique_ptr<OgaPy::OgaTensor> ToOgaTensor(nb::ndarray<>& v, bool copy = true) {
  auto type = ToTensorType(v.dtype());

  std::vector<int64_t> shape(v.ndim());
  for (size_t i = 0; i < v.ndim(); i++)
    shape[i] = v.shape(i);

  bool is_c_contig = true;
  if (v.ndim() > 0) {
    int64_t expected_stride = 1;
    for (size_t i = v.ndim(); i-- > 0;) {
      if (v.stride(i) != expected_stride) {
        is_c_contig = false;
        break;
      }
      expected_stride *= v.shape(i);
    }
  }

  bool is_f_contig = true;
  if (v.ndim() > 0) {
    int64_t expected_stride = 1;
    for (size_t i = 0; i < v.ndim(); ++i) {
      if (v.stride(i) != expected_stride) {
        is_f_contig = false;
        break;
      }
      expected_stride *= v.shape(i);
    }
  }

  if (!is_c_contig && !is_f_contig)
    throw std::runtime_error("Array must be contiguous. Please use NumPy's 'ascontiguousarray' method on the value.");

  OgaTensor* p;
  OgaPy::OgaCheckResult(OgaCreateTensorFromBuffer(copy ? nullptr : v.data(), shape.data(), shape.size(), static_cast<OgaElementType>(type), &p));
  auto tensor = std::unique_ptr<OgaPy::OgaTensor>(new OgaPy::OgaTensor(p));

  if (copy) {
    void* data;
    OgaPy::OgaCheckResult(OgaTensorGetData(p, &data));
    auto ort_data = reinterpret_cast<uint8_t*>(data);
    auto python_data = reinterpret_cast<const uint8_t*>(v.data());
    std::copy(python_data, python_data + v.nbytes(), ort_data);
  }
  return tensor;
}

nb::ndarray<> ToNumpy(OgaTensor& v) {
  size_t rank;
  OgaPy::OgaCheckResult(OgaTensorGetShapeRank(&v, &rank));
  std::vector<size_t> shape(rank);
  std::vector<int64_t> shape_i64(rank);
  OgaPy::OgaCheckResult(OgaTensorGetShape(&v, shape_i64.data(), rank));
  for(size_t i=0; i<rank; ++i) shape[i] = shape_i64[i];

  OgaElementType type_enum;
  OgaPy::OgaCheckResult(OgaTensorGetType(&v, &type_enum));

  void* data;
  OgaPy::OgaCheckResult(OgaTensorGetData(&v, &data));

  return nb::ndarray<>(data, rank, shape.data(), nb::handle(), nullptr, ToDlpackDtype(static_cast<ONNXTensorElementDataType>(type_enum)));
}

nb::ndarray<> ToNumpy(std::unique_ptr<OgaPy::OgaTensor> v) {
  OgaTensor* v_ptr = reinterpret_cast<OgaTensor*>(v.get());
  size_t rank;
  OgaPy::OgaCheckResult(OgaTensorGetShapeRank(v_ptr, &rank));
  std::vector<size_t> shape(rank);
  std::vector<int64_t> shape_i64(rank);
  OgaPy::OgaCheckResult(OgaTensorGetShape(v_ptr, shape_i64.data(), rank));
  for(size_t i=0; i<rank; ++i) shape[i] = shape_i64[i];

  OgaElementType type_enum;
  OgaPy::OgaCheckResult(OgaTensorGetType(v_ptr, &type_enum));

  void* data;
  OgaPy::OgaCheckResult(OgaTensorGetData(v_ptr, &data));

  nb::capsule owner(v.release(), [](void *p) noexcept { delete reinterpret_cast<OgaPy::OgaTensor*>(p); });

  return nb::ndarray<>(data, rank, shape.data(), owner, nullptr, ToDlpackDtype(static_cast<ONNXTensorElementDataType>(type_enum)));
}

struct PyGeneratorParams {
  PyGeneratorParams(const OgaPy::OgaModel& model) {
    ::OgaGeneratorParams* p;
    OgaPy::OgaCheckResult(OgaCreateGeneratorParams(model.get(), &p));
    params_.reset(new OgaPy::OgaGeneratorParams(p));
  }

  operator OgaGeneratorParams*() { return reinterpret_cast<OgaGeneratorParams*>(params_.get()); }

  std::unique_ptr<OgaPy::OgaGeneratorParams> params_;

  void SetSearchOptions(const nb::kwargs& dict) {
    for (const auto& entry : dict) {
      auto name = nb::cast<std::string>(entry.first);
      if (nb::isinstance<nb::float_>(entry.second)) {
        OgaPy::OgaCheckResult(OgaGeneratorParamsSetSearchNumber(reinterpret_cast<OgaGeneratorParams*>(params_.get()), name.c_str(), nb::cast<double>(entry.second)));
      } else if (nb::isinstance<nb::bool_>(entry.second)) {
        OgaPy::OgaCheckResult(OgaGeneratorParamsSetSearchBool(reinterpret_cast<OgaGeneratorParams*>(params_.get()), name.c_str(), nb::cast<bool>(entry.second)));
      } else if (nb::isinstance<nb::int_>(entry.second)) {
        OgaPy::OgaCheckResult(OgaGeneratorParamsSetSearchNumber(reinterpret_cast<OgaGeneratorParams*>(params_.get()), name.c_str(), nb::cast<int>(entry.second)));
      } else
        throw std::runtime_error("Unknown search option type, can be float/bool/int:" + name);
    }
  }

  void TryGraphCaptureWithMaxBatchSize(nb::int_ max_batch_size) {
    std::cerr << "TryGraphCaptureWithMaxBatchSize is deprecated and will be removed in a future release" << std::endl;
  }

  void SetGuidance(const std::string& type, const std::string& data, bool enable_ff_tokens = false) {
    OgaPy::OgaCheckResult(OgaGeneratorParamsSetGuidance(reinterpret_cast<OgaGeneratorParams*>(params_.get()), type.c_str(), data.c_str(), enable_ff_tokens));
  }

  std::vector<nb::object> refs_;  // References to data we want to ensure doesn't get garbage collected
};

struct PyGenerator {
  PyGenerator(const OgaPy::OgaModel& model, PyGeneratorParams& params) {
    OgaGenerator* p;
    OgaPy::OgaCheckResult(OgaCreateGenerator(model.get(), params, &p));
    generator_.reset(new OgaPy::OgaGenerator(p));
  }

  nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>> GetNextTokens() {
    // Use wrapper method that handles reference counting
    auto view = generator_->GetNextTokens();
    // Copy data to numpy array (temporal borrow - data invalidated by next generator call)
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1>> result = 
        nb::ndarray<nb::numpy, int32_t, nb::shape<-1>>(
            nb::ndarray<nb::numpy, int32_t>::allocate({view->size()}));
    std::copy(view->begin(), view->end(), result.mutable_data());
    delete view;
    return result;
  }

  nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>> GetSequence(int index) {
    // Use wrapper method that handles reference counting
    auto view = generator_->GetSequenceData(index);
    // Create numpy array that references the view's data
    // Store view in refs_ to keep it (and parent) alive
    auto array_view = nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>>(
        view->data(), {view->size()}, nb::handle());
    // Keep view alive by storing it
    refs_.emplace_back(nb::cast(view, nb::rv_policy::take_ownership));
    return array_view;
  }

  nb::ndarray<> GetInput(const std::string& name) {
    OgaTensor* p;
    OgaPy::OgaCheckResult(OgaGenerator_GetInput(reinterpret_cast<OgaGenerator*>(generator_.get()), name.c_str(), &p));
    return ToNumpy(*p);
  }

  nb::ndarray<> GetOutput(const std::string& name) {
    OgaTensor* p;
    OgaPy::OgaCheckResult(OgaGenerator_GetOutput(reinterpret_cast<OgaGenerator*>(generator_.get()), name.c_str(), &p));
    return ToNumpy(*p);
  }

  void SetModelInput(const std::string& name, nb::ndarray<>& value) {
    OgaPy::OgaCheckResult(OgaGenerator_SetModelInput(reinterpret_cast<OgaGenerator*>(generator_.get()), name.c_str(), reinterpret_cast<OgaTensor*>(ToOgaTensor(value, false).get())));
  }

  void SetInputs(OgaPy::OgaNamedTensors& named_tensors) {
    OgaPy::OgaCheckResult(OgaGenerator_SetInputs(reinterpret_cast<OgaGenerator*>(generator_.get()), named_tensors.get()));
  }

  void AppendTokens(OgaPy::OgaTensor& tokens) {
    auto span = ToSpan<const int32_t>(*tokens.get());
    OgaPy::OgaCheckResult(OgaGenerator_AppendTokens(reinterpret_cast<OgaGenerator*>(generator_.get()), span.data(), span.size()));
  }

  void AppendTokens(nb::ndarray<const int32_t, nb::ndim<1>>& tokens) {
    auto span = ToSpan(tokens);
    OgaPy::OgaCheckResult(OgaGenerator_AppendTokens(reinterpret_cast<OgaGenerator*>(generator_.get()), span.data(), span.size()));
  }

  nb::ndarray<> GetLogits() {
    OgaTensor* p;
    OgaPy::OgaCheckResult(OgaGenerator_GetLogits(reinterpret_cast<OgaGenerator*>(generator_.get()), &p));
    return ToNumpy(std::unique_ptr<OgaPy::OgaTensor>(new OgaPy::OgaTensor(p)));
  }

  void SetLogits(nb::ndarray<>& new_logits) {
    auto tensor = ToOgaTensor(new_logits, false);
    OgaPy::OgaCheckResult(OgaGenerator_SetLogits(reinterpret_cast<OgaGenerator*>(generator_.get()), reinterpret_cast<OgaTensor*>(tensor.get())));
  }

  void GenerateNextToken() {
    OgaPy::OgaCheckResult(OgaGenerator_GenerateNextToken(reinterpret_cast<OgaGenerator*>(generator_.get())));
  }

  void RewindTo(size_t new_length) {
    OgaPy::OgaCheckResult(OgaGenerator_RewindTo(reinterpret_cast<OgaGenerator*>(generator_.get()), new_length));
  }

  bool IsDone() const {
    return OgaGenerator_IsDone(reinterpret_cast<const OgaGenerator*>(generator_.get()));
  }

  void SetActiveAdapter(OgaPy::OgaAdapters& adapters, const std::string& adapter_name) {
    OgaPy::OgaCheckResult(OgaSetActiveAdapter(reinterpret_cast<OgaGenerator*>(generator_.get()), adapters.get(), adapter_name.c_str()));
  }

 private:
  std::unique_ptr<OgaPy::OgaGenerator> generator_;
};

void SetLogOptions(const nb::kwargs& dict) {
  for (const auto& entry : dict) {
    auto name = nb::cast<std::string>(entry.first);
    if (nb::isinstance<nb::bool_>(entry.second)) {
      OgaPy::OgaCheckResult(OgaSetLogBool(name.c_str(), nb::cast<bool>(entry.second)));
    } else if (nb::isinstance<nb::str>(entry.second)) {
      OgaPy::OgaCheckResult(OgaSetLogString(name.c_str(), nb::cast<std::string>(entry.second).c_str()));
    } else
      throw std::runtime_error("Unknown log option type, can be bool/string:" + name);
  }
}

void SetLogCallback(std::optional<nb::callable> callback) {
  // Use a pointer to heap-allocated callable to avoid lifetime issues
  static std::unique_ptr<nb::callable> log_callback_ptr;
  
  if (callback.has_value()) {
    // Store the callback on the heap and keep it alive
    log_callback_ptr = std::make_unique<nb::callable>(callback.value());
    
    OgaPy::OgaCheckResult(OgaSetLogCallback([](const char* message, size_t length) {
      if (log_callback_ptr) {
        nb::gil_scoped_acquire gil;
        (*log_callback_ptr)(std::string_view(message, length));
      }
    }));
  } else {
    // Clear the callback
    log_callback_ptr.reset();
    OgaPy::OgaCheckResult(OgaSetLogCallback(nullptr));
  }
}

NB_MODULE(onnxruntime_genai, m) {
  // Initialize intrusive reference counting for nanobind
  nb::intrusive_init(
      [](PyObject *o) noexcept {
          nb::gil_scoped_acquire guard;
          Py_INCREF(o);
      },
      [](PyObject *o) noexcept {
          nb::gil_scoped_acquire guard;
          Py_DECREF(o);
      });

  m.doc() = R"pbdoc(
        Ort Generators library
        ----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  // Add a cleanup call to happen before global variables are destroyed
  static int unused{};  // The capsule needs something to reference
  nb::capsule cleanup(
      &unused, "cleanup", [](void*) noexcept {
        OgaShutdown();
      });
  m.attr("_cleanup") = cleanup;

  nb::class_<PyGeneratorParams>(m, "GeneratorParams")
      .def(nb::init<const OgaPy::OgaModel&>())
      .def("try_graph_capture_with_max_batch_size", &PyGeneratorParams::TryGraphCaptureWithMaxBatchSize)
      .def("set_search_options", &PyGeneratorParams::SetSearchOptions)  // See config.h 'struct Search' for the options
      .def("set_guidance", &PyGeneratorParams::SetGuidance,
           "type"_a, "data"_a,
           "enable_ff_tokens"_a = false);

  nb::class_<OgaPy::OgaTokenizerStream>(
      m, "TokenizerStream",
      nb::intrusive_ptr<OgaPy::OgaTokenizerStream>(
          [](OgaPy::OgaTokenizerStream *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("decode", [](OgaPy::OgaTokenizerStream& t, int32_t token) { 
        const char* out;
        OgaPy::OgaCheckResult(OgaTokenizerStreamDecode(t.get(), token, &out));
        return out; 
      });

  nb::class_<OgaPy::OgaNamedTensors>(
      m, "NamedTensors",
      nb::intrusive_ptr<OgaPy::OgaNamedTensors>(
          [](OgaPy::OgaNamedTensors *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__", [](OgaPy::OgaNamedTensors* self) { 
        ::OgaNamedTensors* p;
        OgaPy::OgaCheckResult(OgaCreateNamedTensors(&p));
        new (self) OgaPy::OgaNamedTensors(p);
       })
      .def("__getitem__", [](OgaPy::OgaNamedTensors& named_tensors, const std::string& name) {
        OgaTensor* p;
        OgaPy::OgaCheckResult(OgaNamedTensorsGet(named_tensors.get(), name.c_str(), &p));
        if (!p)
          throw std::runtime_error("Tensor with name: " + name + " not found.");
        return new OgaPy::OgaTensor(p);
      })
      .def("__setitem__", [](OgaPy::OgaNamedTensors& named_tensors, const std::string& name, nb::ndarray<>& value) {
        auto tensor = ToOgaTensor(value);
        OgaPy::OgaCheckResult(OgaNamedTensorsSet(named_tensors.get(), name.c_str(), tensor->get()));
      })
      .def("__setitem__", [](OgaPy::OgaNamedTensors& named_tensors, const std::string& name, OgaPy::OgaTensor& value) {
        OgaPy::OgaCheckResult(OgaNamedTensorsSet(named_tensors.get(), name.c_str(), value.get()));
      })
      .def("__contains__", [](const OgaPy::OgaNamedTensors& named_tensors, const std::string& name) {
        OgaTensor* p;
        OgaPy::OgaCheckResult(OgaNamedTensorsGet(const_cast<OgaNamedTensors*>(named_tensors.get()), name.c_str(), &p));
        auto tensor = new OgaPy::OgaTensor(p);
        return tensor != nullptr;
      })
      .def("__delitem__", [](OgaPy::OgaNamedTensors& named_tensors, const std::string& name) {
        OgaPy::OgaCheckResult(OgaNamedTensorsDelete(named_tensors.get(), name.c_str()));
      })
      .def("__len__", [](const OgaPy::OgaNamedTensors& named_tensors) {
        size_t count;
        OgaPy::OgaCheckResult(OgaNamedTensorsCount(named_tensors.get(), &count));
        return count;
      })
      .def("keys", [](OgaPy::OgaNamedTensors& named_tensors) {
        std::vector<std::string> keys;
        OgaStringArray* p;
        OgaPy::OgaCheckResult(OgaNamedTensorsGetNames(named_tensors.get(), &p));
        auto names = new OgaPy::OgaStringArray(p);
        size_t count;
        OgaPy::OgaCheckResult(OgaStringArrayGetCount(reinterpret_cast<OgaStringArray*>(names), &count));
        for (size_t i = 0; i < count; i++) {
          const char* str;
          OgaPy::OgaCheckResult(OgaStringArrayGetString(reinterpret_cast<OgaStringArray*>(names), i, &str));
          keys.push_back(str);
        }
        return keys;
      });

  nb::class_<OgaPy::OgaTensor>(
      m, "Tensor",
      nb::intrusive_ptr<OgaPy::OgaTensor>(
          [](OgaPy::OgaTensor *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__", [](OgaPy::OgaTensor* self, nb::ndarray<>& v) {
        auto temp_tensor = ToOgaTensor(v); // Returns unique_ptr<OgaPy::OgaTensor>
        // Move the C pointer ownership from temp_tensor to self
        ::OgaTensor* c_ptr = temp_tensor->get();
        // Construct self with the C pointer
        new (self) OgaPy::OgaTensor(c_ptr);
        // Now prevent temp_tensor from destroying the C pointer
        // We do this by creating a new OgaTensor wrapper that doesn't own it
        temp_tensor.reset(new OgaPy::OgaTensor(nullptr));
      })
      .def("shape", [](OgaPy::OgaTensor& t) { 
        size_t rank;
        OgaPy::OgaCheckResult(OgaTensorGetShapeRank(t.get(), &rank));
        std::vector<int64_t> shape(rank);
        OgaPy::OgaCheckResult(OgaTensorGetShape(t.get(), shape.data(), rank));
        return shape;
      })
      .def("type", [](OgaPy::OgaTensor& t) { 
        OgaElementType type;
        OgaPy::OgaCheckResult(OgaTensorGetType(t.get(), &type));
        return type;
      })
      .def("data", [](OgaPy::OgaTensor& t) {
        void* data;
        OgaPy::OgaCheckResult(OgaTensorGetData(t.get(), &data));
        return nb::capsule(data, "pointer");
      })
      .def("as_numpy", [](OgaPy::OgaTensor& t) { return ToNumpy(*t.get()); });

  nb::class_<OgaPy::OgaTokenizer>(
      m, "Tokenizer",
      nb::intrusive_ptr<OgaPy::OgaTokenizer>(
          [](OgaPy::OgaTokenizer *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__", [](OgaPy::OgaTokenizer* self, const OgaPy::OgaModel& model) {
        ::OgaTokenizer* p;
        OgaPy::OgaCheckResult(OgaCreateTokenizer(model.get(), &p));
        new (self) OgaPy::OgaTokenizer(p);
      })
      .def_prop_ro("bos_token_id", [](const OgaPy::OgaTokenizer& t) {
        int32_t token_id;
        OgaPy::OgaCheckResult(OgaTokenizerGetBosTokenId(t.get(), &token_id));
        return token_id;
      })
      .def_prop_ro("eos_token_ids", [](OgaPy::OgaTokenizer& t) {
        // Use wrapper method that handles reference counting
        auto view = t.GetEosTokenIds();
        // Create numpy array that references view's data
        auto array_view = nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>>(
            view->data(), {view->size()}, nb::cast(view, nb::rv_policy::take_ownership));
        return array_view;
      })
      .def_prop_ro("pad_token_id", [](const OgaPy::OgaTokenizer& t) {
        int32_t token_id;
        OgaPy::OgaCheckResult(OgaTokenizerGetPadTokenId(t.get(), &token_id));
        return token_id;
      })
      .def("update_options", [](OgaPy::OgaTokenizer& t, nb::kwargs kwargs) {
        std::vector<std::string> key_storage;
        std::vector<std::string> value_storage;
        key_storage.reserve(kwargs.size());
        value_storage.reserve(kwargs.size());

        std::vector<const char*> keys;
        std::vector<const char*> values;
        keys.reserve(kwargs.size());
        values.reserve(kwargs.size());

        for (const auto& item : kwargs) {
            key_storage.emplace_back(nb::cast<std::string>(item.first));
            value_storage.emplace_back(nb::cast<std::string>(item.second));
            keys.push_back(key_storage.back().c_str());
            values.push_back(value_storage.back().c_str());
        }
        OgaPy::OgaCheckResult(OgaUpdateTokenizerOptions(t.get(), keys.data(), values.data(), kwargs.size()));
       })
      .def("encode", [](OgaPy::OgaTokenizer& t, std::string s) -> nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>> {
        // Create sequences
        OgaSequences* sequences_ptr;
        OgaPy::OgaCheckResult(OgaCreateSequences(&sequences_ptr));
        
        auto sequences = new OgaPy::OgaSequences(sequences_ptr);
        
        // Encode using C API
        OgaPy::OgaCheckResult(OgaTokenizerEncode(t.get(), s.c_str(), sequences->get()));
        
        // Use wrapper method to get sequence data with proper ref counting
        auto view = sequences->GetSequenceData(0);
        
        // Create numpy array that references view's data
        // The view keeps the parent alive
        auto array_view = nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>>(
            view->data(), {view->size()}, nb::cast(view, nb::rv_policy::take_ownership));
        
        // Clean up sequences (view keeps it alive)
        delete sequences;
        
        return array_view;
      })
      .def("to_token_id", [](const OgaPy::OgaTokenizer& t, const char* str) {
        int32_t token_id;
        OgaPy::OgaCheckResult(OgaTokenizerToTokenId(t.get(), str, &token_id));
        return token_id;
       })
      .def("decode", [](const OgaPy::OgaTokenizer& t, nb::ndarray<const int32_t, nb::ndim<1>> tokens) -> std::string { 
        auto span = ToSpan(tokens);
        const char* p;
        OgaPy::OgaCheckResult(OgaTokenizerDecode(t.get(), span.data(), span.size(), &p));
        return std::string(OgaPy::OgaString(p));
      })
      .def("apply_chat_template", [](const OgaPy::OgaTokenizer& t, const char* messages, const char* template_str, const char* tools, bool add_generation_prompt) -> std::string { 
        const char *p;
        OgaPy::OgaCheckResult(OgaTokenizerApplyChatTemplate(t.get(), template_str, messages, tools, add_generation_prompt, &p));
        return std::string(OgaPy::OgaString(p));
      }, "messages"_a, nb::kw_only(), "template_str"_a = nullptr, "tools"_a = nullptr, "add_generation_prompt"_a = true)
      .def("encode_batch", [](const OgaPy::OgaTokenizer& t, std::vector<std::string> strings) {
        std::vector<const char*> c_strings;
        for (const auto& s : strings)
          c_strings.push_back(s.c_str());
        OgaTensor* p;
        OgaPy::OgaCheckResult(OgaTokenizerEncodeBatch(t.get(), c_strings.data(), c_strings.size(), &p));
        return new OgaPy::OgaTensor(p);
       })
      .def("decode_batch", [](const OgaPy::OgaTokenizer& t, const OgaPy::OgaTensor& tokens) {
        std::vector<std::string> strings;
        OgaStringArray* p;
        OgaPy::OgaCheckResult(OgaTokenizerDecodeBatch(t.get(), tokens.get(), &p));
        auto decoded = new OgaPy::OgaStringArray(p);
        size_t count;
        OgaPy::OgaCheckResult(OgaStringArrayGetCount(reinterpret_cast<OgaStringArray*>(decoded), &count));
        for (size_t i = 0; i < count; i++) {
          const char* str;
          OgaPy::OgaCheckResult(OgaStringArrayGetString(reinterpret_cast<OgaStringArray*>(decoded), i, &str));
          strings.push_back(str);
        }
        return strings; 
      })
      .def("create_stream", [](const OgaPy::OgaTokenizer& t) { 
        OgaTokenizerStream* p;
        OgaPy::OgaCheckResult(OgaCreateTokenizerStream(t.get(), &p));
        return new OgaPy::OgaTokenizerStream(p);
      });

  nb::class_<OgaPy::OgaConfig>(
      m, "Config",
      nb::intrusive_ptr<OgaPy::OgaConfig>(
          [](OgaPy::OgaConfig *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__", [](OgaPy::OgaConfig* self, const std::string& config_path) {
        ::OgaConfig* p;
        OgaPy::OgaCheckResult(OgaCreateConfig(config_path.c_str(), &p));
        new (self) OgaPy::OgaConfig(p);
      })
      .def("append_provider", [](OgaPy::OgaConfig& config, const char* provider) { OgaPy::OgaCheckResult(OgaConfigAppendProvider(config.get(), provider));})
      .def("set_provider_option", [](OgaPy::OgaConfig& config, const char* provider, const char* name, const char* value) { OgaPy::OgaCheckResult(OgaConfigSetProviderOption(config.get(), provider, name, value));})
      .def("clear_providers", [](OgaPy::OgaConfig& config) { OgaPy::OgaCheckResult(OgaConfigClearProviders(config.get()));})
      .def("add_model_data", [](OgaPy::OgaConfig& config, const std::string& model_filename, nb::object obj) {
        if (nb::isinstance<nb::bytes>(obj)) {
          auto model_bytes = nb::cast<nb::bytes>(obj);
          OgaPy::OgaCheckResult(OgaConfigAddModelData(config.get(), model_filename.c_str(), model_bytes.data(), model_bytes.size()));
        } else if (nb::isinstance<nb::ndarray<>>(obj)) {
          auto array = nb::cast<nb::ndarray<nb::ro, uint8_t, nb::ndim<1>>>(obj);
          OgaPy::OgaCheckResult(OgaConfigAddModelData(config.get(), model_filename.c_str(), array.data(), array.nbytes()));
        } else {
          throw std::runtime_error("Unsupported input type. Expected bytes or a 1D uint8 numpy array.");
        }
      })
      .def("remove_model_data", [](OgaPy::OgaConfig& config, const std::string& model_filename) {
        OgaPy::OgaCheckResult(OgaConfigRemoveModelData(config.get(), model_filename.c_str()));
      })
      .def("overlay", [](OgaPy::OgaConfig& config, const char* json) { OgaPy::OgaCheckResult(OgaConfigOverlay(config.get(), json)); })
      .def("set_decoder_provider_options_hardware_device_type", [](OgaPy::OgaConfig& config, const char* provider, const char* hardware_device_type) { OgaPy::OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareDeviceType(config.get(), provider, hardware_device_type)); })
      .def("set_decoder_provider_options_hardware_device_id", [](OgaPy::OgaConfig& config, const char* provider, uint32_t hardware_device_id) { OgaPy::OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareDeviceId(config.get(), provider, hardware_device_id)); })
      .def("set_decoder_provider_options_hardware_vendor_id", [](OgaPy::OgaConfig& config, const char* provider, uint32_t hardware_vendor_id) { OgaPy::OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareVendorId(config.get(), provider, hardware_vendor_id)); })
      .def("clear_decoder_provider_options_hardware_device_type", [](OgaPy::OgaConfig& config, const char* provider) { OgaPy::OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareDeviceType(config.get(), provider)); })
      .def("clear_decoder_provider_options_hardware_device_id", [](OgaPy::OgaConfig& config, const char* provider) { OgaPy::OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareDeviceId(config.get(), provider)); })
      .def("clear_decoder_provider_options_hardware_vendor_id", [](OgaPy::OgaConfig& config, const char* provider) { OgaPy::OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareVendorId(config.get(), provider)); });

  nb::class_<OgaPy::OgaModel>(
      m, "Model",
      nb::intrusive_ptr<OgaPy::OgaModel>(
          [](OgaPy::OgaModel *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__", [](OgaPy::OgaModel* self, const OgaPy::OgaConfig& config) {
        ::OgaModel* p;
        OgaPy::OgaCheckResult(OgaCreateModelFromConfig(config.get(), &p));
        new (self) OgaPy::OgaModel(p);
      })
      .def("__init__", [](OgaPy::OgaModel* self, const std::string& config_path) {
        ::OgaModel* p;
        OgaPy::OgaCheckResult(OgaCreateModel(config_path.c_str(), &p));
        new (self) OgaPy::OgaModel(p);
      })
      .def_prop_ro("type", [](const OgaPy::OgaModel& model) -> std::string { 
        const char *p;
        OgaPy::OgaCheckResult(OgaModelGetType(model.get(), &p));
        return std::string(OgaPy::OgaString(p));
      })
      .def_prop_ro(
          "device_type", [](const OgaPy::OgaModel& model) -> std::string { 
            const char* p;
            OgaPy::OgaCheckResult(OgaModelGetDeviceType(model.get(), &p));
            return std::string(OgaPy::OgaString(p));
          }, "The device type the model is running on")
      .def("create_multimodal_processor", [](const OgaPy::OgaModel& model) { 
        OgaMultiModalProcessor* p;
        OgaPy::OgaCheckResult(OgaCreateMultiModalProcessor(model.get(), &p));
        return new OgaPy::OgaMultiModalProcessor(p);
      });

  nb::class_<PyGenerator>(m, "Generator")
      .def(nb::init<const OgaPy::OgaModel&, PyGeneratorParams&>())
      .def("is_done", &PyGenerator::IsDone)
      .def("get_input", &PyGenerator::GetInput)
      .def("get_output", &PyGenerator::GetOutput)
      .def("set_inputs", &PyGenerator::SetInputs)
      .def("set_model_input", &PyGenerator::SetModelInput)
      .def("append_tokens", nb::overload_cast<nb::ndarray<const int32_t, nb::ndim<1>>&>(&PyGenerator::AppendTokens))
      .def("append_tokens", nb::overload_cast<OgaPy::OgaTensor&>(&PyGenerator::AppendTokens))
      .def("get_logits", &PyGenerator::GetLogits)
      .def("set_logits", &PyGenerator::SetLogits)
      .def("generate_next_token", &PyGenerator::GenerateNextToken)
      .def("rewind_to", &PyGenerator::RewindTo)
      .def("get_next_tokens", &PyGenerator::GetNextTokens)
      .def("get_sequence", &PyGenerator::GetSequence)
      .def("set_active_adapter", &PyGenerator::SetActiveAdapter);

  nb::class_<OgaPy::OgaImages>(
      m, "Images",
      nb::intrusive_ptr<OgaPy::OgaImages>(
          [](OgaPy::OgaImages *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def_static("open", [](nb::args image_paths) {
        std::vector<std::string> image_paths_string;
        std::vector<const char*> image_paths_vector;
        for (auto image_path : image_paths) {
          if (!nb::isinstance<nb::str>(image_path))
            throw std::runtime_error("Image paths must be strings.");
          image_paths_string.push_back(nb::cast<std::string>(image_path));
          image_paths_vector.push_back(image_paths_string.back().c_str());
        }
        OgaStringArray* p_strs;
        OgaPy::OgaCheckResult(OgaCreateStringArrayFromStrings(image_paths_vector.data(), image_paths_vector.size(), &p_strs));
        auto strs = new OgaPy::OgaStringArray(p_strs);
        OgaImages* p;
        OgaPy::OgaCheckResult(OgaLoadImages(reinterpret_cast<OgaStringArray*>(strs), &p));
        return new OgaPy::OgaImages(p);
      })
      .def_static("open_bytes", [](nb::args image_datas) {
        std::vector<const void*> image_raw_data(image_datas.size());
        std::vector<size_t> image_sizes(image_datas.size());
        for (size_t i = 0; i < image_datas.size(); ++i) {
          if (!nb::isinstance<nb::bytes>(image_datas[i]))
            throw std::runtime_error("Image data must be bytes.");
          auto bytes = nb::cast<nb::bytes>(image_datas[i]);
          image_raw_data[i] = bytes.data();
          image_sizes[i] = bytes.size();
        }
        OgaImages* p;
        OgaPy::OgaCheckResult(OgaLoadImagesFromBuffers(image_raw_data.data(), image_sizes.data(), image_raw_data.size(), &p));
        return new OgaPy::OgaImages(p);
      });

  nb::class_<OgaPy::OgaAudios>(
      m, "Audios",
      nb::intrusive_ptr<OgaPy::OgaAudios>(
          [](OgaPy::OgaAudios *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def_static("open", [](nb::args audio_paths) {
        std::vector<std::string> audio_paths_string;
        std::vector<const char*> audio_paths_vector;

        for (const auto& audio_path : audio_paths) {
          if (!nb::isinstance<nb::str>(audio_path))
            throw std::runtime_error("Audio paths must be strings.");
          audio_paths_string.push_back(nb::cast<std::string>(audio_path));
          audio_paths_vector.push_back(audio_paths_string.back().c_str());
        }
        OgaStringArray* p_strs;
        OgaPy::OgaCheckResult(OgaCreateStringArrayFromStrings(audio_paths_vector.data(), audio_paths_vector.size(), &p_strs));
        auto strs = new OgaPy::OgaStringArray(p_strs);
        OgaAudios* p;
        OgaPy::OgaCheckResult(OgaLoadAudios(reinterpret_cast<OgaStringArray*>(strs), &p));
        return new OgaPy::OgaAudios(p);
      })
      .def_static("open_bytes", [](nb::args audio_datas) {
        std::vector<const void*> audio_raw_data(audio_datas.size());
        std::vector<size_t> audio_sizes(audio_datas.size());
        for (size_t i = 0; i < audio_datas.size(); ++i) {
          if (!nb::isinstance<nb::bytes>(audio_datas[i]))
            throw std::runtime_error("Audio data must be bytes.");
          auto bytes = nb::cast<nb::bytes>(audio_datas[i]);
          audio_raw_data[i] = bytes.data();
          audio_sizes[i] = bytes.size();
        }

        OgaAudios*p;
        OgaPy::OgaCheckResult(OgaLoadAudiosFromBuffers(audio_raw_data.data(), audio_sizes.data(), audio_raw_data.size(), &p));
        return new OgaPy::OgaAudios(p);
      });

  nb::class_<OgaPy::OgaMultiModalProcessor>(
      m, "MultiModalProcessor",
      nb::intrusive_ptr<OgaPy::OgaMultiModalProcessor>(
          [](OgaPy::OgaMultiModalProcessor *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__call__", [](OgaPy::OgaMultiModalProcessor& processor, nb::object prompts, const nb::kwargs& kwargs) {
            OgaImages* images{};
            OgaAudios* audios{};
            if (kwargs.contains("images")) {
              images = reinterpret_cast<OgaImages*>(nb::cast<OgaPy::OgaImages*>(kwargs["images"]));
            }
            if (kwargs.contains("audios")) {
              audios = reinterpret_cast<OgaAudios*>(nb::cast<OgaPy::OgaAudios*>(kwargs["audios"]));
            }

            std::vector<std::string> prompts_str;
            std::vector<const char*> c_prompts;
            if (nb::isinstance<nb::str>(prompts)) {
              // One prompt
              OgaNamedTensors* p;
              OgaPy::OgaCheckResult(OgaProcessorProcessImagesAndAudios(processor.get(), nb::cast<std::string>(prompts).c_str(), images, audios, &p));
              return new OgaPy::OgaNamedTensors(p);
            } else if (nb::isinstance<nb::list>(prompts)) {
              // Multiple prompts
              for (const auto& prompt : prompts) {
                if (!nb::isinstance<nb::str>(prompt)) {
                  throw std::runtime_error("One or more items in the list of provided prompts is not a string.");
                }
                prompts_str.push_back(nb::cast<std::string>(prompt));
                c_prompts.push_back(prompts_str.back().c_str());
              }
            } else if (!prompts.is_none()) {
              // Unsupported type for prompts
              throw std::runtime_error("Unsupported type for prompts. Prompts must be a string or a list of strings.");
            }

            OgaStringArray* p_strs;
            OgaPy::OgaCheckResult(OgaCreateStringArrayFromStrings(c_prompts.data(), c_prompts.size(), &p_strs));
            auto strs = new OgaPy::OgaStringArray(p_strs);
            OgaNamedTensors* p;
            OgaPy::OgaCheckResult(OgaProcessorProcessImagesAndAudiosAndPrompts(processor.get(), reinterpret_cast<OgaStringArray*>(strs), images, audios, &p));
            return new OgaPy::OgaNamedTensors(p);
          }, "prompts"_a = nb::none(), nb::arg())
      .def("create_stream", [](OgaPy::OgaMultiModalProcessor& processor) { 
        OgaTokenizerStream* p;
        OgaPy::OgaCheckResult(OgaCreateTokenizerStreamFromProcessor(processor.get(), &p));
        return new OgaPy::OgaTokenizerStream(p);
      })
      .def("decode", [](OgaPy::OgaMultiModalProcessor& processor, nb::ndarray<const int32_t, nb::ndim<1>> tokens) -> std::string {
        auto span = ToSpan(tokens);
        const char *p;
        OgaPy::OgaCheckResult(OgaProcessorDecode(processor.get(), span.data(), span.size(), &p));
        return std::string(OgaPy::OgaString(p));
      });

  nb::class_<OgaPy::OgaAdapters>(
      m, "Adapters",
      nb::intrusive_ptr<OgaPy::OgaAdapters>(
          [](OgaPy::OgaAdapters *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__", [](OgaPy::OgaAdapters* self, OgaPy::OgaModel& model) {
        ::OgaAdapters* p;
        OgaPy::OgaCheckResult(OgaCreateAdapters(model.get(), &p));
        new (self) OgaPy::OgaAdapters(p);
      })
      .def("unload", [](OgaPy::OgaAdapters& adapters, const char* adapter_name){ OgaPy::OgaCheckResult(OgaUnloadAdapter(adapters.get(), adapter_name));})
      .def("load", [](OgaPy::OgaAdapters& adapters, const char* adapter_file_path, const char* adapter_name){ OgaPy::OgaCheckResult(OgaLoadAdapter(adapters.get(), adapter_file_path, adapter_name));});

  nb::class_<OgaPy::OgaRequest>(
      m, "Request",
      nb::intrusive_ptr<OgaPy::OgaRequest>(
          [](OgaPy::OgaRequest *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__",
          [](OgaPy::OgaRequest* self, PyGeneratorParams& params) {
            ::OgaRequest* p;
            OgaPy::OgaCheckResult(OgaCreateRequest(params, &p));
            new (self) OgaPy::OgaRequest(p);
          })
      .def("add_tokens", [](OgaPy::OgaRequest& request, nb::ndarray<const int32_t, nb::ndim<1>> tokens) {
        OgaSequences* p;
        OgaPy::OgaCheckResult(OgaCreateSequences(&p));
        auto sequences = new OgaPy::OgaSequences(p);
        auto tokens_span = ToSpan(tokens);
        OgaPy::OgaCheckResult(OgaAppendTokenSequence(tokens_span.data(), tokens_span.size(), reinterpret_cast<OgaSequences*>(sequences)));
        OgaPy::OgaCheckResult(OgaRequestAddTokens(request.get(), reinterpret_cast<OgaSequences*>(sequences)));
      })
      .def("has_unseen_tokens", [](const OgaPy::OgaRequest& request) {
        bool has_unseen_tokens;
        OgaPy::OgaCheckResult(OgaRequestHasUnseenTokens(request.get(), &has_unseen_tokens));
        return has_unseen_tokens;
      })
      .def("is_done", [](const OgaPy::OgaRequest& request) {
        bool is_done;
        OgaPy::OgaCheckResult(OgaRequestIsDone(request.get(), &is_done));
        return is_done;
      })
      .def("get_unseen_token", [](OgaPy::OgaRequest& request) {
        int32_t token;
        OgaPy::OgaCheckResult(OgaRequestGetUnseenToken(request.get(), &token));
        return token;
      })
      .def("set_opaque_data", [](OgaPy::OgaRequest& request, nb::object opaque_data) {
        OgaPy::OgaCheckResult(OgaRequestSetOpaqueData(request.get(), opaque_data.ptr()));
      })
      .def("get_opaque_data", [](OgaPy::OgaRequest& request) -> nb::object {
        void* data;
        OgaPy::OgaCheckResult(OgaRequestGetOpaqueData(request.get(), &data));
        if (!data)
          return nb::none();
        return nb::borrow<nb::object>(static_cast<PyObject*>(data));
      });

  nb::class_<OgaPy::OgaEngine>(
      m, "Engine",
      nb::intrusive_ptr<OgaPy::OgaEngine>(
          [](OgaPy::OgaEngine *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__init__", [](OgaPy::OgaEngine* self, OgaPy::OgaModel& model) {
        ::OgaEngine* p;
        OgaPy::OgaCheckResult(OgaCreateEngine(model.get(), &p));
        new (self) OgaPy::OgaEngine(p);
      })
      .def("add_request", [](OgaPy::OgaEngine& engine, OgaPy::OgaRequest& request){ OgaPy::OgaCheckResult(OgaEngineAddRequest(engine.get(), request.get())); })
      .def("step", [](OgaPy::OgaEngine& engine) { 
        OgaRequest* p;
        OgaPy::OgaCheckResult(OgaEngineStep(engine.get(), &p));
        return new OgaPy::OgaRequest(p);
      })
      .def("remove_request", [](OgaPy::OgaEngine& engine, OgaPy::OgaRequest& request){ OgaPy::OgaCheckResult(OgaEngineRemoveRequest(engine.get(), request.get()));})
      .def("has_pending_requests", [](OgaPy::OgaEngine& engine) {
        bool has_pending_requests;
        OgaPy::OgaCheckResult(OgaEngineHasPendingRequests(engine.get(), &has_pending_requests));
        return has_pending_requests;
      });

  // Bind borrowed array view classes with buffer protocol for numpy interop
  nb::class_<OgaPy::SequenceDataView>(
      m, "SequenceDataView",
      nb::intrusive_ptr<OgaPy::SequenceDataView>(
          [](OgaPy::SequenceDataView *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__len__", [](const OgaPy::SequenceDataView& view) { return view.size(); })
      .def("__getitem__", [](const OgaPy::SequenceDataView& view, size_t index) { return view[index]; })
      .def_buffer([](OgaPy::SequenceDataView* view) {
        return nb::ndarray<nb::numpy, const int32_t>(
            view->data(), {view->size()}, nb::handle());
      });

  nb::class_<OgaPy::GeneratorSequenceDataView>(
      m, "GeneratorSequenceDataView",
      nb::intrusive_ptr<OgaPy::GeneratorSequenceDataView>(
          [](OgaPy::GeneratorSequenceDataView *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__len__", [](const OgaPy::GeneratorSequenceDataView& view) { return view.size(); })
      .def("__getitem__", [](const OgaPy::GeneratorSequenceDataView& view, size_t index) { return view[index]; })
      .def_buffer([](OgaPy::GeneratorSequenceDataView* view) {
        return nb::ndarray<nb::numpy, const int32_t>(
            view->data(), {view->size()}, nb::handle());
      });

  nb::class_<OgaPy::NextTokensView>(
      m, "NextTokensView",
      nb::intrusive_ptr<OgaPy::NextTokensView>(
          [](OgaPy::NextTokensView *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__len__", [](const OgaPy::NextTokensView& view) { return view.size(); })
      .def("__getitem__", [](const OgaPy::NextTokensView& view, size_t index) { return view[index]; })
      .def_buffer([](OgaPy::NextTokensView* view) {
        return nb::ndarray<nb::numpy, const int32_t>(
            view->data(), {view->size()}, nb::handle());
      });

  nb::class_<OgaPy::EosTokenIdsView>(
      m, "EosTokenIdsView",
      nb::intrusive_ptr<OgaPy::EosTokenIdsView>(
          [](OgaPy::EosTokenIdsView *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def("__len__", [](const OgaPy::EosTokenIdsView& view) { return view.size(); })
      .def("__getitem__", [](const OgaPy::EosTokenIdsView& view, size_t index) { return view[index]; })
      .def_buffer([](OgaPy::EosTokenIdsView* view) {
        return nb::ndarray<nb::numpy, const int32_t>(
            view->data(), {view->size()}, nb::handle());
      });

  m.def("set_log_options", &SetLogOptions);
  m.def("set_log_callback", [](nb::handle callback) {
    if (callback.is_none()) {
      SetLogCallback(std::nullopt);
    } else {
      SetLogCallback(nb::cast<nb::callable>(callback));
    }
  });

  m.def("is_cuda_available", []() { return USE_CUDA != 0; });
  m.def("is_dml_available", []() { return USE_DML != 0; });
  m.def("is_rocm_available", []() { return USE_ROCM != 0; });
  m.def("is_webgpu_available", []() { return true; });
  m.def("is_qnn_available", []() { return true; });
  m.def("is_openvino_available", []() { return true; });

  m.def("set_current_gpu_device_id", [](int device_id) { OgaPy::OgaCheckResult(OgaSetCurrentGpuDeviceId(device_id)); });
  m.def("get_current_gpu_device_id", []() { 
    int device_id;
    OgaPy::OgaCheckResult(OgaGetCurrentGpuDeviceId(&device_id));
    return device_id;
  });

  m.def("register_execution_provider_library", [](const std::string& provider_name, const std::string& path_str) {
    OgaRegisterExecutionProviderLibrary(provider_name.c_str(), path_str.c_str());
  });

  m.def("unregister_execution_provider_library", [](const std::string& provider_name) {
    OgaUnregisterExecutionProviderLibrary(provider_name.c_str());
  });
}
