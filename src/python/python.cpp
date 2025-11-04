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
 assert(static_cast<ONNXTensorElementDataType>(type) == Ort::TypeToTensorType<std::remove_const_t<T>>);

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

// Creates a new C-API-level OgaTensor from a numpy array by copying the data.
// The returned ::OgaTensor* has an external refcount of 1.
// The caller is responsible for balancing this with a call to OgaDestroyTensor.
::OgaTensor* CreateOgaTensorFromNdarray(nb::ndarray<>& v) {
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

 if (!is_c_contig)
  throw std::runtime_error("Array must be contiguous. Please use NumPy's 'ascontiguousarray' method on the value.");

 ::OgaTensor* p;
 // Pass nullptr to data to force allocation of a new buffer
 OgaPy::OgaCheckResult(OgaCreateTensorFromBuffer(nullptr, shape.data(), shape.size(), static_cast<OgaElementType>(type), &p));
 // p now has an external refcount of 1

 // Copy data from numpy array into the new tensor
 void* data;
 OgaPy::OgaCheckResult(OgaTensorGetData(p, &data));
 auto ort_data = reinterpret_cast<uint8_t*>(data);
 auto python_data = reinterpret_cast<const uint8_t*>(v.data());
 std::copy(python_data, python_data + v.nbytes(), ort_data);

 return p;
}

// Creates a zero-copy numpy array that views the data owned by 'v'.
// The numpy array's lifetime is tied to the 'v' object.
nb::ndarray<> ToNumpyView(OgaPy::OgaTensor& v) {
 ::OgaTensor* v_ptr = v.get();
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

 // The owner is the Python object 'v' itself. nanobind will inc_ref it.
 return nb::ndarray<>(data, rank, shape.data(), nb::find(v), nullptr, ToDlpackDtype(static_cast<ONNXTensorElementDataType>(type_enum)));
}

// Creates a numpy array that takes ownership of a *newly created* OgaTensor
// via a nanobind capsule. Used for GetLogits, GetInput, GetOutput.
nb::ndarray<nb::numpy> ToNumpy(std::unique_ptr<OgaPy::OgaTensor> v) {
 ::OgaTensor* v_ptr = v->get();
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

 // The capsule owns the OgaPy::OgaTensor wrapper.
 // When the ndarray is GC'd, the capsule is destroyed,
 // which deletes the wrapper, which calls OgaDestroyTensor(v_ptr).
 nb::capsule owner(v.release(), [](void *p) noexcept { delete reinterpret_cast<OgaPy::OgaTensor*>(p); });

 return nb::ndarray<nb::numpy>(data, rank, shape.data(), owner, nullptr, ToDlpackDtype(static_cast<ONNXTensorElementDataType>(type_enum)));
}

// PyGeneratorParams and PyGenerator structs are no longer needed.
// We bind OgaPy::OgaGeneratorParams and OgaPy::OgaGenerator directly.

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
 static int unused{}; // The capsule needs something to reference
 nb::capsule cleanup(
   &unused, "cleanup", [](void*) noexcept {
    OgaShutdown();
   });
 m.attr("_cleanup") = cleanup;

  // Bind OgaPy::OgaGeneratorParams directly
 nb::class_<OgaPy::OgaGeneratorParams>(m, "GeneratorParams",
        nb::intrusive_ptr<OgaPy::OgaGeneratorParams>(
     [](OgaPy::OgaGeneratorParams *o, PyObject *po) noexcept { o->set_self_py(po); }))
   .def("__init__", [](OgaPy::OgaGeneratorParams* self, const OgaPy::OgaModel& model) {
     ::OgaGeneratorParams* p;
     OgaPy::OgaCheckResult(OgaCreateGeneratorParams(model.get(), &p)); // p has ext_ref=1
     new (self) OgaPy::OgaGeneratorParams(p);
          // 'self' wrapper now owns the ext_ref=1. When 'self' is GC'd,
          // its destructor will call OgaDestroyGeneratorParams(p), balancing the ref.
   })
   .def("try_graph_capture_with_max_batch_size", &OgaPy::OgaGeneratorParams::TryGraphCaptureWithMaxBatchSize)
   .def("set_search_options", [](OgaPy::OgaGeneratorParams& self, const nb::kwargs& dict) {
     for (const auto& entry : dict) {
      auto name = nb::cast<std::string>(entry.first);
      if (nb::isinstance<nb::float_>(entry.second))
        self.SetSearchNumber(name.c_str(), nb::cast<double>(entry.second));
      else if (nb::isinstance<nb::bool_>(entry.second))
        self.SetSearchBool(name.c_str(), nb::cast<bool>(entry.second));
      else if (nb::isinstance<nb::int_>(entry.second))
        self.SetSearchNumber(name.c_str(), nb::cast<int>(entry.second));
      else
        throw std::runtime_error("Unknown search option type, can be float/bool/int:" + name);
     }
   })
   .def("set_guidance", &OgaPy::OgaGeneratorParams::SetGuidance,
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
    ::OgaTensor* p;
    OgaPy::OgaCheckResult(OgaNamedTensorsGet(named_tensors.get(), name.c_str(), &p));
    if (!p)
     throw std::runtime_error("Tensor with name: " + name + " not found.");
    return new OgaPy::OgaTensor(p); // p has ext_ref=1, wrapper will release it
   })
   .def("__setitem__", [](OgaPy::OgaNamedTensors& named_tensors, const std::string& name, nb::ndarray<>& value) {
    // This is for: named_tensors["key"] = np.array(...)
    ::OgaTensor* p = CreateOgaTensorFromNdarray(value); // p has ext_ref=1
    OgaPy::OgaCheckResult(OgaNamedTensorsSet(named_tensors.get(), name.c_str(), p)); // map adds its own ref
    OgaDestroyTensor(p); // Release the ext_ref=1 from creation
   })
   .def("__setitem__", [](OgaPy::OgaNamedTensors& named_tensors, const std::string& name, OgaPy::OgaTensor& value) {
    // This is for: named_tensors["key"] = og.Tensor(...)
    OgaPy::OgaCheckResult(OgaNamedTensorsSet(named_tensors.get(), name.c_str(), value.get()));
   })
   .def("__contains__", [](const OgaPy::OgaNamedTensors& named_tensors, const std::string& name) {
    ::OgaTensor* p;
    OgaPy::OgaCheckResult(OgaNamedTensorsGet(const_cast<OgaNamedTensors*>(named_tensors.get()), name.c_str(), &p));
    bool found = (p != nullptr);
    if (found)
      OgaDestroyTensor(p); // Release the ref from OgaNamedTensorsGet
    return found;
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
    ::OgaStringArray* p;
    OgaPy::OgaCheckResult(OgaNamedTensorsGetNames(named_tensors.get(), &p));

    // Use unique_ptr to manage the wrapper's lifetime
    auto names = std::unique_ptr<OgaPy::OgaStringArray>(new OgaPy::OgaStringArray(p));

    size_t count;
    OgaPy::OgaCheckResult(OgaStringArrayGetCount(names->get(), &count));
    keys.reserve(count);
    for (size_t i = 0; i < count; i++) {
     const char* str;
     OgaPy::OgaCheckResult(OgaStringArrayGetString(names->get(), i, &str));
     keys.push_back(str);
    }
    return keys;
   });

 nb::class_<OgaPy::OgaTensor>(
   m, "Tensor",
   nb::intrusive_ptr<OgaPy::OgaTensor>(
     [](OgaPy::OgaTensor *o, PyObject *po) noexcept { o->set_self_py(po); }))
   .def("__init__", [](OgaPy::OgaTensor* self, nb::ndarray<>& v) {
    // This is for: my_tensor = og.Tensor(np.array(...))
    ::OgaTensor* p = CreateOgaTensorFromNdarray(v); // p has ext_ref=1
    new (self) OgaPy::OgaTensor(p);
    // 'self' wrapper now owns the ext_ref=1
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
   .def("as_numpy", &ToNumpyView); // <-- FIXED

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
    auto view = t.GetEosTokenIds();
    size_t shape[1] = {view->size()};
    return nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>>(
      view->data(), 1, shape, nb::find(view));
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
    ::OgaSequences* sequences_ptr;
    OgaPy::OgaCheckResult(OgaCreateSequences(&sequences_ptr));
    auto sequences = new OgaPy::OgaSequences(sequences_ptr);
    OgaPy::intrusive_inc_ref(sequences);
    OgaPy::OgaCheckResult(OgaTokenizerEncode(t.get(), s.c_str(), sequences->get()));
    auto view = sequences->GetSequenceData(0);
    size_t shape[1] = {view->size()};
    nb::capsule owner(view, [](void* p) noexcept {
     delete reinterpret_cast<OgaPy::SequenceDataView*>(p);
    });
    OgaPy::intrusive_dec_ref(sequences);
    return nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>>(
      view->data(), 1, shape, owner);
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
   .def("apply_chat_template", [](const OgaPy::OgaTokenizer& t, const char* messages, std::optional<std::string> template_str, std::optional<std::string> tools, bool add_generation_prompt) -> std::string {
    const char *p;
    const char* template_ptr = template_str.has_value() ? template_str->c_str() : nullptr;
    const char* tools_ptr = tools.has_value() ? tools->c_str() : nullptr;
    OgaPy::OgaCheckResult(OgaTokenizerApplyChatTemplate(t.get(), template_ptr, messages, tools_ptr, add_generation_prompt, &p));
    return std::string(OgaPy::OgaString(p));
   }, "messages"_a, nb::kw_only(), "template_str"_a = nb::none(), "tools"_a = nb::none(), "add_generation_prompt"_a = true)
   .def("encode_batch", [](const OgaPy::OgaTokenizer& t, std::vector<std::string> strings) {
    std::vector<const char*> c_strings;
    c_strings.reserve(strings.size());
    for (const auto& s : strings)
     c_strings.push_back(s.c_str());
    ::OgaTensor* p;
    OgaPy::OgaCheckResult(OgaTokenizerEncodeBatch(t.get(), c_strings.data(), c_strings.size(), &p));
    return new OgaPy::OgaTensor(p); // p has ext_ref=1, wrapper will release
   })
   .def("decode_batch", [](const OgaPy::OgaTokenizer& t, const OgaPy::OgaTensor& tokens) {
    std::vector<std::string> strings;
    ::OgaStringArray* p;
    OgaPy::OgaCheckResult(OgaTokenizerDecodeBatch(t.get(), tokens.get(), &p));
    auto decoded = std::unique_ptr<OgaPy::OgaStringArray>(new OgaPy::OgaStringArray(p));
    size_t count;
    OgaPy::OgaCheckResult(OgaStringArrayGetCount(decoded->get(), &count));
    strings.reserve(count);
    for (size_t i = 0; i < count; i++) {
     const char* str;
     OgaPy::OgaCheckResult(OgaStringArrayGetString(decoded->get(), i, &str));
     strings.push_back(str);
    }
    return strings;
   })
   .def("create_stream", [](const OgaPy::OgaTokenizer& t) {
    ::OgaTokenizerStream* p;
    OgaPy::OgaCheckResult(OgaCreateTokenizerStream(t.get(), &p));
    return new OgaPy::OgaTokenizerStream(p); // Wrapper owns new'd object
   });

 nb::class_<OgaPy::OgaConfig>(
   m, "Config",
   nb::intrusive_ptr<OgaPy::OgaConfig>(
     [](OgaPy::OgaConfig *o, PyObject *po) noexcept { o->set_self_py(po); }))
   .def("__init__", [](OgaPy::OgaConfig* self, const std::string& config_path) {
    ::OgaConfig* p;
    OgaPy::OgaCheckResult(OgaCreateConfig(config_path.c_str(), &p));
    new (self) OgaPy::OgaConfig(p); // Wrapper owns new'd object
   })
   .def("append_provider", [](OgaPy::OgaConfig& config, const char* provider) { OgaPy::OgaCheckResult(OgaConfigAppendProvider(config.get(), provider));})
   .def("set_provider_option", [](OgaPy::OgaConfig& config, const char* provider, const char* name, const char* value) { OgaPy::OgaCheckResult(OgaConfigSetProviderOption(config.get(), provider, name, value));})
   .def("clear_providers", [](OgaPy::OgaConfig& config) { OgaPy::OgaCheckResult(OgaConfigClearProviders(config.get()));})
   .def("add_model_data", [](OgaPy::OgaConfig& config, const std::string& model_filename, nb::bytes model_bytes) {
     OgaPy::OgaCheckResult(OgaConfigAddModelData(config.get(), model_filename.c_str(), model_bytes.data(), model_bytes.size()));
   })
   .def("add_model_data", [](OgaPy::OgaConfig& config, const std::string& model_filename, nb::ndarray<nb::ro, uint8_t, nb::ndim<1>> array) {
     OgaPy::OgaCheckResult(OgaConfigAddModelData(config.get(), model_filename.c_str(), array.data(), array.nbytes()));
   })
   .def("add_model_data", [](OgaPy::OgaConfig& config, const std::string& model_filename, nb::ndarray<nb::memview, nb::ro, uint8_t, nb::ndim<1>> array) {
     OgaPy::OgaCheckResult(OgaConfigAddModelData(config.get(), model_filename.c_str(), array.data(), array.nbytes()));
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
    OgaPy::OgaCheckResult(OgaCreateModelFromConfig(config.get(), &p)); // p has ext_ref=1
    new (self) OgaPy::OgaModel(p); // wrapper will release ref
   })
   .def("__init__", [](OgaPy::OgaModel* self, const std::string& config_path) {
    ::OgaModel* p;
    OgaPy::OgaCheckResult(OgaCreateModel(config_path.c_str(), &p)); // p has ext_ref=1
    new (self) OgaPy::OgaModel(p); // wrapper will release ref
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
    ::OgaMultiModalProcessor* p;
    OgaPy::OgaCheckResult(OgaCreateMultiModalProcessor(model.get(), &p)); // p has ext_ref=1
    return new OgaPy::OgaMultiModalProcessor(p); // wrapper will release ref
   });

  // Bind OgaPy::OgaGenerator directly
 nb::class_<OgaPy::OgaGenerator>(m, "Generator",
        nb::intrusive_ptr<OgaPy::OgaGenerator>(
     [](OgaPy::OgaGenerator *o, PyObject *po) noexcept { o->set_self_py(po); }))
   .def("__init__", [](OgaPy::OgaGenerator* self, const OgaPy::OgaModel& model, OgaPy::OgaGeneratorParams& params) {
     ::OgaGenerator* p;
     // OgaCreateGenerator returns a raw 'new' pointer, not ref-counted
     OgaPy::OgaCheckResult(OgaCreateGenerator(model.get(), params.get(), &p));
     new (self) OgaPy::OgaGenerator(p); // wrapper destructor will delete p
   })
   .def("is_done", &OgaPy::OgaGenerator::IsDone)
   .def("get_input", [](OgaPy::OgaGenerator& self, const std::string& name) -> nb::ndarray<nb::numpy> {
     ::OgaTensor* p = self.GetInput(name.c_str()); // p has ext_ref=1
     // ToNumpy takes ownership of the wrapper, which balances the ref
     return ToNumpy(std::unique_ptr<OgaPy::OgaTensor>(new OgaPy::OgaTensor(p)));
   })
   .def("get_output", [](OgaPy::OgaGenerator& self, const std::string& name) -> nb::ndarray<nb::numpy> {
     ::OgaTensor* p = self.GetOutput(name.c_str()); // p has ext_ref=1
     return ToNumpy(std::unique_ptr<OgaPy::OgaTensor>(new OgaPy::OgaTensor(p)));
   })
   .def("set_inputs", &OgaPy::OgaGenerator::SetInputs)
   .def("set_model_input", [](OgaPy::OgaGenerator& self, const std::string& name, OgaPy::OgaTensor& value) {
     self.SetModelInput(name.c_str(), value.get());
   })
   .def("set_model_input", [](OgaPy::OgaGenerator& self, const std::string& name, nb::ndarray<>& value) {
     ::OgaTensor* p = CreateOgaTensorFromNdarray(value); // p has ext_ref=1
     self.SetModelInput(name.c_str(), p); // C-API stores its own shared_ptr
     OgaDestroyTensor(p); // Release the ext_ref=1 from creation
   })
   .def("append_tokens", [](OgaPy::OgaGenerator& self, OgaPy::OgaTensor& tokens) {
     auto span = ToSpan<const int32_t>(*tokens.get());
     self.AppendTokens(span.data(), span.size());
   })
   .def("append_tokens", [](OgaPy::OgaGenerator& self, nb::ndarray<const int32_t, nb::c_contig> tokens) {
     if (tokens.ndim() == 0) {
       throw std::runtime_error("Input array cannot be scalar.");
     }
     self.AppendTokens(tokens.data(), tokens.size());
   })
   .def("get_logits", [](OgaPy::OgaGenerator& self) -> nb::ndarray<nb::numpy> {
     ::OgaTensor* p = self.GetLogits(); // p has ext_ref=1
     return ToNumpy(std::unique_ptr<OgaPy::OgaTensor>(new OgaPy::OgaTensor(p)));
   })
   .def("set_logits", [](OgaPy::OgaGenerator& self, OgaPy::OgaTensor& value) {
     self.SetLogits(value.get());
   })
   .def("set_logits", [](OgaPy::OgaGenerator& self, nb::ndarray<>& value) {
     ::OgaTensor* p = CreateOgaTensorFromNdarray(value); // p has ext_ref=1
     self.SetLogits(p); // C-API copies the data
     OgaDestroyTensor(p); // Release the ext_ref=1
   })
   .def("generate_next_token", &OgaPy::OgaGenerator::GenerateNextToken)
   .def("rewind_to", &OgaPy::OgaGenerator::RewindTo)
   .def("get_next_tokens", [](OgaPy::OgaGenerator& self) {
     auto view = self.GetNextTokens(); // Returns NextTokensView*
     size_t shape[1] = {view->size()};
     return nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>>(
       view->data(), 1, shape, nb::find(view));
   })
   .def("get_sequence", [](OgaPy::OgaGenerator& self, int index) {
     auto view = self.GetSequenceData(index); // Returns GeneratorSequenceDataView*
     size_t shape[1] = {view->size()};
     return nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>>(
       view->data(), 1, shape, nb::find(view));
   })
   .def("set_active_adapter", &OgaPy::OgaGenerator::SetActiveAdapter);

 nb::class_<OgaPy::OgaImages>(
   m, "Images",
   nb::intrusive_ptr<OgaPy::OgaImages>(
     [](OgaPy::OgaImages *o, PyObject *po) noexcept { o->set_self_py(po); }))
   .def_static("open", [](nb::args image_paths) {
    std::vector<std::string> image_paths_string;
    std::vector<const char*> image_paths_vector;
    image_paths_vector.reserve(image_paths.size());
    image_paths_string.reserve(image_paths.size());
    for (auto image_path : image_paths) {
     if (!nb::isinstance<nb::str>(image_path))
      throw std::runtime_error("Image paths must be strings.");
     image_paths_string.push_back(nb::cast<std::string>(image_path));
     image_paths_vector.push_back(image_paths_string.back().c_str());
    }
    ::OgaStringArray* p_strs;
    OgaPy::OgaCheckResult(OgaCreateStringArrayFromStrings(image_paths_vector.data(), image_paths_vector.size(), &p_strs));
    auto strs = std::unique_ptr<OgaPy::OgaStringArray>(new OgaPy::OgaStringArray(p_strs));
    ::OgaImages* p;
    OgaPy::OgaCheckResult(OgaLoadImages(strs->get(), &p));
    return new OgaPy::OgaImages(p); // wrapper owns new'd object
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
    ::OgaImages* p;
    OgaPy::OgaCheckResult(OgaLoadImagesFromBuffers(image_raw_data.data(), image_sizes.data(), image_raw_data.size(), &p));
    return new OgaPy::OgaImages(p); // wrapper owns new'd object
   });

 nb::class_<OgaPy::OgaAudios>(
   m, "Audios",
   nb::intrusive_ptr<OgaPy::OgaAudios>(
     [](OgaPy::OgaAudios *o, PyObject *po) noexcept { o->set_self_py(po); }))
   .def_static("open", [](nb::args audio_paths) {
    std::vector<std::string> audio_paths_string;
    std::vector<const char*> audio_paths_vector;
    audio_paths_string.reserve(audio_paths.size());
    audio_paths_vector.reserve(audio_paths.size());

    for (const auto& audio_path : audio_paths) {
     if (!nb::isinstance<nb::str>(audio_path))
      throw std::runtime_error("Audio paths must be strings.");
     audio_paths_string.push_back(nb::cast<std::string>(audio_path));
     audio_paths_vector.push_back(audio_paths_string.back().c_str());
    }
    ::OgaStringArray* p_strs;
    OgaPy::OgaCheckResult(OgaCreateStringArrayFromStrings(audio_paths_vector.data(), audio_paths_vector.size(), &p_strs));
    auto strs = std::unique_ptr<OgaPy::OgaStringArray>(new OgaPy::OgaStringArray(p_strs));
    ::OgaAudios* p;
    OgaPy::OgaCheckResult(OgaLoadAudios(strs->get(), &p));
    return new OgaPy::OgaAudios(p); // wrapper owns new'd object
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

    ::OgaAudios*p;
    OgaPy::OgaCheckResult(OgaLoadAudiosFromBuffers(audio_raw_data.data(), audio_sizes.data(), audio_raw_data.size(), &p));
    return new OgaPy::OgaAudios(p); // wrapper owns new'd object
   });

 nb::class_<OgaPy::OgaMultiModalProcessor>(
   m, "MultiModalProcessor",
   nb::intrusive_ptr<OgaPy::OgaMultiModalProcessor>(
     [](OgaPy::OgaMultiModalProcessor *o, PyObject *po) noexcept { o->set_self_py(po); }))
   .def("__call__", [](OgaPy::OgaMultiModalProcessor& processor, nb::object prompts, const nb::kwargs& kwargs) {
      OgaImages* images{};
      OgaAudios* audios{};
      if (kwargs.contains("images")) {
       images = nb::cast<OgaPy::OgaImages*>(kwargs["images"])->get();
      }
      if (kwargs.contains("audios")) {
       audios = nb::cast<OgaPy::OgaAudios*>(kwargs["audios"])->get();
      }

      std::vector<std::string> prompts_str;
      std::vector<const char*> c_prompts;
      if (nb::isinstance<nb::str>(prompts)) {
       // One prompt
       ::OgaNamedTensors* p;
       OgaPy::OgaCheckResult(OgaProcessorProcessImagesAndAudios(processor.get(), nb::cast<std::string>(prompts).c_str(), images, audios, &p));
       return new OgaPy::OgaNamedTensors(p);
      } else if (nb::isinstance<nb::list>(prompts)) {
       // Multiple prompts
       prompts_str.reserve(nb::len(prompts));
       c_prompts.reserve(nb::len(prompts));
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

      ::OgaStringArray* p_strs;
      OgaPy::OgaCheckResult(OgaCreateStringArrayFromStrings(c_prompts.data(), c_prompts.size(), &p_strs));
      auto strs = std::unique_ptr<OgaPy::OgaStringArray>(new OgaPy::OgaStringArray(p_strs));
      ::OgaNamedTensors* p;
      OgaPy::OgaCheckResult(OgaProcessorProcessImagesAndAudiosAndPrompts(processor.get(), strs->get(), images, audios, &p));
      return new OgaPy::OgaNamedTensors(p);
     }, "prompts"_a = nb::none(), nb::arg())
   .def("create_stream", [](OgaPy::OgaMultiModalProcessor& processor) {
    ::OgaTokenizerStream* p;
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
     [](OgaPy::OgaRequest* self, OgaPy::OgaGeneratorParams& params) {
      ::OgaRequest* p;
      OgaPy::OgaCheckResult(OgaCreateRequest(params.get(), &p));
      new (self) OgaPy::OgaRequest(p);
     })
   .def("add_tokens", [](OgaPy::OgaRequest& request, nb::ndarray<const int32_t, nb::ndim<1>> tokens) {
    ::OgaSequences* p;
    OgaPy::OgaCheckResult(OgaCreateSequences(&p));
    auto sequences = std::unique_ptr<OgaPy::OgaSequences>(new OgaPy::OgaSequences(p));
    auto tokens_span = ToSpan(tokens);
    OgaPy::OgaCheckResult(OgaAppendTokenSequence(tokens_span.data(), tokens_span.size(), sequences->get()));
    OgaPy::OgaCheckResult(OgaRequestAddTokens(request.get(), sequences->get()));
   })
   .def("has_unseen_tokens", [](const OgaPy::OgaRequest& request) {
    return request.HasUnseenTokens();
   })
   .def("is_done", [](const OgaPy::OgaRequest& request) {
    return request.IsDone();
   })
   .def("get_unseen_token", [](OgaPy::OgaRequest& request) {
    return request.GetUnseenToken();
   })
   .def("set_opaque_data", [](OgaPy::OgaRequest& request, nb::object opaque_data) {
    request.SetOpaqueData(opaque_data.ptr());
   })
   .def("get_opaque_data", [](OgaPy::OgaRequest& request) -> nb::object {
    void* data = request.GetOpaqueData();
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
   .def("add_request", [](OgaPy::OgaEngine& engine, OgaPy::OgaRequest& request){ engine.AddRequest(&request); })
   .def("step", [](OgaPy::OgaEngine& engine) {
    return engine.Step();
   })
   .def("remove_request", [](OgaPy::OgaEngine& engine, OgaPy::OgaRequest& request){ engine.RemoveRequest(&request); })
   .def("has_pending_requests", [](OgaPy::OgaEngine& engine) {
    return engine.HasPendingRequests();
   });

 // Note: BorrowedArrayView classes (SequenceDataView, etc.) are internal wrappers
 // and not exposed to Python. They're used internally to manage lifetimes, and
 // Python always receives numpy ndarrays.


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
