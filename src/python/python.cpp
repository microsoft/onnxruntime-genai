// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#define OGA_USE_SPAN 1
#include "../models/onnxruntime_api.h"
#include "../ort_genai.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>

namespace nb = nanobind;
using namespace nb::literals;

// Cross-reference each Python binding to the C API entry point that implements it
// (declared, with parameter docs, in src/ort_genai_c.h). These strings become the
// binding docstrings, so they show up in hover tooltips and the generated .pyi and
// make the C/C++ implementation one "Go to Symbol in Workspace" (Ctrl+T) away.
//
// This mapping is hand-written: nanobind cannot derive it, since a binding is just a
// lambda/function pointer from its point of view. To keep it from rotting as new
// bindings are added, tools/python/check_python_bindings_capi.py fails if any binding
// lacks one of these annotations (run as part of the Python test suite).
//
//   OGA_CAPI("OgaFoo")            -> a method/function backed by C API symbol OgaFoo
//   OGA_CLASS_CAPI("Generators::Foo") -> a class, noting the core C++ type it wraps
//   OGA_NO_CAPI("reason")        -> a binding with no C API equivalent (e.g. build flags)
#define OGA_CAPI(sym) "C API: " sym " (src/ort_genai_c.h)"
#define OGA_CLASS_CAPI(core) "C API: src/ort_genai_c.h (" core ")"
#define OGA_NO_CAPI(reason) reason


// Map a numpy/dlpack dtype onto the corresponding ONNX tensor element type.
ONNXTensorElementDataType ToTensorType(const nb::dlpack::dtype& type) {
  using code = nb::dlpack::dtype_code;
  const auto c = static_cast<code>(type.code);
  const auto bits = type.bits;
  switch (c) {
    case code::Bool:
      return Ort::TypeToTensorType<bool>;
    case code::Int:
      switch (bits) {
        case 8:
          return Ort::TypeToTensorType<int8_t>;
        case 16:
          return Ort::TypeToTensorType<int16_t>;
        case 32:
          return Ort::TypeToTensorType<int32_t>;
        case 64:
          return Ort::TypeToTensorType<int64_t>;
      }
      break;
    case code::UInt:
      switch (bits) {
        case 8:
          return Ort::TypeToTensorType<uint8_t>;
        case 16:
          return Ort::TypeToTensorType<uint16_t>;
        case 32:
          return Ort::TypeToTensorType<uint32_t>;
        case 64:
          return Ort::TypeToTensorType<uint64_t>;
      }
      break;
    case code::Float:
      switch (bits) {
        case 16:
          return Ort::TypeToTensorType<Ort::Float16_t>;
        case 32:
          return Ort::TypeToTensorType<float>;
        case 64:
          return Ort::TypeToTensorType<double>;
      }
      break;
    default:
      break;
  }
  throw std::runtime_error("Unsupported numpy type");
}

// Map an ONNX tensor element type onto the corresponding numpy/dlpack dtype.
nb::dlpack::dtype ToDlpackType(ONNXTensorElementDataType type) {
  using code = nb::dlpack::dtype_code;
  auto make = [](code c, uint8_t bits) {
    return nb::dlpack::dtype{static_cast<uint8_t>(c), bits, 1};
  };
  switch (type) {
    case Ort::TypeToTensorType<bool>:
      return make(code::Bool, 8);
    case Ort::TypeToTensorType<int8_t>:
      return make(code::Int, 8);
    case Ort::TypeToTensorType<uint8_t>:
      return make(code::UInt, 8);
    case Ort::TypeToTensorType<int16_t>:
      return make(code::Int, 16);
    case Ort::TypeToTensorType<uint16_t>:
      return make(code::UInt, 16);
    case Ort::TypeToTensorType<int32_t>:
      return make(code::Int, 32);
    case Ort::TypeToTensorType<uint32_t>:
      return make(code::UInt, 32);
    case Ort::TypeToTensorType<int64_t>:
      return make(code::Int, 64);
    case Ort::TypeToTensorType<uint64_t>:
      return make(code::UInt, 64);
    case Ort::TypeToTensorType<Ort::Float16_t>:
      return make(code::Float, 16);
    case Ort::TypeToTensorType<float>:
      return make(code::Float, 32);
    case Ort::TypeToTensorType<double>:
      return make(code::Float, 64);
    default:
      throw std::runtime_error("Unsupported onnx type");
  }
}

template <typename T>
std::span<T> ToSpan(nb::ndarray<T, nb::c_contig>& v) {
  return {v.data(), v.size()};
}

template <typename T>
std::span<T> ToSpan(OgaTensor& v) {
  assert(static_cast<ONNXTensorElementDataType>(v.Type()) == Ort::TypeToTensorType<T>);
  auto shape = v.Shape();
  auto element_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  return {reinterpret_cast<T*>(v.Data()), static_cast<size_t>(element_count)};
}

// Copy a contiguous span into a freshly allocated, numpy-owned 1-D array.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> ToPython(std::span<T> v) {
  using U = std::remove_const_t<T>;
  auto* buffer = new U[v.size()];
  std::memcpy(buffer, v.data(), v.size() * sizeof(T));
  nb::capsule owner(buffer, [](void* p) noexcept { delete[] reinterpret_cast<U*>(p); });
  size_t shape[1] = {v.size()};
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(buffer, 1, shape, owner, nullptr,
                                                nb::dtype<U>(), nb::device::cpu::value, 0);
}

// Whether a numpy array is either C- or F-contiguous. Strides are in elements.
bool IsContiguous(const nb::ndarray<>& v) {
  size_t ndim = v.ndim();
  if (ndim == 0)
    return true;
  {  // C-contiguous
    int64_t expected = 1;
    bool ok = true;
    for (size_t i = ndim; i-- > 0;) {
      if (v.shape(i) != 1 && v.stride(i) != expected) {
        ok = false;
        break;
      }
      expected *= static_cast<int64_t>(v.shape(i));
    }
    if (ok)
      return true;
  }
  {  // F-contiguous
    int64_t expected = 1;
    bool ok = true;
    for (size_t i = 0; i < ndim; ++i) {
      if (v.shape(i) != 1 && v.stride(i) != expected) {
        ok = false;
        break;
      }
      expected *= static_cast<int64_t>(v.shape(i));
    }
    if (ok)
      return true;
  }
  return false;
}

std::unique_ptr<OgaTensor> ToOgaTensor(const nb::ndarray<>& v, bool copy = true) {
  auto type = ToTensorType(v.dtype());

  std::vector<int64_t> shape(v.ndim());
  for (size_t i = 0; i < v.ndim(); i++)
    shape[i] = static_cast<int64_t>(v.shape(i));

  if (!IsContiguous(v))
    throw std::runtime_error("Array must be contiguous. Please use NumPy's 'ascontiguousarray' method on the value.");

  auto tensor = OgaTensor::Create(copy ? nullptr : v.data(), shape, static_cast<OgaElementType>(type));
  if (copy) {
    auto ort_data = reinterpret_cast<uint8_t*>(tensor->Data());
    auto python_data = reinterpret_cast<const uint8_t*>(v.data());
    std::copy(python_data, python_data + v.nbytes(), ort_data);
  }
  return tensor;
}

// Copy an OgaTensor into a freshly allocated, numpy-owned array.
nb::ndarray<nb::numpy> ToNumpy(OgaTensor& v) {
  auto shape_vec = v.Shape();
  auto type = static_cast<ONNXTensorElementDataType>(v.Type());
  auto element_size = Ort::SizeOf(type);

  std::vector<size_t> shape(shape_vec.size());
  size_t element_count = 1;
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    shape[i] = static_cast<size_t>(shape_vec[i]);
    element_count *= shape[i];
  }

  size_t nbytes = element_count * element_size;
  auto* buffer = new uint8_t[nbytes];
  std::memcpy(buffer, v.Data(), nbytes);
  nb::capsule owner(buffer, [](void* p) noexcept { delete[] reinterpret_cast<uint8_t*>(p); });

  return nb::ndarray<nb::numpy>(buffer, shape.size(), shape.data(), owner, nullptr,
                                ToDlpackType(type), nb::device::cpu::value, 0);
}

struct PyGeneratorParams {
  PyGeneratorParams(const OgaModel& model) : params_{OgaGeneratorParams::Create(model)} {}

  operator const OgaGeneratorParams&() const { return *params_; }

  std::unique_ptr<OgaGeneratorParams> params_;

  void SetSearchOptions(const nb::kwargs& dict) {
    for (auto item : dict) {
      auto name = nb::cast<std::string>(item.first);
      if (nb::isinstance<nb::float_>(item.second)) {
        params_->SetSearchOption(name.c_str(), nb::cast<double>(item.second));
      } else if (nb::isinstance<nb::bool_>(item.second)) {
        params_->SetSearchOptionBool(name.c_str(), nb::cast<bool>(item.second));
      } else if (nb::isinstance<nb::int_>(item.second)) {
        params_->SetSearchOption(name.c_str(), nb::cast<int>(item.second));
      } else
        throw std::runtime_error("Unknown search option type, can be float/bool/int:" + name);
    }
  }

  void SetGuidance(const std::string& type, const std::string& data, bool enable_ff_tokens = false) {
    params_->SetGuidance(type.c_str(), data.c_str(), enable_ff_tokens);
  }

  nb::dict GetSearchOptions() {
    nb::dict d;
    d["batch_size"] = params_->GetSearchNumber("batch_size");
    d["chunk_size"] = params_->GetSearchNumber("chunk_size");
    d["diversity_penalty"] = params_->GetSearchNumber("diversity_penalty");
    d["do_sample"] = params_->GetSearchBool("do_sample");
    d["early_stopping"] = params_->GetSearchBool("early_stopping");
    d["length_penalty"] = params_->GetSearchNumber("length_penalty");
    d["max_length"] = params_->GetSearchNumber("max_length");
    d["min_length"] = params_->GetSearchNumber("min_length");
    d["no_repeat_ngram_size"] = params_->GetSearchNumber("no_repeat_ngram_size");
    d["num_beams"] = params_->GetSearchNumber("num_beams");
    d["num_return_sequences"] = params_->GetSearchNumber("num_return_sequences");
    d["past_present_share_buffer"] = params_->GetSearchBool("past_present_share_buffer");
    d["random_seed"] = params_->GetSearchNumber("random_seed");
    d["repetition_penalty"] = params_->GetSearchNumber("repetition_penalty");
    d["temperature"] = params_->GetSearchNumber("temperature");
    d["top_k"] = params_->GetSearchNumber("top_k");
    d["top_p"] = params_->GetSearchNumber("top_p");
    return d;
  }

  std::vector<nb::object> refs_;  // References to data we want to ensure doesn't get garbage collected
};

struct PyGenerator {
  PyGenerator(const OgaModel& model, PyGeneratorParams& params) {
    generator_ = OgaGenerator::Create(model, *params.params_);
  }

  nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>> GetNextTokens() {
    return ToPython(generator_->GetNextTokens());
  }

  nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>> GetSequence(int index) {
    return ToPython(generator_->GetSequence(index));
  }

  nb::ndarray<nb::numpy> GetInput(const std::string& name) {
    return ToNumpy(*generator_->GetInput(name.c_str()));
  }

  nb::ndarray<nb::numpy> GetOutput(const std::string& name) {
    return ToNumpy(*generator_->GetOutput(name.c_str()));
  }

  void SetModelInput(const std::string& name, const nb::ndarray<>& value) {
    generator_->SetModelInput(name.c_str(), *ToOgaTensor(value, false));
  }

  void SetInputs(OgaNamedTensors& named_tensors) {
    generator_->SetInputs(named_tensors);
  }

  void AppendTokens(OgaTensor& tokens) {
    generator_->AppendTokens(ToSpan<int32_t>(tokens));
  }

  void AppendTokens(nb::ndarray<int32_t, nb::c_contig> tokens) {
    generator_->AppendTokens(ToSpan(tokens));
  }

  size_t TokenCount() const {
    return generator_->TokenCount();
  }

  nb::ndarray<nb::numpy> GetLogits() {
    return ToNumpy(*generator_->GetLogits());
  }

  void SetLogits(const nb::ndarray<>& new_logits) {
    generator_->SetLogits(*ToOgaTensor(new_logits, false));
  }

  void GenerateNextToken() {
    generator_->GenerateNextToken();
  }

  void RewindTo(size_t new_length) {
    generator_->RewindTo(new_length);
  }

  bool IsDone() {
    return generator_->IsDone();
  }

  void SetActiveAdapter(OgaAdapters& adapters, const std::string& adapter_name) {
    generator_->SetActiveAdapter(adapters, adapter_name.c_str());
  }

  void SetRuntimeOption(const std::string& key, const std::string& value) {
    generator_->SetRuntimeOption(key.c_str(), value.c_str());
  }

 private:
  std::unique_ptr<OgaGenerator> generator_;
};

void SetLogOptions(const nb::kwargs& dict) {
  for (auto item : dict) {
    auto name = nb::cast<std::string>(item.first);
    if (nb::isinstance<nb::bool_>(item.second)) {
      Oga::SetLogBool(name.c_str(), nb::cast<bool>(item.second));
    } else if (nb::isinstance<nb::str>(item.second)) {
      Oga::SetLogString(name.c_str(), nb::cast<std::string>(item.second).c_str());
    } else
      throw std::runtime_error("Unknown log option type, can be bool/string:" + name);
  }
}

void SetLogCallback(std::optional<nb::callable> callback) {
  static std::optional<nb::callable> log_callback;
  log_callback = std::move(callback);

  if (log_callback.has_value()) {
    Oga::SetLogCallback([](const char* message, size_t length) {
      nb::gil_scoped_acquire gil;
      (*log_callback)(nb::str(message, length));
    });
  } else {
    Oga::SetLogCallback(nullptr);
  }
}

NB_MODULE(onnxruntime_genai, m) {
  m.doc() = R"pbdoc(
        Ort Generators library
        ----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  // Add a cleanup call to happen before global variables are destroyed
  static int unused{};  // The capsule needs something to reference
  nb::capsule cleanup(&unused, "cleanup", [](void*) noexcept { OgaShutdown(); });
  m.attr("_cleanup") = cleanup;

  nb::class_<PyGeneratorParams>(m, "GeneratorParams", OGA_CLASS_CAPI("Generators::GeneratorParams"))
      .def(nb::init<const OgaModel&>(), OGA_CAPI("OgaCreateGeneratorParams"))
      .def("set_search_options", &PyGeneratorParams::SetSearchOptions, OGA_CAPI("OgaGeneratorParamsSetSearchNumber / OgaGeneratorParamsSetSearchBool"))  // See config.h 'struct Search' for the options
      .def("set_guidance", &PyGeneratorParams::SetGuidance,
           "type"_a, "data"_a, "enable_ff_tokens"_a = false, OGA_CAPI("OgaGeneratorParamsSetGuidance"))
      .def("get_search_options", &PyGeneratorParams::GetSearchOptions, OGA_CAPI("OgaGeneratorParamsGetSearchNumber / OgaGeneratorParamsGetSearchBool"));

  nb::class_<OgaTokenizerStream>(m, "TokenizerStream", OGA_CLASS_CAPI("Generators::TokenizerStream"))
      .def("decode", [](OgaTokenizerStream& t, int32_t token) { return t.Decode(token); }, OGA_CAPI("OgaTokenizerStreamDecode"));

  nb::class_<OgaNamedTensors>(m, "NamedTensors", OGA_CLASS_CAPI("Generators::NamedTensors"))
      .def(nb::new_([]() { return OgaNamedTensors::Create(); }), OGA_CAPI("OgaCreateNamedTensors"))
      .def("__getitem__", [](OgaNamedTensors& named_tensors, const std::string& name) {
        auto tensor = named_tensors.Get(name.c_str());
        if (!tensor)
          throw std::runtime_error("Tensor with name: " + name + " not found.");
        return tensor;
      }, OGA_CAPI("OgaNamedTensorsGet"))
      .def("__setitem__", [](OgaNamedTensors& named_tensors, const std::string& name, const nb::ndarray<>& value) {
        named_tensors.Set(name.c_str(), *ToOgaTensor(value));
      }, OGA_CAPI("OgaNamedTensorsSet"))
      .def("__setitem__", [](OgaNamedTensors& named_tensors, const std::string& name, OgaTensor& value) {
        named_tensors.Set(name.c_str(), value);
      }, OGA_CAPI("OgaNamedTensorsSet"))
      .def("__contains__", [](const OgaNamedTensors& named_tensors, const std::string& name) {
        return const_cast<OgaNamedTensors&>(named_tensors).Get(name.c_str()) != nullptr;
      }, OGA_CAPI("OgaNamedTensorsGet"))
      .def("__delitem__", [](OgaNamedTensors& named_tensors, const std::string& name) {
        named_tensors.Delete(name.c_str());
      }, OGA_CAPI("OgaNamedTensorsDelete"))
      .def("__len__", &OgaNamedTensors::Count, OGA_CAPI("OgaNamedTensorsCount"))
      .def("keys", [](OgaNamedTensors& named_tensors) {
        std::vector<std::string> keys;
        auto names = named_tensors.GetNames();
        for (size_t i = 0; i < names->Count(); i++)
          keys.push_back(names->Get(i));
        return keys;
      }, OGA_CAPI("OgaNamedTensorsGetNames"));

  nb::class_<OgaTensor>(m, "Tensor", OGA_CLASS_CAPI("Generators::Tensor"))
      .def(nb::new_([](const nb::ndarray<>& v) { return ToOgaTensor(v); }), OGA_CAPI("OgaCreateTensorFromBuffer"))
      .def("shape", &OgaTensor::Shape, OGA_CAPI("OgaTensorGetShape"))
      .def("type", [](OgaTensor& t) { return static_cast<int>(t.Type()); }, OGA_CAPI("OgaTensorGetType"))
      .def("data", [](OgaTensor& t) { return reinterpret_cast<uintptr_t>(t.Data()); }, OGA_CAPI("OgaTensorGetData"))
      .def("as_numpy", [](OgaTensor& t) { return ToNumpy(t); }, OGA_CAPI("OgaTensorGetData"));

  nb::class_<OgaTokenizer>(m, "Tokenizer", OGA_CLASS_CAPI("Generators::Tokenizer"))
      .def(nb::new_([](const OgaModel& model) { return OgaTokenizer::Create(model); }), OGA_CAPI("OgaCreateTokenizer"))
      .def_prop_ro("bos_token_id", &OgaTokenizer::GetBosTokenId, OGA_CAPI("OgaTokenizerGetBosTokenId"))
      .def_prop_ro("eos_token_ids", [](const OgaTokenizer& t) {
        return ToPython(t.GetEosTokenIds());
      }, OGA_CAPI("OgaTokenizerGetEosTokenIds"))
      .def_prop_ro("pad_token_id", &OgaTokenizer::GetPadTokenId, OGA_CAPI("OgaTokenizerGetPadTokenId"))
      .def("update_options", [](OgaTokenizer& t, nb::kwargs kwargs) {
        std::vector<std::string> key_storage;
        std::vector<std::string> value_storage;
        key_storage.reserve(kwargs.size());
        value_storage.reserve(kwargs.size());

        std::vector<const char*> keys;
        std::vector<const char*> values;
        keys.reserve(kwargs.size());
        values.reserve(kwargs.size());

        for (auto item : kwargs) {
          key_storage.emplace_back(nb::cast<std::string>(nb::str(item.first)));
          value_storage.emplace_back(nb::cast<std::string>(nb::str(item.second)));
          keys.push_back(key_storage.back().c_str());
          values.push_back(value_storage.back().c_str());
        }

        t.UpdateOptions(keys.data(), values.data(), kwargs.size()); }, OGA_CAPI("OgaUpdateTokenizerOptions"))
      .def("encode", [](const OgaTokenizer& t, std::string s) {
        auto sequences = OgaSequences::Create();
        t.Encode(s.c_str(), *sequences);
        return ToPython(sequences->Get(0)); }, OGA_CAPI("OgaTokenizerEncode"))
      .def("to_token_id", &OgaTokenizer::ToTokenId, OGA_CAPI("OgaTokenizerToTokenId"))
      .def("decode", [](const OgaTokenizer& t, nb::ndarray<int32_t, nb::c_contig> tokens) -> std::string { return t.Decode(ToSpan(tokens)).p_; }, OGA_CAPI("OgaTokenizerDecode"))
      .def(
          "apply_chat_template",
          [](const OgaTokenizer& t, const std::string& messages, std::optional<std::string> template_str, std::optional<std::string> tools, bool add_generation_prompt) -> std::string {
            return t.ApplyChatTemplate(template_str ? template_str->c_str() : nullptr, messages.c_str(),
                                       tools ? tools->c_str() : nullptr, add_generation_prompt)
                .p_;
          },
          "messages"_a, nb::kw_only(), "template_str"_a = nb::none(), "tools"_a = nb::none(), "add_generation_prompt"_a = true, OGA_CAPI("OgaTokenizerApplyChatTemplate"))
      .def("encode_batch", [](const OgaTokenizer& t, std::vector<std::string> strings) {
        std::vector<const char*> c_strings;
        for (const auto& s : strings)
          c_strings.push_back(s.c_str());
        return t.EncodeBatch(c_strings.data(), c_strings.size()); }, OGA_CAPI("OgaTokenizerEncodeBatch"))
      .def("decode_batch", [](const OgaTokenizer& t, const OgaTensor& tokens) {
        std::vector<std::string> strings;
        auto decoded = t.DecodeBatch(tokens);
        for (size_t i = 0; i < decoded->Count(); i++)
          strings.push_back(decoded->Get(i));
        return strings; }, OGA_CAPI("OgaTokenizerDecodeBatch"))
      .def("create_stream", [](const OgaTokenizer& t) { return OgaTokenizerStream::Create(t); }, OGA_CAPI("OgaCreateTokenizerStream"));

  nb::class_<OgaConfig>(m, "Config", OGA_CLASS_CAPI("Generators::Config"))
      .def(nb::new_([](const std::string& config_path) { return OgaConfig::Create(config_path.c_str()); }), OGA_CAPI("OgaCreateConfig"))
      .def_static(
          "from_package_ep",
          [](const std::string& config_path, const std::string& ep) {
            return OgaConfig::CreateFromPackageEp(config_path.c_str(), ep.empty() ? nullptr : ep.c_str());
          },
          "config_path"_a, "ep"_a,
          "Load an OgaConfig from a model package, selecting the variant whose execution "
          "provider matches `ep`. Pass an empty string to auto-detect when the package "
          "declares a single ep across all variants. " OGA_CAPI("OgaCreateConfigFromPackageEp"))
      .def("append_provider", &OgaConfig::AppendProvider, OGA_CAPI("OgaConfigAppendProvider"))
      .def("set_provider_option", &OgaConfig::SetProviderOption, OGA_CAPI("OgaConfigSetProviderOption"))
      .def("clear_providers", &OgaConfig::ClearProviders, OGA_CAPI("OgaConfigClearProviders"))
      .def("add_model_data", [](OgaConfig& config, const std::string& model_filename, nb::object obj) {
        Py_buffer view;
        if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_SIMPLE) != 0) {
          PyErr_Clear();
          throw std::runtime_error("Unsupported input type. Expected bytes or buffer.");
        }
        config.AddModelData(model_filename, view.buf, static_cast<size_t>(view.len));
        PyBuffer_Release(&view);
      }, OGA_CAPI("OgaConfigAddModelData"))
      .def("remove_model_data", [](OgaConfig& config, const std::string& model_filename) {
        config.RemoveModelData(model_filename.c_str());
      }, OGA_CAPI("OgaConfigRemoveModelData"))
      .def("overlay", &OgaConfig::Overlay, OGA_CAPI("OgaConfigOverlay"))
      .def("set_decoder_provider_options_hardware_device_type", &OgaConfig::SetDecoderProviderOptionsHardwareDeviceType, OGA_CAPI("OgaConfigSetDecoderProviderOptionsHardwareDeviceType"))
      .def("set_decoder_provider_options_hardware_device_id", &OgaConfig::SetDecoderProviderOptionsHardwareDeviceId, OGA_CAPI("OgaConfigSetDecoderProviderOptionsHardwareDeviceId"))
      .def("set_decoder_provider_options_hardware_vendor_id", &OgaConfig::SetDecoderProviderOptionsHardwareVendorId, OGA_CAPI("OgaConfigSetDecoderProviderOptionsHardwareVendorId"))
      .def("clear_decoder_provider_options_hardware_device_type", &OgaConfig::ClearDecoderProviderOptionsHardwareDeviceType, OGA_CAPI("OgaConfigClearDecoderProviderOptionsHardwareDeviceType"))
      .def("clear_decoder_provider_options_hardware_device_id", &OgaConfig::ClearDecoderProviderOptionsHardwareDeviceId, OGA_CAPI("OgaConfigClearDecoderProviderOptionsHardwareDeviceId"))
      .def("clear_decoder_provider_options_hardware_vendor_id", &OgaConfig::ClearDecoderProviderOptionsHardwareVendorId, OGA_CAPI("OgaConfigClearDecoderProviderOptionsHardwareVendorId"));

  nb::class_<OgaModel>(m, "Model", OGA_CLASS_CAPI("Generators::Model"))
      .def(nb::new_([](const OgaConfig& config) { return OgaModel::Create(config); }), OGA_CAPI("OgaCreateModelFromConfig"))
      .def(nb::new_([](const std::string& config_path) { return OgaModel::Create(config_path.c_str()); }), OGA_CAPI("OgaCreateModel"))
      .def_prop_ro("type", [](const OgaModel& model) -> std::string { return model.GetType().p_; }, OGA_CAPI("OgaModelGetType"))
      .def_prop_ro(
          "device_type", [](const OgaModel& model) -> std::string { return model.GetDeviceType().p_; }, "The device type the model is running on. " OGA_CAPI("OgaModelGetDeviceType"))
      .def("create_multimodal_processor", [](const OgaModel& model) { return OgaMultiModalProcessor::Create(model); }, OGA_CAPI("OgaCreateMultiModalProcessor"))
      .def("create_streaming_processor", [](OgaModel& model) { return OgaStreamingProcessor::Create(model); }, "Create a StreamingProcessor for mel spectrogram extraction from raw audio. " OGA_CAPI("OgaCreateStreamingProcessor"));

  nb::class_<PyGenerator>(m, "Generator", OGA_CLASS_CAPI("Generators::Generator"))
      .def(nb::init<const OgaModel&, PyGeneratorParams&>(), OGA_CAPI("OgaCreateGenerator"))
      .def("is_done", &PyGenerator::IsDone, OGA_CAPI("OgaGenerator_IsDone"))
      .def("get_input", &PyGenerator::GetInput, OGA_CAPI("OgaGenerator_GetInput"))
      .def("get_output", &PyGenerator::GetOutput, OGA_CAPI("OgaGenerator_GetOutput"))
      .def("set_inputs", &PyGenerator::SetInputs, OGA_CAPI("OgaGenerator_SetInputs"))
      .def("set_model_input", &PyGenerator::SetModelInput, OGA_CAPI("OgaGenerator_SetModelInput"))
      .def("append_tokens", [](PyGenerator& g, nb::ndarray<int32_t, nb::c_contig> tokens) { g.AppendTokens(tokens); }, OGA_CAPI("OgaGenerator_AppendTokens"))
      .def("append_tokens", [](PyGenerator& g, OgaTensor& tokens) { g.AppendTokens(tokens); }, OGA_CAPI("OgaGenerator_AppendTokens"))
      .def("token_count", &PyGenerator::TokenCount, OGA_CAPI("OgaGenerator_TokenCount"))
      .def("get_logits", &PyGenerator::GetLogits, OGA_CAPI("OgaGenerator_GetLogits"))
      .def("set_logits", &PyGenerator::SetLogits, OGA_CAPI("OgaGenerator_SetLogits"))
      .def("generate_next_token", &PyGenerator::GenerateNextToken, OGA_CAPI("OgaGenerator_GenerateNextToken"))
      .def("rewind_to", &PyGenerator::RewindTo, OGA_CAPI("OgaGenerator_RewindTo"))
      .def("get_next_tokens", &PyGenerator::GetNextTokens, OGA_CAPI("OgaGenerator_GetNextTokens"))
      .def("get_sequence", &PyGenerator::GetSequence, OGA_CAPI("OgaGenerator_GetSequenceData"))
      .def("set_active_adapter", &PyGenerator::SetActiveAdapter, OGA_CAPI("OgaSetActiveAdapter"))
      .def("set_runtime_option", &PyGenerator::SetRuntimeOption, OGA_CAPI("OgaGenerator_SetRuntimeOption"));

  nb::class_<OgaImages>(m, "Images", OGA_CLASS_CAPI("Generators::Images"))
      .def_static("open", [](nb::args image_paths) {
        std::vector<std::string> image_paths_string;
        std::vector<const char*> image_paths_vector;
        for (auto image_path : image_paths) {
          if (!nb::isinstance<nb::str>(image_path))
            throw std::runtime_error("Image paths must be strings.");
          image_paths_string.push_back(nb::cast<std::string>(image_path));
        }
        for (const auto& image_path : image_paths_string)
          image_paths_vector.push_back(image_path.c_str());

        return OgaImages::Load(image_paths_vector);
      }, OGA_CAPI("OgaLoadImages"))
      .def_static("open_bytes", [](nb::args image_datas) {
        std::vector<nb::bytes> holders;
        std::vector<const void*> image_raw_data(image_datas.size());
        std::vector<size_t> image_sizes(image_datas.size());
        holders.reserve(image_datas.size());
        for (size_t i = 0; i < image_datas.size(); ++i) {
          if (!nb::isinstance<nb::bytes>(image_datas[i]))
            throw std::runtime_error("Image data must be bytes.");
          holders.push_back(nb::cast<nb::bytes>(image_datas[i]));
          image_raw_data[i] = holders[i].data();
          image_sizes[i] = holders[i].size();
        }

        return OgaImages::Load(image_raw_data.data(), image_sizes.data(), image_raw_data.size());
      }, OGA_CAPI("OgaLoadImagesFromBuffers"));

  nb::class_<OgaAudios>(m, "Audios", OGA_CLASS_CAPI("Generators::Audios"))
      .def_static("open", [](nb::args audio_paths) {
        std::vector<std::string> audio_paths_string;
        std::vector<const char*> audio_paths_vector;

        for (auto audio_path : audio_paths) {
          if (!nb::isinstance<nb::str>(audio_path))
            throw std::runtime_error("Audio paths must be strings.");
          audio_paths_string.push_back(nb::cast<std::string>(audio_path));
        }
        for (const auto& audio_path : audio_paths_string)
          audio_paths_vector.push_back(audio_path.c_str());

        return OgaAudios::Load(audio_paths_vector);
      }, OGA_CAPI("OgaLoadAudios"))
      .def_static("open_bytes", [](nb::args audio_datas) {
        std::vector<nb::bytes> holders;
        std::vector<const void*> audio_raw_data(audio_datas.size());
        std::vector<size_t> audio_sizes(audio_datas.size());
        holders.reserve(audio_datas.size());
        for (size_t i = 0; i < audio_datas.size(); ++i) {
          if (!nb::isinstance<nb::bytes>(audio_datas[i]))
            throw std::runtime_error("Audio data must be bytes.");
          holders.push_back(nb::cast<nb::bytes>(audio_datas[i]));
          audio_raw_data[i] = holders[i].data();
          audio_sizes[i] = holders[i].size();
        }

        return OgaAudios::Load(audio_raw_data.data(), audio_sizes.data(), audio_raw_data.size());
      }, OGA_CAPI("OgaLoadAudiosFromBuffers"));

  nb::class_<OgaMultiModalProcessor>(m, "MultiModalProcessor", OGA_CLASS_CAPI("Generators::MultiModalProcessor"))
      .def(
          "__call__", [](OgaMultiModalProcessor& processor, nb::object prompts, OgaImages* images, OgaAudios* audios) {
            std::vector<std::string> prompts_str;
            std::vector<const char*> c_prompts;
            if (nb::isinstance<nb::str>(prompts)) {
              // One prompt
              return processor.ProcessImagesAndAudios(nb::cast<std::string>(prompts).c_str(), images, audios);
            } else if (nb::isinstance<nb::list>(prompts)) {
              // Multiple prompts
              for (auto prompt : nb::cast<nb::list>(prompts)) {
                if (!nb::isinstance<nb::str>(prompt)) {
                  throw std::runtime_error("One or more items in the list of provided prompts is not a string.");
                }
                prompts_str.push_back(nb::cast<std::string>(prompt));
              }
            } else if (!prompts.is_none()) {
              // Unsupported type for prompts
              throw std::runtime_error("Unsupported type for prompts. Prompts must be a string or a list of strings.");
            }

            for (const auto& prompt : prompts_str)
              c_prompts.push_back(prompt.c_str());

            return processor.ProcessImagesAndAudios(c_prompts, images, audios);
          },
          "prompt"_a = nb::none(), "images"_a.none() = nb::none(), "audios"_a.none() = nb::none(), OGA_CAPI("OgaProcessorProcessImagesAndAudios / OgaProcessorProcessImagesAndAudiosAndPrompts"))
      .def("create_stream", [](OgaMultiModalProcessor& processor) { return OgaTokenizerStream::Create(processor); }, OGA_CAPI("OgaCreateTokenizerStreamFromProcessor"))
      .def("decode", [](OgaMultiModalProcessor& processor, nb::ndarray<int32_t, nb::c_contig> tokens) -> std::string {
        return processor.Decode(ToSpan(tokens)).p_;
      }, OGA_CAPI("OgaProcessorDecode"));

  nb::class_<OgaAdapters>(m, "Adapters", OGA_CLASS_CAPI("Generators::Adapters"))
      .def(nb::new_([](OgaModel& model) {
        return OgaAdapters::Create(model);
      }), OGA_CAPI("OgaCreateAdapters"))
      .def("unload", &OgaAdapters::UnloadAdapter, OGA_CAPI("OgaUnloadAdapter"))
      .def("load", &OgaAdapters::LoadAdapter, OGA_CAPI("OgaLoadAdapter"));

  nb::class_<OgaRequest>(m, "Request", OGA_CLASS_CAPI("Generators::Request"))
      .def(nb::new_(
          [](PyGeneratorParams& params) {
            return OgaRequest::Create(*params.params_);
          }), OGA_CAPI("OgaCreateRequest"))
      .def("add_tokens", [](OgaRequest& request, nb::ndarray<int32_t, nb::c_contig> tokens) {
        auto sequences = OgaSequences::Create();
        auto tokens_span = ToSpan(tokens);
        sequences->Append(tokens_span.data(), tokens_span.size());
        request.AddTokens(*sequences);
      }, OGA_CAPI("OgaRequestAddTokens"))
      .def("has_unseen_tokens", &OgaRequest::HasUnseenTokens, OGA_CAPI("OgaRequestHasUnseenTokens"))
      .def("is_done", &OgaRequest::IsDone, OGA_CAPI("OgaRequestIsDone"))
      .def("get_unseen_token", &OgaRequest::GetUnseenToken, OGA_CAPI("OgaRequestGetUnseenToken"))
      .def("set_opaque_data", [](OgaRequest& request, nb::object opaque_data) {
        request.SetOpaqueData(opaque_data.ptr());
      }, OGA_CAPI("OgaRequestSetOpaqueData"))
      .def("get_opaque_data", [](OgaRequest& request) -> nb::object {
        auto opaque_data = request.GetOpaqueData();
        if (!opaque_data)
          return nb::none();
        return nb::borrow(static_cast<PyObject*>(opaque_data));
      }, OGA_CAPI("OgaRequestGetOpaqueData"));

  nb::class_<OgaEngine>(m, "Engine", OGA_CLASS_CAPI("Generators::Engine"))
      .def(nb::new_([](OgaModel& model) { return OgaEngine::Create(model); }), OGA_CAPI("OgaCreateEngine"))
      .def("add_request", &OgaEngine::Add, OGA_CAPI("OgaEngineAddRequest"))
      .def("step", &OgaEngine::Step, OGA_CAPI("OgaEngineStep"))
      .def("remove_request", &OgaEngine::Remove, OGA_CAPI("OgaEngineRemoveRequest"))
      .def("has_pending_requests", &OgaEngine::HasPendingRequests, OGA_CAPI("OgaEngineHasPendingRequests"));

  nb::class_<OgaStreamingProcessor>(m, "StreamingProcessor", OGA_CLASS_CAPI("Generators::StreamingProcessor"))
      .def(nb::new_([](OgaModel& model) { return OgaStreamingProcessor::Create(model); }),
           "Create a StreamingProcessor for mel spectrogram extraction.\n"
           "The model must be of type 'nemotron_speech'. " OGA_CAPI("OgaCreateStreamingProcessor"))
      .def(
          "process",
          [](OgaStreamingProcessor& proc, nb::ndarray<float, nb::ndim<1>, nb::c_contig> audio_chunk) -> nb::object {
            auto result = proc.Process(audio_chunk.data(), audio_chunk.size());
            if (result) {
              return nb::cast(std::move(result));
            }
            return nb::none();
          },
          "Feed raw PCM audio. Returns a NamedTensors if a full chunk is ready, or None if more audio is needed. " OGA_CAPI("OgaStreamingProcessorProcess"))
      .def(
          "flush",
          [](OgaStreamingProcessor& proc) -> nb::object {
            auto result = proc.Flush();
            if (result) {
              return nb::cast(std::move(result));
            }
            return nb::none();
          },
          "Flush remaining buffered audio (pads with silence). Returns NamedTensors or None. " OGA_CAPI("OgaStreamingProcessorFlush"))
      .def(
          "set_option",
          [](OgaStreamingProcessor& proc, const std::string& key, const std::string& value) {
            proc.SetOption(key.c_str(), value.c_str());
          },
          "Set a processor option. Keys: 'use_vad', 'vad_threshold', 'silence_duration_ms', 'prefix_padding_ms'. " OGA_CAPI("OgaStreamingProcessorSetOption"))
      .def(
          "get_option",
          [](OgaStreamingProcessor& proc, const std::string& key) {
            return std::string(proc.GetOption(key.c_str()));
          },
          "Get a processor option value by key. " OGA_CAPI("OgaStreamingProcessorGetOption"));

  m.def("set_log_options", &SetLogOptions, OGA_CAPI("OgaSetLogBool / OgaSetLogString"));
  m.def("set_log_callback", &SetLogCallback, "callback"_a.none(), OGA_CAPI("OgaSetLogCallback"));

  m.def("is_cuda_available", []() { return USE_CUDA != 0; }, OGA_NO_CAPI("Compile-time capability flag; no C API equivalent"));
  m.def("is_dml_available", []() { return USE_DML != 0; }, OGA_NO_CAPI("Compile-time capability flag; no C API equivalent"));
  m.def("is_rocm_available", []() { return USE_ROCM != 0; }, OGA_NO_CAPI("Compile-time capability flag; no C API equivalent"));
  m.def("is_webgpu_available", []() { return true; }, OGA_NO_CAPI("Compile-time capability flag; no C API equivalent"));
  m.def("is_qnn_available", []() { return true; }, OGA_NO_CAPI("Compile-time capability flag; no C API equivalent"));
  m.def("is_openvino_available", []() { return true; }, OGA_NO_CAPI("Compile-time capability flag; no C API equivalent"));

  m.def("set_current_gpu_device_id", [](int device_id) { Ort::SetCurrentGpuDeviceId(device_id); }, OGA_CAPI("OgaSetCurrentGpuDeviceId"));
  m.def("get_current_gpu_device_id", []() { return Ort::GetCurrentGpuDeviceId(); }, OGA_CAPI("OgaGetCurrentGpuDeviceId"));

  m.def("register_execution_provider_library", [](const std::string& provider_name, const std::string& path_str) {
    OgaRegisterExecutionProviderLibrary(provider_name.c_str(), path_str.c_str());
  }, OGA_CAPI("OgaRegisterExecutionProviderLibrary"));

  m.def("unregister_execution_provider_library", [](const std::string& provider_name) {
    OgaUnregisterExecutionProviderLibrary(provider_name.c_str());
  }, OGA_CAPI("OgaUnregisterExecutionProviderLibrary"));
}
