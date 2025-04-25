// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#define OGA_USE_SPAN 1
#include "../models/onnxruntime_api.h"
#include "../ort_genai.h"
#include <iostream>

using namespace pybind11::literals;

// If a parameter to a C++ function is an array of float16, this type will let pybind11::array_t<Ort::Float16_t> map to numpy's float16 format
namespace pybind11 {
namespace detail {
template <>
struct npy_format_descriptor<Ort::Float16_t> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(23 /*NPY_FLOAT16*/); /* import numpy as np; print(np.dtype(np.float16).num */
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  static std::string format() {
    // following: https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }
};
}  // namespace detail
}  // namespace pybind11

template <typename T>
std::span<T> ToSpan(pybind11::array_t<T> v) {
  if constexpr (std::is_const_v<T>)
    return {v.data(), static_cast<size_t>(v.size())};
  else
    return {v.mutable_data(), static_cast<size_t>(v.size())};
}

template <typename T>
std::span<T> ToSpan(OgaTensor& v) {
  assert(static_cast<ONNXTensorElementDataType>(v.Type()) == Ort::TypeToTensorType<T>);
  auto shape = v.Shape();
  auto element_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  return {reinterpret_cast<T*>(v.Data()), static_cast<size_t>(element_count)};
}

template <typename T>
pybind11::array_t<T> ToPython(std::span<T> v) {
  return pybind11::array_t<T>{{v.size()}, {sizeof(T)}, v.data()};
}

ONNXTensorElementDataType ToTensorType(const pybind11::dtype& type) {
  switch (type.num()) {
    case pybind11::detail::npy_api::NPY_BOOL_:
      return Ort::TypeToTensorType<bool>;
    case pybind11::detail::npy_api::NPY_UINT8_:
      return Ort::TypeToTensorType<uint8_t>;
    case pybind11::detail::npy_api::NPY_INT8_:
      return Ort::TypeToTensorType<int8_t>;
    case pybind11::detail::npy_api::NPY_UINT16_:
      return Ort::TypeToTensorType<uint16_t>;
    case pybind11::detail::npy_api::NPY_INT16_:
      return Ort::TypeToTensorType<int16_t>;
    case pybind11::detail::npy_api::NPY_UINT32_:
      return Ort::TypeToTensorType<uint32_t>;
    case pybind11::detail::npy_api::NPY_INT32_:
      return Ort::TypeToTensorType<int32_t>;
    case pybind11::detail::npy_api::NPY_UINT64_:
      return Ort::TypeToTensorType<uint64_t>;
    case pybind11::detail::npy_api::NPY_INT64_:
      return Ort::TypeToTensorType<int64_t>;
    case 23 /*NPY_FLOAT16*/:
      return Ort::TypeToTensorType<Ort::Float16_t>;
    case pybind11::detail::npy_api::NPY_FLOAT_:
      return Ort::TypeToTensorType<float>;
    case pybind11::detail::npy_api::NPY_DOUBLE_:
      return Ort::TypeToTensorType<double>;
    default:
      throw std::runtime_error("Unsupported numpy type");
  }
}

int ToNumpyType(ONNXTensorElementDataType type) {
  switch (type) {
    case Ort::TypeToTensorType<bool>:
      return pybind11::detail::npy_api::NPY_BOOL_;
    case Ort::TypeToTensorType<int8_t>:
      return pybind11::detail::npy_api::NPY_INT8_;
    case Ort::TypeToTensorType<uint8_t>:
      return pybind11::detail::npy_api::NPY_UINT8_;
    case Ort::TypeToTensorType<int16_t>:
      return pybind11::detail::npy_api::NPY_INT16_;
    case Ort::TypeToTensorType<uint16_t>:
      return pybind11::detail::npy_api::NPY_UINT16_;
    case Ort::TypeToTensorType<int32_t>:
      return pybind11::detail::npy_api::NPY_INT32_;
    case Ort::TypeToTensorType<uint32_t>:
      return pybind11::detail::npy_api::NPY_UINT32_;
    case Ort::TypeToTensorType<int64_t>:
      return pybind11::detail::npy_api::NPY_INT64_;
    case Ort::TypeToTensorType<uint64_t>:
      return pybind11::detail::npy_api::NPY_UINT64_;
    case Ort::TypeToTensorType<Ort::Float16_t>:
      return 23 /*NPY_FLOAT16*/;
    case Ort::TypeToTensorType<float>:
      return pybind11::detail::npy_api::NPY_FLOAT_;
    case Ort::TypeToTensorType<double>:
      return pybind11::detail::npy_api::NPY_DOUBLE_;
    default:
      throw std::runtime_error("Unsupported onnx type");
  }
}

template <typename... Types>
std::string ToFormatDescriptor(ONNXTensorElementDataType type, Ort::TypeList<Types...>) {
  std::string result;
  (void)((type == Ort::TypeToTensorType<Types> ? result = pybind11::format_descriptor<Types>::format(), true : false) || ...);
  if (result.empty())
    throw std::runtime_error("Unsupported onnx type");
  return result;
}

std::string ToFormatDescriptor(ONNXTensorElementDataType type) {
  return ToFormatDescriptor(type, Ort::TensorTypes{});
}

std::unique_ptr<OgaTensor> ToOgaTensor(pybind11::array& v, bool copy = true) {
  auto type = ToTensorType(v.dtype());

  std::vector<int64_t> shape(v.ndim());
  for (pybind11::ssize_t i = 0; i < v.ndim(); i++)
    shape[i] = v.shape()[i];

  // Check if v is contiguous
  if ((v.flags() & (pybind11::array::c_style | pybind11::array::f_style)) == 0)
    throw std::runtime_error("Array must be contiguous. Please use NumPy's 'ascontiguousarray' method on the value.");

  auto tensor = OgaTensor::Create(copy ? nullptr : v.mutable_data(), shape, static_cast<OgaElementType>(type));
  if (copy) {
    auto ort_data = reinterpret_cast<uint8_t*>(tensor->Data());
    auto python_data = reinterpret_cast<const uint8_t*>(v.data());
    std::copy(python_data, python_data + v.nbytes(), ort_data);
  }
  return tensor;
}

pybind11::array ToNumpy(OgaTensor& v) {
  auto shape = v.Shape();
  auto type = static_cast<ONNXTensorElementDataType>(v.Type());
  auto element_size = Ort::SizeOf(type);
  auto data = v.Data();

  std::vector<int64_t> strides(shape.size());
  {
    auto size = element_size;
    for (size_t i = strides.size(); i-- > 0;) {
      strides[i] = size;
      size *= shape[i];
    }
  }

  pybind11::buffer_info bufinfo{
      data,                                          // Pointer to memory buffer
      static_cast<pybind11::ssize_t>(element_size),  // Size of underlying scalar type
      ToFormatDescriptor(type),                      // Python struct-style format descriptor
      static_cast<pybind11::ssize_t>(shape.size()),  // Number of dimensions
      shape,                                         // Buffer dimensions
      strides                                        // Strides (in bytes) for each index
  };

  return pybind11::array{bufinfo};
}

struct PyGeneratorParams {
  PyGeneratorParams(const OgaModel& model) : params_{OgaGeneratorParams::Create(model)} {}

  operator const OgaGeneratorParams&() const { return *params_; }

  std::unique_ptr<OgaGeneratorParams> params_;

  void SetModelInput(const std::string& name, pybind11::array& value) {
    params_->SetModelInput(name.c_str(), *ToOgaTensor(value, false));
    refs_.emplace_back(value);
  }

  void SetInputs(OgaNamedTensors& named_tensors) {
    params_->SetInputs(named_tensors);
  }

  void SetSearchOptions(const pybind11::kwargs& dict) {
    for (auto& entry : dict) {
      auto name = entry.first.cast<std::string>();
      if (pybind11::isinstance<pybind11::float_>(entry.second)) {
        params_->SetSearchOption(name.c_str(), entry.second.cast<double>());
      } else if (pybind11::isinstance<pybind11::bool_>(entry.second)) {
        params_->SetSearchOptionBool(name.c_str(), entry.second.cast<bool>());
      } else if (pybind11::isinstance<pybind11::int_>(entry.second)) {
        params_->SetSearchOption(name.c_str(), entry.second.cast<int>());
      } else
        throw std::runtime_error("Unknown search option type, can be float/bool/int:" + name);
    }
  }

  void TryGraphCaptureWithMaxBatchSize(pybind11::int_ max_batch_size) {
    std::cerr << "TryGraphCaptureWithMaxBatchSize is deprecated and will be removed in a future release" << std::endl;
  }

  std::vector<pybind11::object> refs_;  // References to data we want to ensure doesn't get garbage collected
};

struct PyGenerator {
  PyGenerator(const OgaModel& model, PyGeneratorParams& params) {
    generator_ = OgaGenerator::Create(model, *params.params_);
  }

  pybind11::array_t<int32_t> GetNextTokens() {
    return ToPython(generator_->GetNextTokens());
  }

  pybind11::array_t<int32_t> GetSequence(int index) {
    return ToPython(generator_->GetSequence(index));
  }

  pybind11::array GetOutput(const std::string& name) {
    return ToNumpy(*generator_->GetOutput(name.c_str()));
  }

  void AppendTokens(OgaTensor& tokens) {
    generator_->AppendTokens(ToSpan<int32_t>(tokens));
  }

  void AppendTokens(pybind11::array_t<int32_t>& tokens) {
    generator_->AppendTokens(ToSpan(tokens));
  }

  pybind11::array_t<float> GetLogits() {
    return ToNumpy(*generator_->GetLogits());
  }

  void SetLogits(pybind11::array_t<float> new_logits) {
    generator_->SetLogits(*ToOgaTensor(new_logits, false));
  }

  void GenerateNextToken() {
    generator_->GenerateNextToken();
  }

  void RewindTo(size_t new_length) {
    generator_->RewindTo(new_length);
  }

  bool IsDone() const {
    return generator_->IsDone();
  }

  void SetActiveAdapter(OgaAdapters& adapters, const std::string& adapter_name) {
    generator_->SetActiveAdapter(adapters, adapter_name.c_str());
  }

 private:
  std::unique_ptr<OgaGenerator> generator_;
};

void SetLogOptions(const pybind11::kwargs& dict) {
  for (auto& entry : dict) {
    auto name = entry.first.cast<std::string>();
    if (pybind11::isinstance<pybind11::bool_>(entry.second)) {
      Oga::SetLogBool(name.c_str(), entry.second.cast<bool>());
    } else if (pybind11::isinstance<pybind11::str>(entry.second)) {
      Oga::SetLogString(name.c_str(), entry.second.cast<std::string>().c_str());
    } else
      throw std::runtime_error("Unknown log option type, can be bool/string:" + name);
  }
}

PYBIND11_MODULE(onnxruntime_genai, m) {
  m.doc() = R"pbdoc(
        Ort Generators library
        ----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  // Add a cleanup call to happen before global variables are destroyed
  static int unused{};  // The capsule needs something to reference
  pybind11::capsule cleanup(
      &unused, "cleanup", [](PyObject*) {
        OgaShutdown();
      });
  m.add_object("_cleanup", cleanup);

  pybind11::class_<PyGeneratorParams>(m, "GeneratorParams")
      .def(pybind11::init<const OgaModel&>())
#if 0
      // TODO(baijumeswani): Rename/redesign the whisper_input_features to be more generic
      .def_readwrite("whisper_input_features", &PyGeneratorParams::py_whisper_input_features_)
      .def_readwrite("alignment_heads", &PyGeneratorParams::py_alignment_heads_)
#endif
      .def("set_inputs", &PyGeneratorParams::SetInputs)
      .def("set_model_input", &PyGeneratorParams::SetModelInput)
      .def("try_graph_capture_with_max_batch_size", &PyGeneratorParams::TryGraphCaptureWithMaxBatchSize)
      .def("set_search_options", &PyGeneratorParams::SetSearchOptions);  // See config.h 'struct Search' for the options

  pybind11::class_<OgaTokenizerStream>(m, "TokenizerStream")
      .def("decode", [](OgaTokenizerStream& t, int32_t token) { return t.Decode(token); });

  pybind11::class_<OgaNamedTensors>(m, "NamedTensors")
      .def(pybind11::init([]() { return OgaNamedTensors::Create(); }))
      .def("__getitem__", [](OgaNamedTensors& named_tensors, const std::string& name) {
        auto tensor = named_tensors.Get(name.c_str());
        if (!tensor)
          throw std::runtime_error("Tensor with name: " + name + " not found.");
        return tensor;
      })
      .def("__setitem__", [](OgaNamedTensors& named_tensors, const std::string& name, pybind11::array& value) {
        named_tensors.Set(name.c_str(), *ToOgaTensor(value));
      })
      .def("__setitem__", [](OgaNamedTensors& named_tensors, const std::string& name, OgaTensor& value) {
        named_tensors.Set(name.c_str(), value);
      })
      .def("__contains__", [](const OgaNamedTensors& named_tensors, const std::string& name) {
        return const_cast<OgaNamedTensors&>(named_tensors).Get(name.c_str()) != nullptr;
      })
      .def("__delitem__", [](OgaNamedTensors& named_tensors, const std::string& name) {
        named_tensors.Delete(name.c_str());
      })
      .def("__len__", &OgaNamedTensors::Count)
      .def("keys", [](OgaNamedTensors& named_tensors) {
        std::vector<std::string> keys;
        auto names = named_tensors.GetNames();
        for (size_t i = 0; i < names->Count(); i++)
          keys.push_back(names->Get(i));
        return keys;
      });

  pybind11::class_<OgaTensor>(m, "Tensor")
      .def(pybind11::init([](pybind11::array& v) { return ToOgaTensor(v); }))
      .def("shape", &OgaTensor::Shape)
      .def("type", &OgaTensor::Type)
      .def("data", &OgaTensor::Data)
      .def("as_numpy", [](OgaTensor& t) { return ToNumpy(t); });

  pybind11::class_<OgaTokenizer>(m, "Tokenizer")
      .def(pybind11::init([](const OgaModel& model) { return OgaTokenizer::Create(model); }))
      .def("encode", [](const OgaTokenizer& t, std::string s) -> pybind11::array_t<int32_t> {
        auto sequences = OgaSequences::Create();
        t.Encode(s.c_str(), *sequences);
        return ToPython(sequences->Get(0));
      })
      .def("to_token_id", &OgaTokenizer::ToTokenId)
      .def("decode", [](const OgaTokenizer& t, pybind11::array_t<int32_t> tokens) -> std::string { return t.Decode(ToSpan(tokens)).p_; })
      .def("apply_chat_template", [](const OgaTokenizer& t, const char* template_str, const char* messages, bool add_generation_prompt) -> std::string { return t.ApplyChatTemplate(template_str, messages, add_generation_prompt).p_; }, pybind11::arg("template_str") = nullptr, pybind11::arg("messages"), pybind11::arg("add_generation_prompt"))
      .def("encode_batch", [](const OgaTokenizer& t, std::vector<std::string> strings) {
        std::vector<const char*> c_strings;
        for (const auto& s : strings)
          c_strings.push_back(s.c_str());
        return t.EncodeBatch(c_strings.data(), c_strings.size()); })
      .def("decode_batch", [](const OgaTokenizer& t, const OgaTensor& tokens) {
        std::vector<std::string> strings;
        auto decoded = t.DecodeBatch(tokens);
        for (size_t i = 0; i < decoded->Count(); i++)
          strings.push_back(decoded->Get(i));
        return strings; })
      .def("create_stream", [](const OgaTokenizer& t) { return OgaTokenizerStream::Create(t); });

  pybind11::class_<OgaConfig>(m, "Config")
      .def(pybind11::init([](const std::string& config_path) { return OgaConfig::Create(config_path.c_str()); }))
      .def("append_provider", &OgaConfig::AppendProvider)
      .def("set_provider_option", &OgaConfig::SetProviderOption)
      .def("clear_providers", &OgaConfig::ClearProviders);

  pybind11::class_<OgaModel>(m, "Model")
      .def(pybind11::init([](const OgaConfig& config) { return OgaModel::Create(config); }))
      .def(pybind11::init([](const std::string& config_path) { return OgaModel::Create(config_path.c_str()); }))
      .def_property_readonly("type", [](const OgaModel& model) -> std::string { return model.GetType().p_; })
      .def_property_readonly(
          "device_type", [](const OgaModel& model) -> std::string { return model.GetDeviceType().p_; }, "The device type the model is running on")
      .def("create_multimodal_processor", [](const OgaModel& model) { return OgaMultiModalProcessor::Create(model); });

  pybind11::class_<PyGenerator>(m, "Generator")
      .def(pybind11::init<const OgaModel&, PyGeneratorParams&>())
      .def("is_done", &PyGenerator::IsDone)
      .def("get_output", &PyGenerator::GetOutput)
      .def("append_tokens", pybind11::overload_cast<pybind11::array_t<int32_t>&>(&PyGenerator::AppendTokens))
      .def("append_tokens", pybind11::overload_cast<OgaTensor&>(&PyGenerator::AppendTokens))
      .def("get_logits", &PyGenerator::GetLogits)
      .def("set_logits", &PyGenerator::SetLogits)
      .def("generate_next_token", &PyGenerator::GenerateNextToken)
      .def("rewind_to", &PyGenerator::RewindTo)
      .def("get_next_tokens", &PyGenerator::GetNextTokens)
      .def("get_sequence", &PyGenerator::GetSequence)
      .def("set_active_adapter", &PyGenerator::SetActiveAdapter);

  pybind11::class_<OgaImages>(m, "Images")
      .def_static("open", [](pybind11::args image_paths) {
        std::vector<std::string> image_paths_string;
        std::vector<const char*> image_paths_vector;
        for (auto image_path : image_paths) {
          if (!pybind11::isinstance<pybind11::str>(image_path))
            throw std::runtime_error("Image paths must be strings.");
          image_paths_string.push_back(image_path.cast<std::string>());
          image_paths_vector.push_back(image_paths_string.back().c_str());
        }

        return OgaImages::Load(image_paths_vector);
      })
      .def_static("open_bytes", [](pybind11::args image_datas) {
        std::vector<const void*> image_raw_data(image_datas.size());
        std::vector<size_t> image_sizes(image_datas.size());
        for (size_t i = 0; i < image_datas.size(); ++i) {
          if (!pybind11::isinstance<pybind11::bytes>(image_datas[i]))
            throw std::runtime_error("Image data must be bytes.");
          auto bytes = image_datas[i].cast<pybind11::bytes>();
          pybind11::buffer_info info(pybind11::buffer(bytes).request());
          image_raw_data[i] = reinterpret_cast<void*>(info.ptr);
          image_sizes[i] = info.size;
        }

        return OgaImages::Load(image_raw_data.data(), image_sizes.data(), image_raw_data.size());
      });

  pybind11::class_<OgaAudios>(m, "Audios")
      .def_static("open", [](pybind11::args audio_paths) {
        std::vector<std::string> audio_paths_string;
        std::vector<const char*> audio_paths_vector;

        for (const auto& audio_path : audio_paths) {
          if (!pybind11::isinstance<pybind11::str>(audio_path))
            throw std::runtime_error("Audio paths must be strings.");
          audio_paths_string.push_back(audio_path.cast<std::string>());
          audio_paths_vector.push_back(audio_paths_string.back().c_str());
        }

        return OgaAudios::Load(audio_paths_vector);
      })
      .def_static("open_bytes", [](pybind11::args audio_datas) {
        std::vector<const void*> audio_raw_data(audio_datas.size());
        std::vector<size_t> audio_sizes(audio_datas.size());
        for (size_t i = 0; i < audio_datas.size(); ++i) {
          if (!pybind11::isinstance<pybind11::bytes>(audio_datas[i]))
            throw std::runtime_error("Audio data must be bytes.");
          auto bytes = audio_datas[i].cast<pybind11::bytes>();
          pybind11::buffer_info info(pybind11::buffer(bytes).request());
          audio_raw_data[i] = reinterpret_cast<void*>(info.ptr);
          audio_sizes[i] = info.size;
        }

        return OgaAudios::Load(audio_raw_data.data(), audio_sizes.data(), audio_raw_data.size());
      });

  pybind11::class_<OgaMultiModalProcessor>(m, "MultiModalProcessor")
      .def(
          "__call__", [](OgaMultiModalProcessor& processor, const std::optional<std::string>& prompt, const pybind11::kwargs& kwargs) {
            OgaImages* images{};
            OgaAudios* audios{};
            if (kwargs.contains("images"))
              images = kwargs["images"].cast<OgaImages*>();
            if (kwargs.contains("audios"))
              audios = kwargs["audios"].cast<OgaAudios*>();
            return processor.ProcessImagesAndAudios(prompt.value_or("").c_str(), images, audios);
          },
          pybind11::arg("prompt") = pybind11::none())
      .def("create_stream", [](OgaMultiModalProcessor& processor) { return OgaTokenizerStream::Create(processor); })
      .def("decode", [](OgaMultiModalProcessor& processor, pybind11::array_t<int32_t> tokens) -> std::string {
        return processor.Decode(ToSpan(tokens)).p_;
      });

  pybind11::class_<OgaAdapters>(m, "Adapters")
      .def(pybind11::init([](OgaModel& model) {
        return OgaAdapters::Create(model);
      }))
      .def("unload", &OgaAdapters::UnloadAdapter)
      .def("load", &OgaAdapters::LoadAdapter);

  m.def("set_log_options", &SetLogOptions);

  m.def("is_cuda_available", []() { return USE_CUDA != 0; });
  m.def("is_dml_available", []() { return USE_DML != 0; });
  m.def("is_rocm_available", []() { return USE_ROCM != 0; });
  m.def("is_webgpu_available", []() { return true; });
  m.def("is_qnn_available", []() { return true; });
  m.def("is_openvino_available", []() { return true; });

  m.def("set_current_gpu_device_id", [](int device_id) { Ort::SetCurrentGpuDeviceId(device_id); });
  m.def("get_current_gpu_device_id", []() { return Ort::GetCurrentGpuDeviceId(); });
}
