// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "../generators.h"
#include "../json.h"
#include "../search.h"
#include "../models/model.h"
#include "../logging.h"

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
pybind11::array_t<T> ToPython(std::span<T> v) {
  return pybind11::array_t<T>(v.size(), v.data());
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
  ((type == Ort::TypeToTensorType<Types> ? result = pybind11::format_descriptor<Types>::format(), true : false) || ...);
  if (result.empty())
    throw std::runtime_error("Unsupported onnx type");
  return result;
}

std::string ToFormatDescriptor(ONNXTensorElementDataType type) {
  return ToFormatDescriptor(type, Ort::TensorTypes{});
}

std::unique_ptr<OrtValue> ToOrtValue(pybind11::array& v) {
  auto type = ToTensorType(v.dtype());

  std::vector<int64_t> shape(v.ndim());
  for (pybind11::ssize_t i = 0; i < v.ndim(); i++)
    shape[i] = v.shape()[i];

  auto p_memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  return OrtValue::CreateTensor(*p_memory_info, v.mutable_data(), v.nbytes(), shape, type);
}

pybind11::array ToNumpy(OrtValue* v, const Generators::Model& model) {
  if (!v)
    return {};

  auto type_info = v->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  auto type = type_info->GetElementType();
  auto element_size = Generators::SizeOf(type);
  auto data = v->GetTensorMutableRawData();

  std::unique_ptr<uint8_t[]> cpu_copy;

#if USE_DML
  // TODO: DML version of this
  if (v->GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU && model.device_type_ == Generators::DeviceType::DML) {
    auto data_size = type_info->GetElementCount() * element_size;
    cpu_copy = std::make_unique<uint8_t[]>(data_size);

    ComPtr<ID3D12Resource> gpu_resource;
    Ort::ThrowOnError(model.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(
        model.allocator_device_,
        data,
        &gpu_resource));

    model.GetDmlReadbackHeap()->ReadbackFromGpu(
        std::span(reinterpret_cast<uint8_t*>(cpu_copy.get()), data_size),
        gpu_resource.Get(),
        0,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    data = cpu_copy.get();
  }
#elif USE_CUDA
  if (v->GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU && model.device_type_ == Generators::DeviceType::CUDA) {
    auto data_size = type_info->GetElementCount() * element_size;
    cpu_copy = std::make_unique<uint8_t[]>(data_size);
    Generators::CudaCheck() == cudaMemcpy(cpu_copy.get(), data, data_size, cudaMemcpyDeviceToHost);
    data = cpu_copy.get();
  }
#endif

  std::vector<int64_t> strides(shape.size());
  {
    auto size = Generators::SizeOf(type);
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
namespace Generators {

// A roaming array is one that can be in CPU or GPU memory, and will copy the memory as needed to be used from anywhere
template <typename T>
struct PyRoamingArray : RoamingArray<T> {
  pybind11::array_t<T> GetNumpy() {
    auto v = this->GetCPU();
    py_cpu_array_ = pybind11::array_t<T>({v.size()}, {sizeof(T)}, v.data(), pybind11::capsule(v.data(), [](void*) {}));
    return py_cpu_array_;
  }

  pybind11::array_t<T> py_cpu_array_;
};

template <typename T>
void Declare_DeviceArray(pybind11::module& m, const char* name) {
  using Type = PyRoamingArray<T>;
  pybind11::class_<Type>(m, name)
      .def(
          "get_array", [](Type& t) -> pybind11::array_t<T> { return t.GetNumpy(); }, pybind11::return_value_policy::reference_internal);
}

struct PyGeneratorParams {
  PyGeneratorParams(const Model& model) : params_{std::make_shared<GeneratorParams>(model)} {
  }

  operator const GeneratorParams&() const { return *params_; }

  std::shared_ptr<GeneratorParams> params_;

  // Turn the python py_input_ids_ into the low level parameters
  void Prepare() {
    // TODO: This will switch to using the variant vs being ifs
    if (py_input_ids_.size() != 0) {
      if (py_input_ids_.ndim() == 1) {  // Just a 1D array
        params_->batch_size = 1;
        params_->sequence_length = static_cast<int>(py_input_ids_.shape(0));
      } else {
        if (py_input_ids_.ndim() != 2)
          throw std::runtime_error("Input IDs can only be 1 or 2 dimensional");

        params_->batch_size = static_cast<int>(py_input_ids_.shape(0));
        params_->sequence_length = static_cast<int>(py_input_ids_.shape(1));
      }
      params_->input_ids = ToSpan(py_input_ids_);
    }

    if (py_whisper_input_features_.size() != 0) {
      GeneratorParams::Whisper& whisper = params_->inputs.emplace<GeneratorParams::Whisper>();
      whisper.input_features = std::make_shared<Tensor>(ToOrtValue(py_whisper_input_features_));
      if (py_alignment_heads_.size() != 0) {
        whisper.alignment_heads = std::make_shared<Tensor>(ToOrtValue(py_alignment_heads_));
      }
    }
  }

  void SetModelInput(const std::string& name, pybind11::array& value) {
    params_->extra_inputs.push_back({name, std::make_shared<Tensor>(ToOrtValue(value))});
    refs_.emplace_back(value);
  }

  void SetSearchOptions(const pybind11::kwargs& dict) {
    for (auto& entry : dict) {
      auto name = entry.first.cast<std::string>();
      try {
        if (pybind11::isinstance<pybind11::float_>(entry.second)) {
          SetSearchNumber(params_->search, name, entry.second.cast<double>());
        } else if (pybind11::isinstance<pybind11::bool_>(entry.second)) {
          SetSearchBool(params_->search, name, entry.second.cast<bool>());
        } else if (pybind11::isinstance<pybind11::int_>(entry.second)) {
          SetSearchNumber(params_->search, name, entry.second.cast<int>());
        } else
          throw std::runtime_error("Unknown search option type, can be float/bool/int:" + name);
      } catch (JSON::unknown_value_error&) {
        throw std::runtime_error("Unknown search option:" + name);
      }
    }
  }

  void TryUseCudaGraphWithMaxBatchSize(pybind11::int_ max_batch_size) {
    Log("warning", "try_use_cuda_graph_with_max_batch_size will be deprecated in release 0.3.0. Please use try_graph_capture_with_max_batch_size instead");
    params_->TryGraphCapture(max_batch_size.cast<int>());
  }

  void TryGraphCaptureWithMaxBatchSize(pybind11::int_ max_batch_size) {
    params_->TryGraphCapture(max_batch_size.cast<int>());
  }

  pybind11::array_t<int32_t> py_input_ids_;
  pybind11::array py_whisper_input_features_;
  pybind11::array py_alignment_heads_;

  std::vector<pybind11::object> refs_;  // References to data we want to ensure doesn't get garbage collected
};

struct PyNamedTensors {
  PyNamedTensors(std::unique_ptr<NamedTensors> named_tensors) : named_tensors_{std::move(named_tensors)} {
  }

  std::unique_ptr<NamedTensors> named_tensors_;
};

struct PyGenerator {
  PyGenerator(Model& model, PyGeneratorParams& params) {
    params.Prepare();
    generator_ = CreateGenerator(model, params);
  }

  pybind11::array_t<int32_t> GetNextTokens() {
    py_tokens_.Assign(generator_->search_->GetNextTokens());
    return ToPython(py_tokens_.GetCPU());
  }

  pybind11::array_t<int32_t> GetSequence(int index) {
    py_sequence_.Assign(generator_->search_->GetSequence(index));
    return ToPython(py_sequence_.GetCPU());
  }

  void ComputeLogits() {
    generator_->ComputeLogits();
  }

  pybind11::array GetOutput(const std::string& name) {
    return ToNumpy(generator_->state_->GetOutput(name.c_str()), *(generator_->model_));
  }

  void GenerateNextToken() {
    generator_->GenerateNextToken();
  }

  bool IsDone() const {
    return generator_->IsDone();
  }

 private:
  std::unique_ptr<Generator> generator_;
  PyRoamingArray<int32_t> py_tokens_;
  PyRoamingArray<int32_t> py_indices_;
  PyRoamingArray<int32_t> py_sequence_;
  PyRoamingArray<int32_t> py_sequencelengths_;
};

void SetLogOptions(const pybind11::kwargs& dict) {
  for (auto& entry : dict) {
    auto name = entry.first.cast<std::string>();
    try {
      if (pybind11::isinstance<pybind11::bool_>(entry.second)) {
        SetLogBool(name, entry.second.cast<bool>());
      } else if (pybind11::isinstance<pybind11::str>(entry.second)) {
        SetLogString(name, entry.second.cast<std::string>());
      } else
        throw std::runtime_error("Unknown log option type, can be bool/string:" + name);
    } catch (JSON::unknown_value_error&) {
      throw std::runtime_error("Unknown log option:" + name);
    }
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
        Generators::Shutdown();
      });
  m.add_object("_cleanup", cleanup);

  // So that python users can catch OrtExceptions specifically
  pybind11::register_exception<Ort::Exception>(m, "OrtException");

  Declare_DeviceArray<float>(m, "DeviceArray_float");
  Declare_DeviceArray<int32_t>(m, "DeviceArray_int32");

  pybind11::class_<PyGeneratorParams>(m, "GeneratorParams")
      .def(pybind11::init<const Model&>())
      // TODO(ryanhill): Remove these entirely or replace with a single property that returns the entire config?
      .def_property_readonly("pad_token_id", [](const PyGeneratorParams& v) { return v.params_->config.model.pad_token_id; })
      .def_property_readonly("eos_token_id", [](const PyGeneratorParams& v) { return v.params_->config.model.eos_token_id; })
      .def_property_readonly("vocab_size", [](const PyGeneratorParams& v) { return v.params_->config.model.vocab_size; })
      .def_readwrite("input_ids", &PyGeneratorParams::py_input_ids_)
      // TODO(baijumeswani): Rename/redesign the whisper_input_features to be more generic
      .def_readwrite("whisper_input_features", &PyGeneratorParams::py_whisper_input_features_)
      .def_readwrite("alignment_heads", &PyGeneratorParams::py_alignment_heads_)
      .def("set_inputs", [](PyGeneratorParams& generator_params, PyNamedTensors* named_tensors) {
        if (!named_tensors || !named_tensors->named_tensors_)
          throw std::runtime_error("No inputs provided.");

        generator_params.params_->SetInputs(*named_tensors->named_tensors_);
      })
      .def("set_model_input", &PyGeneratorParams::SetModelInput)
      .def("set_search_options", &PyGeneratorParams::SetSearchOptions)                                     // See config.h 'struct Search' for the options
      .def("try_use_cuda_graph_with_max_batch_size", &PyGeneratorParams::TryUseCudaGraphWithMaxBatchSize)  // will be deprecated
      .def("try_graph_capture_with_max_batch_size", &PyGeneratorParams::TryGraphCaptureWithMaxBatchSize);

  pybind11::class_<TokenizerStream>(m, "TokenizerStream")
      .def("decode", [](TokenizerStream& t, int32_t token) { return t.Decode(token); });

  pybind11::class_<Tokenizer, std::shared_ptr<Tokenizer>>(m, "Tokenizer")
      .def(pybind11::init([](Model& model) { return model.CreateTokenizer(); }))
      .def("encode", &Tokenizer::Encode)
      .def("to_token_id", &Tokenizer::TokenToTokenId)
      .def("decode", [](const Tokenizer& t, pybind11::array_t<int32_t> tokens) { return t.Decode(ToSpan(tokens)); })
      .def("encode_batch", [](const Tokenizer& t, std::vector<std::string> strings) {
        auto result = t.EncodeBatch(strings);
        return pybind11::array_t<int32_t>({strings.size(), result.size() / strings.size()}, result.data());
      })
      .def("decode_batch", [](const Tokenizer& t, pybind11::array_t<int32_t> tokens) {
        if (tokens.ndim() == 1) {  // Just a 1D array
          return t.DecodeBatch(ToSpan(tokens), 1);
        } else {
          if (tokens.ndim() != 2)
            throw std::runtime_error("token shape can only be 1 or 2 dimensional");

          return t.DecodeBatch(ToSpan(tokens), tokens.shape(0));
        }
      })
      .def("create_stream", [](const Tokenizer& t) { return t.CreateStream(); });

  pybind11::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(pybind11::init([](const std::string& config_path) {
        return CreateModel(GetOrtEnv(), config_path.c_str());
      }))
      .def("generate", [](Model& model, PyGeneratorParams& params) { params.Prepare(); return Generate(model, params); })
      .def_property_readonly(
          "device_type", [](const Model& model) { return to_string(model.device_type_); }, "The device type the model is running on")
      .def("create_multimodal_processor", [](const Model& model) { return model.CreateMultiModalProcessor(); });

  pybind11::class_<PyGenerator>(m, "Generator")
      .def(pybind11::init<Model&, PyGeneratorParams&>())
      .def("is_done", &PyGenerator::IsDone)
      .def("compute_logits", &PyGenerator::ComputeLogits)
      .def("get_output", &PyGenerator::GetOutput)
      .def("generate_next_token", &PyGenerator::GenerateNextToken)
      .def("get_next_tokens", &PyGenerator::GetNextTokens)
      .def("get_sequence", &PyGenerator::GetSequence);

  pybind11::class_<Images>(m, "Images")
      .def_static("open", [](pybind11::args image_paths) {
        if (image_paths.empty())
          throw std::runtime_error("No images provided");

        std::vector<std::string> image_paths_string;
        std::vector<const char*> image_paths_vector;
        for (auto image_path : image_paths) {
          if (!pybind11::isinstance<pybind11::str>(image_path))
            throw std::runtime_error("Image paths must be strings.");
          image_paths_string.push_back(image_path.cast<std::string>());
          image_paths_vector.push_back(image_paths_string.back().c_str());
        }

        return LoadImages(image_paths_vector);
      })
      .def_static("open_bytes", [](pybind11::args image_datas) {
        if (image_datas.empty())
          throw std::runtime_error("No images provided");

        std::unique_ptr<ort_extensions::ImageRawData[]> image_raw_data = std::make_unique<ort_extensions::ImageRawData[]>(image_datas.size());
        for (size_t i = 0; i < image_datas.size(); ++i) {
          if (!pybind11::isinstance<pybind11::bytes>(image_datas[i]))
            throw std::runtime_error("Image data must be bytes.");
          auto bytes = image_datas[i].cast<pybind11::bytes>();
          pybind11::buffer_info info(pybind11::buffer(bytes).request());
          uint8_t* data = reinterpret_cast<uint8_t*>(info.ptr);
          image_raw_data[i] = ort_extensions::ImageRawData(data, data + info.size);
        }

        return std::make_unique<Images>(std::move(image_raw_data), image_datas.size());
      });

  pybind11::class_<Audios>(m, "Audios")
      .def_static("open", [](pybind11::args audio_paths) {
        if (audio_paths.empty())
          throw std::runtime_error("No audios provided");

        std::vector<std::string> audio_paths_string;
        std::vector<const char*> audio_paths_vector;

        for (const auto& audio_path : audio_paths) {
          if (!pybind11::isinstance<pybind11::str>(audio_path))
            throw std::runtime_error("Audio paths must be strings.");
          audio_paths_string.push_back(audio_path.cast<std::string>());
          audio_paths_vector.push_back(audio_paths_string.back().c_str());
        }

        return LoadAudios(audio_paths_vector);
      });

  pybind11::class_<PyNamedTensors>(m, "NamedTensors");

  pybind11::class_<MultiModalProcessor, std::shared_ptr<MultiModalProcessor>>(m, "MultiModalProcessor")
      .def(
          "__call__", [](MultiModalProcessor& processor, const std::optional<std::string>& prompt, const pybind11::kwargs& kwargs) -> std::unique_ptr<PyNamedTensors> {
            if (kwargs.contains("images")) {
              if (processor.image_processor_ == nullptr) {
                throw std::runtime_error("Image processor is not available for this model.");
              }
              const Images* images = kwargs["images"].cast<const Images*>();
              if (!prompt.has_value()) {
                throw std::runtime_error("Prompt is required for processing the image.");
              }
              return std::make_unique<PyNamedTensors>(
                  processor.image_processor_->Process(*processor.tokenizer_, *prompt, images));
            } else if (kwargs.contains("audios")) {
              const Audios* audios = kwargs["audios"].cast<const Audios*>();
              return std::make_unique<PyNamedTensors>(
                  processor.audio_processor_->Process(audios));
            } else {
              throw std::runtime_error("Nothing to process.");
            }
          },
          pybind11::arg("prompt") = pybind11::none())
      .def("create_stream", [](MultiModalProcessor& processor) { return processor.tokenizer_->CreateStream(); })
      .def("decode", [](MultiModalProcessor& processor, pybind11::array_t<int32_t> tokens) {
        return processor.tokenizer_->Decode(ToSpan(tokens));
      });

  m.def("set_log_options", &SetLogOptions);

  m.def("is_cuda_available", []() { return USE_CUDA != 0; });
  m.def("is_dml_available", []() { return USE_DML != 0; });
  m.def("is_rocm_available", []() { return USE_ROCM != 0; });

  m.def("set_current_gpu_device_id", [](int device_id) { Ort::SetCurrentGpuDeviceId(device_id); });
  m.def("get_current_gpu_device_id", []() { return Ort::GetCurrentGpuDeviceId(); });
}

}  // namespace Generators
