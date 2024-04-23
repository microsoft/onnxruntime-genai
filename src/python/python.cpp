#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "../generators.h"
#include "../json.h"
#include "../search.h"
#include "../models/model.h"

using namespace pybind11::literals;

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

namespace Generators {

std::unique_ptr<OrtEnv> g_ort_env;

OrtEnv& GetOrtEnv() {
  if (!g_ort_env) {
    g_ort_env = OrtEnv::Create();
  }
  return *g_ort_env;
}

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
#ifdef __APPLE__
      std::span shape(reinterpret_cast<const int64_t*>(py_whisper_input_features_.shape()),
                      py_whisper_input_features_.ndim());
#else
      std::span<const int64_t> shape(py_whisper_input_features_.shape(), py_whisper_input_features_.ndim());
#endif
      whisper.input_features = OrtValue::CreateTensor<float>(Ort::Allocator::GetWithDefaultOptions().GetInfo(), ToSpan(py_whisper_input_features_), shape);
      whisper.decoder_input_ids = ToSpan(py_whisper_decoder_input_ids_);
      params_->batch_size = 1;
      params_->sequence_length = static_cast<int>(py_whisper_decoder_input_ids_.shape(1));
      params_->input_ids = ToSpan(py_whisper_decoder_input_ids_);
    }
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
      } catch (JSON::unknown_value_error& e) {
        throw std::runtime_error("Unknown search option:" + name);
      }
    }
  }

  void TryUseCudaGraphWithMaxBatchSize(pybind11::int_ max_batch_size) {
    params_->max_batch_size = max_batch_size.cast<int>();
  }

  pybind11::array_t<int32_t> py_input_ids_;
  pybind11::array_t<float> py_whisper_input_features_;
  pybind11::array_t<int32_t> py_whisper_decoder_input_ids_;
};

struct PyGenerator {
  PyGenerator(Model& model, PyGeneratorParams& params) {
    params.Prepare();
    model.GetMaxBatchSizeFromGeneratorParams(params);
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
    } catch (JSON::unknown_value_error& e) {
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

  // So that python users can catch OrtExceptions specifically
  pybind11::register_exception<Ort::Exception>(m, "OrtException");

  Declare_DeviceArray<float>(m, "DeviceArray_float");
  Declare_DeviceArray<int32_t>(m, "DeviceArray_int32");

  pybind11::class_<PyGeneratorParams>(m, "GeneratorParams")
      .def(pybind11::init<const Model&>())
      .def_property_readonly("pad_token_id", [](const PyGeneratorParams& v) { return v.params_->pad_token_id; })
      .def_property_readonly("eos_token_id", [](const PyGeneratorParams& v) { return v.params_->eos_token_id; })
      .def_property_readonly("vocab_size", [](const PyGeneratorParams& v) { return v.params_->vocab_size; })
      .def_readwrite("input_ids", &PyGeneratorParams::py_input_ids_)
      .def_readwrite("whisper_input_features", &PyGeneratorParams::py_whisper_input_features_)
      .def_readwrite("whisper_decoder_input_ids", &PyGeneratorParams::py_whisper_decoder_input_ids_)
      .def("set_search_options", &PyGeneratorParams::SetSearchOptions)  // See config.h 'struct Search' for the options
      .def("try_use_cuda_graph_with_max_batch_size", &PyGeneratorParams::TryUseCudaGraphWithMaxBatchSize);

  // We need to init the OrtApi before we can use it
  Ort::InitApi();

  pybind11::class_<TokenizerStream>(m, "TokenizerStream")
      .def("decode", [](TokenizerStream& t, int32_t token) { return t.Decode(token); });

  pybind11::class_<Tokenizer, std::shared_ptr<Tokenizer>>(m, "Tokenizer")
      .def(pybind11::init([](Model& model) { return model.CreateTokenizer(); }))
      .def("encode", &Tokenizer::Encode)
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
      .def("generate", [](Model& model, PyGeneratorParams& params) { params.Prepare(); model.GetMaxBatchSizeFromGeneratorParams(params); return Generate(model, params); })
      .def_property_readonly("device_type", [](const Model& s) { return s.device_type_; });

  pybind11::class_<PyGenerator>(m, "Generator")
      .def(pybind11::init<Model&, PyGeneratorParams&>())
      .def("is_done", &PyGenerator::IsDone)
      .def("compute_logits", &PyGenerator::ComputeLogits)
      .def("generate_next_token", &PyGenerator::GenerateNextToken)
      .def("get_next_tokens", &PyGenerator::GetNextTokens)
      .def("get_sequence", &PyGenerator::GetSequence);

  m.def("set_log_options", &SetLogOptions);

  m.def("is_cuda_available", []() {
#if USE_CUDA
    return true;
#else
        return false;
#endif
  });

  m.def("is_dml_available", []() {
#if USE_DML
    return true;
#else
        return false;
#endif
  });

  m.def("set_current_gpu_device_id", [](int device_id) { Ort::SetCurrentGpuDeviceId(device_id); });
  m.def("get_current_gpu_device_id", []() { return Ort::GetCurrentGpuDeviceId(); });
}

}  // namespace Generators