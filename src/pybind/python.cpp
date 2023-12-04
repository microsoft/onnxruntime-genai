#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../generators.h"
#include "../search.h"
#include "../search_cuda.h"
#include "../models/model.h"
#include <iostream>

using namespace pybind11::literals;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#ifdef _WIN32
#include <Windows.h>

struct ORTCHAR_String {
  ORTCHAR_String(const char* utf8) {
    int wide_length = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    string_.resize(wide_length);
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, &string_[0], wide_length);
  }

  operator const ORTCHAR_T*() const { return string_.c_str(); }

 private:
  std::wstring string_;
};
#else
#define ORTCHAR_String(string) string
#endif

struct float16 {
  uint16_t v_;
  float AsFloat32() const { return Generators::Float16ToFloat32(v_); }
};

namespace pybind11 {
namespace detail {
template <>
struct npy_format_descriptor<float16> {
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

template <typename T>
pybind11::array_t<T> ToUnownedPython(std::span<T> v) {
  return pybind11::array_t<T>({v.size()}, {sizeof(T)}, v.data(), pybind11::capsule(v.data(), [](void*) {}));
}

namespace Generators {

void TestFP32(pybind11::array_t<float> inputs) {
  pybind11::buffer_info buf_info = inputs.request();
  const float* p = static_cast<const float*>(buf_info.ptr);

  std::cout << "float32 values: ";

  for (unsigned i = 0; i < buf_info.size; i++)
    std::cout << p[i] << " ";
  std::cout << std::endl;
}

void TestFP16(pybind11::array_t<float16> inputs) {
  pybind11::buffer_info buf_info = inputs.request();
  const float16* p = static_cast<const float16*>(buf_info.ptr);

  std::cout << "float16 values: ";

  for (unsigned i = 0; i < buf_info.size; i++)
    std::cout << p[i].AsFloat32() << " ";
  std::cout << std::endl;
}

std::string ToString(const SearchParams& v) {
  std::ostringstream oss;
  oss << "SearchParams("
         "num_beams="
      << v.num_beams << ", batch_size=" << v.batch_size << ", sequence_length=" << v.sequence_length << ", max_length=" << v.max_length << ", pad_token_id=" << v.pad_token_id << ", eos_token_id=" << v.eos_token_id << ", vocab_size=" << v.vocab_size << ", length_penalty=" << v.length_penalty << ", early_stopping=" << v.early_stopping << ")";

  return oss.str();
}

#if 0
std::string ToString(const Gpt_Model& v) {
  std::ostringstream oss;
  oss << "Gpt_Model("
         "vocab_size="
      << v.vocab_size_ << ", head_count=" << v.head_count_ << ", hidden_size=" << v.hidden_size_ << ", layer_count=" << v.layer_count_ << ")";

  return oss.str();
}
#endif

std::unique_ptr<OrtEnv> g_ort_env;

OrtEnv& GetOrtEnv() {
  if (!g_ort_env)
    g_ort_env = OrtEnv::Create();
  return *g_ort_env;
}

// A roaming array is one that can be in CPU or GPU memory, and will copy the memory as needed to be used from anywhere
template <typename T>
struct RoamingArray {
  void SetCPU(std::span<T> cpu) {
    cpu_memory_ = cpu;
    device_memory_ = {};
  }

  void SetGPU(std::span<T> device) {
    device_memory_ = device;
    cpu_memory_ = {};
  }

  std::span<T> GetCPUArray() {
    if (cpu_memory_.empty() && !device_memory_.empty()) {
      cpu_memory_owner_ = CudaMallocHostArray<T>(device_memory_.size(), &cpu_memory_);
      cudaMemcpy(cpu_memory_.data(), device_memory_.data(), cpu_memory_.size_bytes(), cudaMemcpyDeviceToHost);
    }

    return cpu_memory_;
  }

  pybind11::array_t<T> GetNumpyArray() {
    GetCPUArray();
    py_cpu_array_ = pybind11::array_t<T>({cpu_memory_.size()}, {sizeof(T)}, cpu_memory_.data(), pybind11::capsule(cpu_memory_.data(), [](void*) {}));
    return py_cpu_array_;
  }

  std::span<T> GetGPUArray() {
    if (device_memory_.empty() && !cpu_memory_.empty()) {
      device_memory_owner_ = CudaMallocArray<T>(cpu_memory_.size(), &device_memory_);
      cudaMemcpy(device_memory_.data(), cpu_memory_.data(), cpu_memory_.size_bytes(), cudaMemcpyHostToDevice);
    }
    return device_memory_;
  }

  std::span<T> device_memory_;
  cuda_unique_ptr<T> device_memory_owner_;

  std::span<T> cpu_memory_;
  cuda_host_unique_ptr<T> cpu_memory_owner_;
  pybind11::array_t<T> py_cpu_array_;
};

template <typename T>
void Declare_DeviceArray(pybind11::module& m, const char* name) {
  using Type = RoamingArray<T>;
  pybind11::class_<Type>(m, name)
      .def("GetArray", [](Type& t) -> pybind11::array_t<T> { return t.GetNumpyArray(); }, pybind11::return_value_policy::reference_internal);
}

struct PySearchParams : SearchParams {
  // Turn the python py_input_ids_ into the low level parameters
  void Prepare() {
    batch_size = static_cast<int>(py_input_ids_.shape(0));
    sequence_length = static_cast<int>(py_input_ids_.shape(1));
    input_ids = ToSpan(py_input_ids_);
  }

  pybind11::array_t<int32_t> py_input_ids_;
};

struct PyGreedySearch {
  PyGreedySearch(PySearchParams& params, DeviceType device_type) {
    params.Prepare();
    if (device_type == DeviceType::CUDA) {
      cuda_ = std::make_unique<GreedySearch_Cuda>(params);
    }
    else
      cpu_ = std::make_unique<GreedySearch>(params);
  }

  void SetLogits(RoamingArray<float>& inputs) {
    if (cuda_)
      cuda_->SetLogits(inputs.GetGPUArray());
    else
      cpu_->SetLogits(inputs.GetCPUArray());
  }

  int GetSequenceLength() const {
     if (cuda_)
      return cuda_->GetSequenceLength();
     return cpu_->GetSequenceLength();
  }

  RoamingArray<int32_t>& GetNextTokens() {
    if(cuda_)
      py_tokens_.SetGPU(cuda_->GetNextTokens());
    else
      py_tokens_.SetCPU(cpu_->GetNextTokens());
    return py_tokens_;
  }

  RoamingArray<int32_t>& GetSequenceLengths() {
    if(cuda_)
      py_sequencelengths_.SetGPU(cuda_->sequence_lengths_);
    else
      py_sequencelengths_.SetCPU(cpu_->sequence_lengths_);

    return py_sequencelengths_;
  }

  RoamingArray<int32_t>& GetSequence(int index) {
    if (cuda_)
      py_sequence_.SetGPU(cuda_->sequences_.GetSequence(index));
    else
      py_sequence_.SetCPU(cpu_->sequences_.GetSequence(index));
    return py_sequence_;
  }

  bool IsDone() const {
    if (cuda_)
      return cuda_->IsDone();
    return cpu_->IsDone();
  }

  void SelectTop() {
    if(cuda_)
      cuda_->SelectTop();
    else
      cpu_->SelectTop();
  }

  void SampleTopK(int k, float t) {
    if (cuda_)
      cuda_->SampleTopK(k, t);
    else
      cpu_->SampleTopK(k, t);
  }

  void SampleTopP(float p, float t) {
    if(cuda_)
      cuda_->SampleTopP(p, t);
    else
      cpu_->SampleTopP(p, t);
  }

 private:
  std::unique_ptr<GreedySearch> cpu_;
  std::unique_ptr<GreedySearch_Cuda> cuda_;

  RoamingArray<int32_t> py_tokens_;
  RoamingArray<int32_t> py_sequence_;
  RoamingArray<int32_t> py_sequencelengths_;
};

struct PyBeamSearch {
  PyBeamSearch(PySearchParams& params, DeviceType device_type) {
    params.Prepare();
    if (device_type == DeviceType::CUDA) {
      cuda_ = std::make_unique<BeamSearch_Cuda>(params);
    } else
      cpu_ = std::make_unique<BeamSearch>(params);
  }

  void SetLogits(RoamingArray<float>& inputs) {
    if (cuda_)
      cuda_->SetLogits(inputs.GetGPUArray());
    else
      cpu_->SetLogits(inputs.GetCPUArray());
  }

  RoamingArray<int32_t>& GetNextTokens() {
    if (cuda_)
      py_tokens_.SetGPU(cuda_->GetNextTokens());
    else
      py_tokens_.SetCPU(cpu_->GetNextTokens());
    return py_tokens_;
  }

  RoamingArray<int32_t>& GetNextIndices() {
    if(cuda_)
      py_indices_.SetGPU(cuda_->GetNextIndices());
    else
      py_indices_.SetCPU(cpu_->GetNextIndices());
    return py_indices_;
  }

  RoamingArray<int32_t>& GetSequenceLengths() {
    if(cuda_)
      py_sequencelengths_.SetGPU(cuda_->sequence_lengths_);
    else
      py_sequencelengths_.SetCPU(cpu_->sequence_lengths_);
    return py_sequencelengths_;
  }

  RoamingArray<int32_t>& GetSequence(int index) {
    if (cuda_)
      py_sequence_.SetGPU(cuda_->sequences_.GetSequence(index));
    else
      py_sequence_.SetCPU(cpu_->sequences_.GetSequence(index));
    return py_sequence_;
  }

  int GetSequenceLength() const {
    if (cuda_)
      return cuda_->GetSequenceLength();
    return cpu_->GetSequenceLength();
  }

  bool IsDone() const {
    if (cuda_)
      return cuda_->IsDone();
    return cpu_->IsDone();
  }

  void SelectTop() {
    if (cuda_)
      cuda_->SelectTop();
    else
      cpu_->SelectTop();
  }

 private:
  std::unique_ptr<BeamSearch_Cuda> cuda_;
  std::unique_ptr<BeamSearch> cpu_;

  RoamingArray<int32_t> py_tokens_;
  RoamingArray<int32_t> py_indices_;
  RoamingArray<int32_t> py_sequence_;
  RoamingArray<int32_t> py_sequencelengths_;
};

struct PyState {
  PyState(Model& model, RoamingArray<int32_t>& sequence_lengths, const SearchParams& search_params) {
    is_cuda_ = model.device_type_ == DeviceType::CUDA;

    if (is_cuda_)
      state_ = model.CreateState(sequence_lengths.GetGPUArray(), search_params);
    else
      state_ = model.CreateState(sequence_lengths.GetCPUArray(), search_params);
  }

  RoamingArray<float>& Run(int current_length, RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t>& next_indices) {
    if (is_cuda_)
      py_logits_.SetGPU(state_->Run(current_length, next_tokens.GetGPUArray(), next_indices.GetGPUArray()));
    else
      py_logits_.SetCPU(state_->Run(current_length, next_tokens.GetCPUArray(), next_indices.GetCPUArray()));

    return py_logits_;
  }

 private:
  bool is_cuda_ {};
  std::unique_ptr<State> state_;
  RoamingArray<float> py_logits_;
};

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

  Declare_DeviceArray<ScoreType>(m, "DeviceArray_ScoreType");
  Declare_DeviceArray<int32_t>(m, "DeviceArray_int32");

  pybind11::enum_<DeviceType>(m, "DeviceType")
      .value("Auto", DeviceType::Auto)
      .value("CPU", DeviceType::CPU)
      .value("CUDA", DeviceType::CUDA)
      .export_values();

  pybind11::class_<PySearchParams>(m, "SearchParams")
      .def(pybind11::init<const Model&>())
      .def_readonly("pad_token_id", &PySearchParams::pad_token_id)
      .def_readonly("eos_token_id", &PySearchParams::eos_token_id)
      .def_readonly("vocab_size", &PySearchParams::vocab_size)
      .def_readwrite("num_beams", &PySearchParams::num_beams)
//      .def_readwrite("batch_size", &PySearchParams::batch_size)
//      .def_readwrite("sequence_length", &PySearchParams::sequence_length)
      .def_readwrite("max_length", &PySearchParams::max_length)
      .def_readwrite("length_penalty", &PySearchParams::length_penalty)
      .def_readwrite("early_stopping", &PySearchParams::early_stopping)
      .def_readwrite("input_ids", &PySearchParams::py_input_ids_)
#if 0
      .def_property(
          "input_ids",
          [](PySearchParams& s) -> pybind11::array_t<int32_t> { return s.py_input_ids_; },
          [](PySearchParams& s, pybind11::array_t<int32_t> v) {
             s.py_input_ids_=v; s.input_ids = ToSpan(s.py_input_ids_);
          })
#endif
      .def("__repr__", [](PySearchParams& s) { return ToString(s); });

  pybind11::class_<PyGreedySearch>(m, "GreedySearch")
      .def(pybind11::init<PySearchParams&, DeviceType>())
      .def("SetLogits", &PyGreedySearch::SetLogits)
      .def("GetSequenceLength", &PyGreedySearch::GetSequenceLength)
      .def("GetSequenceLengths", &PyGreedySearch::GetSequenceLengths, pybind11::return_value_policy::reference_internal)
      .def("GetNextTokens", &PyGreedySearch::GetNextTokens, pybind11::return_value_policy::reference_internal)
      .def("IsDone", &PyGreedySearch::IsDone)
      .def("SelectTop", &PyGreedySearch::SelectTop)
      .def("SampleTopK", &PyGreedySearch::SampleTopK)
      .def("SampleTopP", &PyGreedySearch::SampleTopP)
      .def("GetSequence", &PyGreedySearch::GetSequence, pybind11::return_value_policy::reference_internal);

  pybind11::class_<PyBeamSearch>(m, "BeamSearch")
      .def(pybind11::init<PySearchParams&, DeviceType>())
      .def("SetLogits", &PyBeamSearch::SetLogits)
      .def("GetSequenceLength", &PyBeamSearch::GetSequenceLength)
      .def("GetSequenceLengths", &PyBeamSearch::GetSequenceLengths, pybind11::return_value_policy::reference_internal)
      .def("GetNextTokens", &PyBeamSearch::GetNextTokens, pybind11::return_value_policy::reference_internal)
      .def("GetNextIndices", &PyBeamSearch::GetNextIndices, pybind11::return_value_policy::reference_internal)
      .def("IsDone", &PyBeamSearch::IsDone)
      .def("SelectTop", &PyBeamSearch::SelectTop)
      .def("GetSequence", &PyBeamSearch::GetSequence, pybind11::return_value_policy::reference_internal);

  // If we support models, we need to init the OrtApi
  Ort::InitApi();

  m.def("print", &TestFP32, "Test float32");
  m.def("print", &TestFP16, "Test float16");

  pybind11::class_<Model>(m, "Model")
      .def(pybind11::init([](const std::string& config_path, DeviceType device_type) {
             auto provider_options = GetDefaultProviderOptions(device_type);
             return new Model(GetOrtEnv(), config_path.c_str(), &provider_options);
           }),
           "str"_a, "device_type"_a = DeviceType::Auto)
      .def("Generate", [](Model& model, PySearchParams& search_params) { search_params.Prepare(); return model.Generate(search_params); })
      .def("CreateState", [](Model& model, RoamingArray<int32_t>& sequence_lengths, const PySearchParams& search_params) { return new PyState(model, sequence_lengths, search_params); })
      .def_property_readonly("DeviceType", [](const Model& s) { return s.device_type_; });

  pybind11::class_<PyState>(m, "State")
      .def(pybind11::init<Model&, RoamingArray<int32_t>&, const PySearchParams&>())
      .def("Run", &PyState::Run, "current_length"_a, "next_tokens"_a, "next_indices"_a = RoamingArray<int32_t>{},
           pybind11::return_value_policy::reference_internal);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

}  // namespace Generators