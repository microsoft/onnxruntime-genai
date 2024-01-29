#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "../generators.h"
#include "../search.h"
#if USE_CUDA
#include "../search_cuda.h"
#endif
#include "../models/model.h"

using namespace pybind11::literals;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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
          "GetArray", [](Type& t) -> pybind11::array_t<T> { return t.GetNumpy(); }, pybind11::return_value_policy::reference_internal);
}

struct PySearchParams : SearchParams {
  // Turn the python py_input_ids_ into the low level parameters
  void Prepare() {
    // TODO: This will switch to using the variant vs being ifs
    if (py_input_ids_.size() != 0) {
      if (py_input_ids_.ndim() == 1) {  // Just a 1D array
        batch_size = 1;
        sequence_length = static_cast<int>(py_input_ids_.shape(0));
      } else {
        if (py_input_ids_.ndim() != 2)
          throw std::runtime_error("Input IDs can only be 1 or 2 dimensional");

        batch_size = static_cast<int>(py_input_ids_.shape(0));
        sequence_length = static_cast<int>(py_input_ids_.shape(1));
      }
      input_ids = ToSpan(py_input_ids_);
    }

    if (py_whisper_input_features_.size() != 0) {
      SearchParams::Whisper& whisper = inputs.emplace<SearchParams::Whisper>();
      std::span<const int64_t> shape(py_whisper_input_features_.shape(), py_whisper_input_features_.ndim());
      whisper.input_features = OrtValue::CreateTensor<float>(Ort::Allocator::GetWithDefaultOptions().GetInfo(), ToSpan(py_whisper_input_features_), shape);
      whisper.decoder_input_ids = ToSpan(py_whisper_decoder_input_ids_);
      batch_size = 1;
      sequence_length = static_cast<int>(py_whisper_decoder_input_ids_.shape(1));
      input_ids = ToSpan(py_whisper_decoder_input_ids_);
    }
  }

  pybind11::array_t<int32_t> py_input_ids_;
  pybind11::array_t<float> py_whisper_input_features_;
  pybind11::array_t<int32_t> py_whisper_decoder_input_ids_;
};

struct PySearch {
  PySearch(PySearchParams& params) {
    params.Prepare();
    search_ = params.CreateSearch();
  }

  void SetLogits(PyRoamingArray<float>& inputs) {
    search_->SetLogits(inputs);
  }

  int GetSequenceLength() const {
    return search_->GetSequenceLength();
  }

  PyRoamingArray<int32_t>& GetNextTokens() {
    py_tokens_.Assign(search_->GetNextTokens());
    return py_tokens_;
  }

  PyRoamingArray<int32_t>& GetNextIndices() {
    py_indices_.Assign(search_->GetNextIndices());
    return py_indices_;
  }

  PyRoamingArray<int32_t>& GetSequenceLengths() {
    py_sequencelengths_.Assign(search_->GetSequenceLengths());
    return py_sequencelengths_;
  }

  PyRoamingArray<int32_t>& GetSequence(int index) {
    py_sequence_.Assign(search_->GetSequence(index));
    return py_sequence_;
  }

  bool IsDone() const {
    return search_->IsDone();
  }

  void SelectTop() {
    search_->SelectTop();
  }

  void SampleTopK(int k, float t) {
    search_->SampleTopK(k, t);
  }

  void SampleTopP(float p, float t) {
    search_->SampleTopP(p, t);
  }

 private:
  std::unique_ptr<Search> search_;
  PyRoamingArray<int32_t> py_tokens_;
  PyRoamingArray<int32_t> py_indices_;
  PyRoamingArray<int32_t> py_sequence_;
  PyRoamingArray<int32_t> py_sequencelengths_;
};

struct PyState {
  PyState(Model& model, PyRoamingArray<int32_t>& sequence_lengths, const SearchParams& search_params) {
    state_ = model.CreateState(sequence_lengths, search_params);
  }

  PyRoamingArray<float>& Run(int current_length, PyRoamingArray<int32_t>& next_tokens, PyRoamingArray<int32_t>& next_indices) {
    py_logits_.Assign(state_->Run(current_length, next_tokens, next_indices));
    return py_logits_;
  }

 private:
  std::unique_ptr<State> state_;
  PyRoamingArray<float> py_logits_;
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

  Declare_DeviceArray<float>(m, "DeviceArray_float");
  Declare_DeviceArray<int32_t>(m, "DeviceArray_int32");

  pybind11::enum_<DeviceType>(m, "DeviceType")
      .value("Auto", DeviceType::Auto)
      .value("CPU", DeviceType::CPU)
      .value("CUDA", DeviceType::CUDA)
      .export_values();

  pybind11::class_<PySearchParams>(m, "SearchParams")
      .def(pybind11::init<const Model&>())
      .def("CreateSearch", [](PySearchParams& v) { return new PySearch(v); })
      .def_readonly("pad_token_id", &PySearchParams::pad_token_id)
      .def_readonly("eos_token_id", &PySearchParams::eos_token_id)
      .def_readonly("vocab_size", &PySearchParams::vocab_size)
      .def_readwrite("num_beams", &PySearchParams::num_beams)
      .def_readwrite("max_length", &PySearchParams::max_length)
      .def_readwrite("length_penalty", &PySearchParams::length_penalty)
      .def_readwrite("early_stopping", &PySearchParams::early_stopping)
      .def_readwrite("input_ids", &PySearchParams::py_input_ids_)
      .def_readwrite("whisper_input_features", &PySearchParams::py_whisper_input_features_)
      .def_readwrite("whisper_decoder_input_ids", &PySearchParams::py_whisper_decoder_input_ids_)
      .def("__repr__", [](PySearchParams& s) { return ToString(s); });

  pybind11::class_<PySearch>(m, "Search")
      .def("SetLogits", &PySearch::SetLogits)
      .def("GetSequenceLength", &PySearch::GetSequenceLength)
      .def("GetSequenceLengths", &PySearch::GetSequenceLengths, pybind11::return_value_policy::reference_internal)
      .def("GetNextTokens", &PySearch::GetNextTokens, pybind11::return_value_policy::reference_internal)
      .def("GetNextIndices", &PySearch::GetNextIndices, pybind11::return_value_policy::reference_internal)
      .def("IsDone", &PySearch::IsDone)
      .def("SelectTop", &PySearch::SelectTop)
      .def("SampleTopK", &PySearch::SampleTopK)
      .def("SampleTopP", &PySearch::SampleTopP)
      .def("GetSequence", &PySearch::GetSequence, pybind11::return_value_policy::reference_internal);

  // We need to init the OrtApi before we can use it
  Ort::InitApi();

  m.def("print", &TestFP32, "Test float32");
  m.def("print", &TestFP16, "Test float16");

#if USE_TOKENIZER
  pybind11::class_<Tokenizer>(m, "Tokenizer")
      .def("encode", &Tokenizer::Encode)
      .def("decode", [](const Tokenizer& t, pybind11::array_t<int32_t> tokens) { return t.Decode(ToSpan(tokens)); });
#endif

  pybind11::class_<Model>(m, "Model")
      .def(pybind11::init([](const std::string& config_path, DeviceType device_type) {
             auto provider_options = GetDefaultProviderOptions(device_type);
             return CreateModel(GetOrtEnv(), config_path.c_str(), &provider_options);
           }),
           "str"_a, "device_type"_a = DeviceType::Auto)
      .def("Generate", [](Model& model, PySearchParams& search_params) { search_params.Prepare(); return model.Generate(search_params); })
#if USE_TOKENIZER
      .def("CreateTokenizer", [](Model& model) { return model.CreateTokenizer(); })
#endif
      .def("CreateState", [](Model& model, PyRoamingArray<int32_t>& sequence_lengths, const PySearchParams& search_params) { return new PyState(model, sequence_lengths, search_params); })
      .def_property_readonly("DeviceType", [](const Model& s) { return s.device_type_; });

  pybind11::class_<PyState>(m, "State")
      .def(pybind11::init<Model&, PyRoamingArray<int32_t>&, const PySearchParams&>())
      .def("Run", &PyState::Run, "current_length"_a, "next_tokens"_a, "next_indices"_a = PyRoamingArray<int32_t>{},
           pybind11::return_value_policy::reference_internal);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

}  // namespace Generators