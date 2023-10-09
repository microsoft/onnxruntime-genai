#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../generators.h"
#include "../search.h"
#include "../search_cuda.h"
#include "../models/gpt_cpu.h"
#include "../models/gpt_cuda.h"
#include <iostream>


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
using ORTCHAR_String = std::string;
#endif

struct float16 {
  uint16_t v_;
  float AsFloat32() const {
    // Extract sign, exponent, and fraction from numpy.float16
    int sign = (v_ & 0x8000) >> 15;
    int exponent = (v_ & 0x7C00) >> 10;
    int fraction = v_ & 0x03FF;

    // Handle special cases
    if (exponent == 0) {
      if (fraction == 0) {
        // Zero
        return sign ? -0.0f : 0.0f;
      } else {
        // Subnormal number
        return std::ldexp((sign ? -1.0f : 1.0f) * fraction / 1024.0f, -14);
      }
    } else if (exponent == 31) {
      if (fraction == 0) {
        // Infinity
        return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
      } else {
        // NaN
        return std::numeric_limits<float>::quiet_NaN();
      }
    }

    // Normalized number
    return std::ldexp((sign ? -1.0f : 1.0f) * (1.0f + fraction / 1024.0f), exponent - 15);
  }
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

std::string ToString(const GptModelParams& v) {
  std::ostringstream oss;
  oss << "GptModelParams("
         "vocab_size="
      << v.vocab_size << ", head_count=" << v.head_count << ", hidden_size=" << v.hidden_size << ", layer_count=" << v.layer_count << ")";

  return oss.str();
}

std::unique_ptr<OrtEnv> g_ort_env;

OrtEnv& GetOrtEnv() {
  if (!g_ort_env)
    g_ort_env = OrtEnv::Create();
  return *g_ort_env;
}

struct PySearchParams : SearchParams {
  pybind11::array_t<int32_t> py_input_ids_;
};

PYBIND11_MODULE(ort_generators, m) {
  m.doc() = R"pbdoc(
        Ort Generators library
        ----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  pybind11::class_<PySearchParams>(m, "SearchParams")
      .def(pybind11::init<>())
      .def_readwrite("num_beams", &PySearchParams::num_beams)
      .def_readwrite("batch_size", &PySearchParams::batch_size)
      .def_readwrite("sequence_length", &PySearchParams::sequence_length)
      .def_readwrite("max_length", &PySearchParams::max_length)
      .def_readwrite("pad_token_id", &PySearchParams::pad_token_id)
      .def_readwrite("eos_token_id", &PySearchParams::eos_token_id)
      .def_readwrite("vocab_size", &PySearchParams::vocab_size)
      .def_readwrite("length_penalty", &PySearchParams::length_penalty)
      .def_readwrite("early_stopping", &PySearchParams::early_stopping)
      .def_property(
          "input_ids",
          [](PySearchParams& s) -> pybind11::array_t<int32_t> { return s.py_input_ids_; },
          [](PySearchParams& s, pybind11::array_t<int32_t> v) { s.py_input_ids_=v; s.input_ids = ToSpan(s.py_input_ids_); })
      .def("__repr__", [](PySearchParams& s) { return ToString(s); });

  pybind11::class_<GreedySearch>(m, "GreedySearch")
      .def(pybind11::init<const PySearchParams&>())
      .def("SetLogits", [](GreedySearch& s, pybind11::array_t<float> inputs) { s.SetLogits(ToSpan(inputs)); })
      .def("GetSequenceLength", &GreedySearch::GetSequenceLength)
      .def("GetSequenceLengths", [](GreedySearch& s) -> pybind11::array_t<int32_t> { return ToPython(s.sequence_lengths_); })
      .def("GetNextTokens", [](GreedySearch& s) -> pybind11::array_t<int32_t> { return ToPython(s.GetNextTokens()); })
      .def("IsDone", &GreedySearch::IsDone)
      .def("SelectTop1", &GreedySearch::SelectTop1)
      .def("GetSequence", [](GreedySearch& s, int index) -> pybind11::array_t<int32_t> { return ToPython(s.sequences_.GetSequence(index)); });

  pybind11::class_<GreedySearch_Cuda>(m, "GreedySearch_Cuda")
      .def(pybind11::init([](const PySearchParams& v) { SearchParams_Cuda s; static_cast<SearchParams&>(s)=v; return new GreedySearch_Cuda(s); }))
      .def("SetLogits", [](GreedySearch_Cuda& s, pybind11::array_t<float> inputs) { s.SetLogits(ToSpan(inputs)); })
      .def("GetSequenceLength", &GreedySearch_Cuda::GetSequenceLength)
      .def("GetSequenceLengths", [](GreedySearch_Cuda& s) -> pybind11::array_t<int32_t> { return ToPython(s.sequence_lengths_); })
      .def("GetNextTokens", [](GreedySearch_Cuda& s) -> pybind11::array_t<int32_t> { return ToPython(s.GetNextTokens()); })
      .def("IsDone", &GreedySearch_Cuda::IsDone)
      .def("SelectTop1", &GreedySearch_Cuda::SelectTop1)
      .def("GetSequence", [](GreedySearch_Cuda& s, int index) -> pybind11::array_t<int32_t> { return ToPython(s.sequences_.GetSequence(index)); });

  pybind11::class_<BeamSearch>(m, "BeamSearch")
      .def(pybind11::init<const PySearchParams&>())
      .def("SetLogits", [](BeamSearch& s, pybind11::array_t<float> inputs) { s.SetLogits(ToSpan(inputs)); })
      .def("GetSequenceLength", &BeamSearch::GetSequenceLength)
      .def("GetSequenceLengths", [](BeamSearch& s) -> pybind11::array_t<int32_t> { return ToPython(s.sequence_lengths_); })
      .def("GetNextTokens", [](BeamSearch& s) -> pybind11::array_t<int32_t> { return ToPython(s.GetNextTokens()); })
      .def("IsDone", &BeamSearch::IsDone)
      .def("SelectTopK", &BeamSearch::SelectTopK)
      .def("GetSequence", [](BeamSearch& s, int index) -> pybind11::array_t<int32_t> { return ToPython(s.sequences_.GetSequence(index)); });

  // If we support models, we need to init the OrtApi
  Ort::InitApi();

  m.def("print", &TestFP32, "Test float32");
  m.def("print", &TestFP16, "Test float16");

  pybind11::class_<Gpt>(m, "Gpt")
      .def(pybind11::init([](const std::string& str) { return new Gpt(GetOrtEnv(), ORTCHAR_String(str.c_str())); }))
      .def("CreateInputs", [](Gpt& s, pybind11::array_t<int32_t> sequence_lengths, const PySearchParams& params) { s.CreateInputs(ToSpan(sequence_lengths), params); })
      .def("GetVocabSize", &Gpt::GetVocabSize)
      .def("Run", [](Gpt& s, pybind11::array_t<int32_t> next_tokens, pybind11::array_t<int32_t> next_indices, int current_length) { s.Run(ToSpan(next_tokens), ToSpan(next_indices), current_length); })
      .def("GetLogits", [](Gpt& s) -> pybind11::array_t<float> { return ToPython(s.GetLogits()); });

  pybind11::class_<Gpt_Cuda>(m, "Gpt_Cuda")
      .def(pybind11::init([](const std::string& str) { return new Gpt_Cuda(GetOrtEnv(), ORTCHAR_String(str.c_str()), nullptr); }))
      .def("CreateInputs", [](Gpt_Cuda& s, pybind11::array_t<int32_t> sequence_lengths, const PySearchParams& params) { s.CreateInputs(ToSpan(sequence_lengths), params); })
      .def("GetVocabSize", &Gpt_Cuda::GetVocabSize)
      .def("Run", [](Gpt_Cuda& s, pybind11::array_t<int32_t> next_tokens, pybind11::array_t<int32_t> next_indices, int current_length) { s.Run(ToSpan(next_tokens), ToSpan(next_indices), current_length); })
      .def("GetLogits", [](Gpt_Cuda& s) -> pybind11::array_t<float> { return ToPython(s.GetLogits()); });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

}  // namespace Generators