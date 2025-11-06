// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "py_wrappers.h"
#include "../generators.h"
#include "../models/model.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/ndarray.h>
#include <cstring>

namespace nb = nanobind;

namespace Generators {

// Wrapper for TokenizerStream
struct PyTokenizerStream {
  std::unique_ptr<TokenizerStream> stream;
  
  PyTokenizerStream(const Tokenizer& tokenizer) {
    stream = tokenizer.CreateStream();
  }
  
  ~PyTokenizerStream() {
    // Explicitly reset to ensure destruction
    stream.reset();
  }
  
  const std::string& Decode(int32_t token) {
    return stream->Decode(token);
  }
};

// Wrapper for Tokenizer
struct PyTokenizer {
  std::shared_ptr<Tokenizer> tokenizer;
  
  PyTokenizer(PyModel& model) {
    tokenizer = model.GetModel()->CreateTokenizer();
  }
  
  std::unique_ptr<PyTokenizerStream> CreateStream() const {
    return std::make_unique<PyTokenizerStream>(*tokenizer);
  }
  
  std::vector<int32_t> Encode(std::string_view text) const {
    std::string text_str(text);
    return tokenizer->Encode(text_str.c_str());
  }
  
  std::string Decode(const std::vector<int32_t>& tokens) const {
    return tokenizer->Decode(tokens);
  }
  
  std::shared_ptr<Tensor> EncodeBatch(const std::vector<std::string>& texts) const {
    std::vector<const char*> c_strs;
    c_strs.reserve(texts.size());
    for (const auto& text : texts) {
      c_strs.push_back(text.c_str());
    }
    return tokenizer->EncodeBatch(c_strs);
  }
  
  nb::ndarray<nb::numpy, int32_t> EncodeBatchAsNumpy(const std::vector<std::string>& texts) const {
    // Manually encode each text
    std::vector<std::vector<int32_t>> sequences;
    size_t max_length = 0;
    for (const auto& text : texts) {
      auto seq = tokenizer->Encode(text.c_str());
      max_length = std::max(max_length, seq.size());
      sequences.push_back(std::move(seq));
    }
    
    // Pad sequences to max_length
    int32_t pad_token = tokenizer->GetPadTokenId();
    for (auto& seq : sequences) {
      while (seq.size() < max_length) {
        seq.push_back(pad_token);
      }
    }
    
    // Create numpy array
    size_t batch_size = sequences.size();
    size_t total_elements = batch_size * max_length;
    int32_t* data_ptr = new int32_t[total_elements];
    
    // Copy data row by row
    for (size_t i = 0; i < batch_size; ++i) {
      std::memcpy(data_ptr + i * max_length, sequences[i].data(), max_length * sizeof(int32_t));
    }
    
    nb::capsule owner(data_ptr, [](void* p) noexcept {
      delete[] static_cast<int32_t*>(p);
    });
    
    size_t shape[2] = {batch_size, max_length};
    
    return nb::ndarray<nb::numpy, int32_t>(
      data_ptr, 2, shape, owner, nullptr, nb::dtype<int32_t>()
    );
  }
  
  std::vector<std::string> DecodeBatch(nb::ndarray<> array) const {
    // Get shape and data from numpy array
    if (array.ndim() != 2) {
      throw std::runtime_error("DecodeBatch expects a 2D array");
    }
    
    size_t batch_size = array.shape(0);
    size_t seq_len = array.shape(1);
    
    // Get data pointer
    const int32_t* data = static_cast<const int32_t*>(array.data());
    
    return tokenizer->DecodeBatch(std::span<const int32_t>(data, batch_size * seq_len), batch_size);
  }
  
  std::string ApplyChatTemplate(std::string_view messages, bool add_generation_prompt = true, 
                                 std::optional<std::string_view> template_str = std::nullopt) const {
    // phi3-qa.py uses: tokenizer.apply_chat_template(json.dumps(input_message), add_generation_prompt=True)
    // The messages is already a JSON string
    std::string messages_str(messages);
    std::string template_storage;
    const char* template_c_str = nullptr;
    if (template_str) {
      template_storage = std::string(*template_str);
      template_c_str = template_storage.c_str();
    }
    return tokenizer->ApplyChatTemplate(template_c_str, messages_str.c_str(), nullptr, add_generation_prompt);
  }
  
  int32_t GetBosTokenId() const {
    return tokenizer->GetBosTokenId();
  }
  
  const std::vector<int32_t>& GetEosTokenIds() const {
    return tokenizer->GetEosTokenIds();
  }
  
  int32_t GetPadTokenId() const {
    return tokenizer->GetPadTokenId();
  }
};

void BindTokenizer(nb::module_& m) {
  // TokenizerStream
  nb::class_<PyTokenizerStream>(m, "TokenizerStream")
    .def("decode", &PyTokenizerStream::Decode, "token"_a,
         nb::rv_policy::reference_internal);  // Return reference to internal string
  
  // Tokenizer
  nb::class_<PyTokenizer>(m, "Tokenizer")
    .def(nb::init<PyModel&>(), "model"_a)
    .def("create_stream", &PyTokenizer::CreateStream)
    .def("encode", &PyTokenizer::Encode, "text"_a)
    .def("decode", &PyTokenizer::Decode, "tokens"_a)
    .def("encode_batch", &PyTokenizer::EncodeBatchAsNumpy, "texts"_a)
    .def("decode_batch", &PyTokenizer::DecodeBatch, "array"_a)
    .def("apply_chat_template", 
         [](const PyTokenizer& self, std::string_view messages, bool add_generation_prompt, 
            nb::object template_str) {
           if (template_str.is_none()) {
             return self.ApplyChatTemplate(messages, add_generation_prompt, std::nullopt);
           } else {
             std::string template_string = nb::cast<std::string>(template_str);
             return self.ApplyChatTemplate(messages, add_generation_prompt, std::string_view(template_string));
           }
         }, 
         "messages"_a, "add_generation_prompt"_a = true, "template_str"_a = nb::none())
    .def_prop_ro("bos_token_id", &PyTokenizer::GetBosTokenId)
    .def_prop_ro("eos_token_ids", &PyTokenizer::GetEosTokenIds)
    .def_prop_ro("pad_token_id", &PyTokenizer::GetPadTokenId);
}

} // namespace Generators
