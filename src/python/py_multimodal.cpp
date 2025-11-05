// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "py_named_tensors.h"
#include "../generators.h"
#include "../models/model.h"
#include "../models/processor.h"
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace Generators {

// Wrapper for MultiModalProcessor
struct PyMultiModalProcessor {
  std::shared_ptr<MultiModalProcessor> processor;
  
  PyMultiModalProcessor(std::shared_ptr<MultiModalProcessor> proc)
    : processor(std::move(proc)) {}
  
  std::unique_ptr<PyNamedTensors> Process(const std::string& prompt,
                                          const Images* images,
                                          const Audios* audios) const {
    auto cpp_result = processor->Process(prompt, images, audios);
    return ConvertNamedTensors(std::move(cpp_result));
  }
  
  std::unique_ptr<PyNamedTensors> ProcessBatch(const std::vector<std::string>& prompts,
                                               const Images* images,
                                               const Audios* audios) const {
    std::vector<const char*> prompt_ptrs;
    prompt_ptrs.reserve(prompts.size());
    for (const auto& prompt : prompts) {
      prompt_ptrs.push_back(prompt.c_str());
    }
    auto cpp_result = processor->Process(std::span<const char*>(prompt_ptrs.data(), prompt_ptrs.size()), images, audios);
    return ConvertNamedTensors(std::move(cpp_result));
  }
};

void BindMultiModal(nb::module_& m) {
  // Images class
  nb::class_<Images>(m, "Images")
    .def_static("open", [](nb::args args) -> std::unique_ptr<Images> {
      std::vector<const char*> paths;
      paths.reserve(args.size());
      std::vector<std::string> path_storage;  // Keep strings alive
      path_storage.reserve(args.size());
      
      for (auto arg : args) {
        path_storage.emplace_back(nb::cast<std::string>(arg));
        paths.push_back(path_storage.back().c_str());
      }
      
      return LoadImages(std::span<const char* const>(paths.data(), paths.size()));
    })
    .def_static("open_bytes", [](nb::bytes image_data) -> std::unique_ptr<Images> {
      std::vector<const void*> data_ptrs{image_data.c_str()};
      std::vector<size_t> data_sizes{image_data.size()};
      
      return LoadImagesFromBuffers(
        std::span<const void*>(data_ptrs.data(), data_ptrs.size()),
        std::span<const size_t>(data_sizes.data(), data_sizes.size())
      );
    }, nb::arg("image_data"));
  
  // Audios class
  nb::class_<Audios>(m, "Audios")
    .def_static("open", [](nb::args args) -> std::unique_ptr<Audios> {
      std::vector<const char*> paths;
      paths.reserve(args.size());
      std::vector<std::string> path_storage;  // Keep strings alive
      path_storage.reserve(args.size());
      
      for (auto arg : args) {
        path_storage.emplace_back(nb::cast<std::string>(arg));
        paths.push_back(path_storage.back().c_str());
      }
      
      return LoadAudios(std::span<const char* const>(paths.data(), paths.size()));
    });
  
  // MultiModalProcessor class - wrapped
  nb::class_<PyMultiModalProcessor>(m, "MultiModalProcessor")
    .def("__call__", &PyMultiModalProcessor::Process,
         nb::arg("prompt"),
         nb::arg("images") = nullptr,
         nb::arg("audios") = nullptr,
         nb::rv_policy::take_ownership)
    .def("__call__", &PyMultiModalProcessor::ProcessBatch,
         nb::arg("prompts"),
         nb::arg("images") = nullptr,
         nb::arg("audios") = nullptr,
         nb::rv_policy::take_ownership);
}

} // namespace Generators
