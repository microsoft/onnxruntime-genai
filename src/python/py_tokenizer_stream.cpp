// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "py_utils.h"
#include "../models/model.h"

namespace nb = nanobind;

namespace Generators {

void BindTokenizerStream(nb::module_& m) {
  nb::class_<TokenizerStream>(m, "TokenizerStream")
    .def("decode", [](TokenizerStream& self, int32_t token) -> const std::string& {
      return self.Decode(token);
    }, nb::rv_policy::reference_internal);
}

} // namespace Generators
