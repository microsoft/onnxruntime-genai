// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/intrusive/counter.h>
#include "../../ort_genai_c.h"

namespace nb = nanobind;

namespace OgaPy {

// Base class for all C API wrapper objects with intrusive reference counting
struct OgaObject {
  void inc_ref() noexcept { m_ref_count.inc_ref(); }
  bool dec_ref() noexcept { return m_ref_count.dec_ref(); }
  void set_self_py(PyObject *self) noexcept { m_ref_count.set_self_py(self); }
  
  virtual ~OgaObject() = default;

private:
  nb::intrusive_counter m_ref_count;
};

// Convenience functions for intrusive reference counting
inline void intrusive_inc_ref(OgaObject* o) noexcept {
  if (o) o->inc_ref();
}

inline void intrusive_dec_ref(OgaObject* o) noexcept {
  if (o && o->dec_ref())
    delete o;
}

} // namespace OgaPy
