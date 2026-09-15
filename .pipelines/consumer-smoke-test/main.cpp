// Links against the imported target onnxruntime-genai::onnxruntime-genai to
// prove the exported CMake package resolves headers and libraries correctly.
#include "ort_genai_c.h"

int main() {
  OgaShutdown();
  return 0;
}
