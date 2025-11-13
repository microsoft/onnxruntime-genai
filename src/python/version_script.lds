# Linux version script for onnxruntime-genai Python extension
# Only export the Python module initialization function
VERS_1.0 {
  global:
    PyInit_onnxruntime_genai;

  # Hide everything else (including all C++ symbols)
  local:
    *;
};
