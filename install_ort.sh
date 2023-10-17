#!/bin/bash

mkdir -p src/ort

cp ~/onnxruntime/include/onnxruntime/core/session/onnxruntime_c*.h src/ort/
cp ~/onnxruntime/build/Linux/Release/libonnxruntime*.so* src/ort/