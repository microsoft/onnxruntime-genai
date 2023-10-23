#!/bin/bash

mkdir -p ort/

cp /workspace/kvaishnavi/onnxruntime/include/onnxruntime/core/session/onnxruntime_c*.h ort/
cp /workspace/kvaishnavi/onnxruntime/build/Linux/Release/libonnxruntime*.so* ort/