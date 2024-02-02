#!/bin/bash

mkdir -p ort/

cp /(please change)/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h ort/include/
cp /(please change)/onnxruntime/build/Linux/Release/libonnxruntime*.so* ort/lib/
