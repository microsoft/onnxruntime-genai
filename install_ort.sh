#!/bin/bash

mkdir -p ort/

cp /home/aciddelgado/ort_always_main/include/onnxruntime/core/session/onnxruntime_c_api.h ort/
cp /home/aciddelgado/ort_always_main/build/Linux/RelWithDebInfo/libonnxruntime*.so* ort/
