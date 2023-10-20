#!/bin/bash

mkdir -p ort/

cp /bert_ort/kvaishnavi/onnxruntime/include/onnxruntime/core/session/onnxruntime_c*.h ort/
cp /bert_ort/kvaishnavi/onnxruntime/build/Linux/Release/libonnxruntime*.so* ort/