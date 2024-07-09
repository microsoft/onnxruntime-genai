
Pod::Spec.new do |s|
  s.name	= "onnxruntime-genai-objc"
  s.version	= "0.4.0"
  s.summary = "SUMMARY"
  s.homepage = "https://github.com/microsoft/onnxruntime-genai"
  s.license	= { :type => "MIT" }
  s.author  = {"ONNX Runtime" => "onnxruntime@microsoft.com"}
  s.source  = { :path => './objectivec'}
  
  s.static_framework = true
  s.public_header_files = [
    "objectivec/include/ort_genai_objc.h"
  ]

  s.source_files = [
    "objectivec/include/ort_genai_objc.h",
    "objectivec/cxx_api.h",
    "objectivec/oga_model.mm",
    "objectivec/error_utils.h",
    "objectivec/error_utils.mm",
    "src/ort_genai.h"
  ]

  s.dependency 'onnxruntime-objc', '~> 1.18.0'
end
  
