Pod::Spec.new do |s|
  s.name	  = "onnxruntime-genai-objc"
  s.version	= "0.5.0"
  s.summary = "SUMMARY"
  s.homepage = "https://github.com/microsoft/onnxruntime-genai"
  s.license	= { :type => "MIT" }
  s.author  = {"ONNX Runtime" => "onnxruntime@microsoft.com"}
  s.source  = { :http => 'file:///Users/chester/Projects/onnxruntime-genai/src/archive.zip'}
  s.platform = :ios, '13.0', :osx, '11.0'

  s.default_subspec  = 'Core'
  s.static_framework = true

  s.subspec 'Core' do |sp|
    sp.requires_arc = true
    sp.public_header_files = [
      "objectivec/include/ort_genai_objc.h"
    ]
    sp.source_files = [
      "objectivec/include/ort_genai_objc.h",
      "objectivec/cxx_api.h",
      "objectivec/error_utils.h",
      "objectivec/error_utils.mm",
      "objectivec/oga_internal.h",
      "objectivec/oga_internal.mm",
      "objectivec/oga_model.mm",
      "objectivec/oga_tokenizer.mm",
      "objectivec/oga_sequences.mm",
      "objectivec/oga_generator.mm",
      "objectivec/oga_generator_params.mm",
      "objectivec/oga_multi_modal_processor.mm",
      "objectivec/oga_named_tensors.mm",
      "objectivec/oga_tensor.mm",
      "objectivec/oga_images.mm",
    ]

  end
  s.dependency 'onnxruntime-objc', '~> 1.19.0'
end
  
