Pod::Spec.new do |s|
  s.name	  = "onnxruntime-genai-c"
  s.version	= "0.5.0"
  s.summary = "SUMMARY"
  s.homepage = "https://github.com/microsoft/onnxruntime-genai"
  s.license	= { :type => "MIT" }
  s.author  = {"ONNX Runtime" => "onnxruntime@microsoft.com"}
  s.source  = { :http => 'file:///Users/chester/Projects/onnxruntime-genai/build/apple_framework/framework_out/archive.zip'}
  s.platform = :ios, '13.0', :osx, '11.0'
  
  s.vendored_frameworks = "onnxruntime-genai.xcframework"
  s.static_framework = true

  s.source_files = [
    "Headers/*.h"
  ]

  s.libraries = "c++"
  s.dependency 'onnxruntime-c', '~> 1.19.0'
end
  
