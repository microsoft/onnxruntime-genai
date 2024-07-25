#include "onnxruntime_c_api.h"
#include "ocos.h"
#include <string>
#include <vector>

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib2;
extern OrtOpLoader genai_op_loader;

static int GetOrtVersion(const OrtApiBase* api_base = nullptr) noexcept {
  // the version will be cached after the first call on RegisterCustomOps
  static int ort_version = 17;  // the default version is 1.17.0

  if (api_base != nullptr) {
    std::string str_version = api_base->GetVersionString();

    std::size_t first_dot = str_version.find('.');
    if (first_dot != std::string::npos) {
      std::size_t second_dot = str_version.find('.', first_dot + 1);
      // If there is no second dot and the string has more than one character after the first dot, set second_dot to the string length
      if (second_dot == std::string::npos && first_dot + 1 < str_version.length()) {
        second_dot = str_version.length();
      }

      if (second_dot != std::string::npos) {
        std::string str_minor_version = str_version.substr(first_dot + 1, second_dot - first_dot - 1);
        int ver = std::atoi(str_minor_version.c_str());
        // Only change ort_version if conversion is successful (non-zero value)
        if (ver != 0) {
          ort_version = ver;
        }
      }
    }
  }

  return ort_version;
}

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtStatus* status = nullptr;

  // the following will initiate some global objects which
  //  means any other function invocatoin prior to these calls to trigger undefined behavior.
  auto ver = GetOrtVersion(api);
  const OrtApi* ortApi = api->GetApi(ver);
  OrtW::API::instance(ortApi);

  OrtCustomOpDomain* domain = nullptr;

  if (status = ortApi->CreateCustomOpDomain("onnx.genai", &domain); status) {
    return status;
  }

  for (size_t i = 0; i < genai_op_loader.GetCustomOps().size(); i++) {
    if (status = ortApi->CustomOpDomain_Add(domain, genai_op_loader.GetCustomOps().at(i)); status) {
      return status;
    }
  }
  
  if (status = ortApi->AddCustomOpDomain(options, domain); status) {
    return status;
  }

  return status;
}

extern "C" int ORT_API_CALL GetActiveOrtAPIVersion() {
  int ver = 0;
  ver = GetOrtVersion();
  return ver;
}