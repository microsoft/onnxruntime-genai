// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../json.h"
#include "model_package.h"
#include "onnxruntime_api.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace Generators {

OrtEnv& GetOrtEnv();

namespace {

// ---- fs::path <-> std::filesystem::path bridge -------------------------------
//
// The codebase's `fs::path` (src/filesystem.h) is the public surface; it
// treats narrow strings as UTF-8 and uses `MultiByteToWideChar(CP_UTF8, ...)`
// on Windows. `std::filesystem::path`'s narrow ctor on Windows interprets
// its input as the *active code page*. To keep encoding correct for non-ASCII
// paths we must round-trip through the wide form on Windows; on Unix the
// native narrow encoding IS UTF-8 and the simple ctor is correct.

std::filesystem::path ToStd(const fs::path& p) {
#ifdef _WIN32
  return std::filesystem::path(p.c_str());
#else
  return std::filesystem::path(p.string());
#endif
}

fs::path FromStd(const std::filesystem::path& p) {
#ifdef _WIN32
  const std::wstring w = p.wstring();
  if (w.empty()) return fs::path{};
  const int needed = WideCharToMultiByte(
      CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
      nullptr, 0, nullptr, nullptr);
  std::string utf8(static_cast<std::size_t>(needed), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                      utf8.data(), needed, nullptr, nullptr);
  return fs::path(utf8);
#else
  return fs::path(p.string());
#endif
}

// ---- helpers ----------------------------------------------------------------

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("model package: failed to open " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Reject path fragments that came from package JSON if they could escape
// the package root or hit an absolute / drive-qualified path. Names like
// "decoder", "cpu", "abc123def" are allowed. Names like "..", "../foo",
// "/etc/passwd", "C:\\windows", "a/b" are rejected.
void ValidatePathFragment(std::string_view fragment, std::string_view kind) {
  if (fragment.empty()) {
    throw std::runtime_error(std::string("model package: empty ") + std::string(kind));
  }
  if (fragment == "." || fragment == "..") {
    throw std::runtime_error(
        std::string("model package: invalid ") + std::string(kind) + " '" +
        std::string(fragment) + "' (relative-path token)");
  }
  for (char c : fragment) {
    if (c == '/' || c == '\\') {
      throw std::runtime_error(
          std::string("model package: invalid ") + std::string(kind) + " '" +
          std::string(fragment) + "' (contains path separator)");
    }
  }
  // Catch ".." substring even when joined with separators — defence in depth.
  if (fragment.find("..") != std::string_view::npos) {
    throw std::runtime_error(
        std::string("model package: invalid ") + std::string(kind) + " '" +
        std::string(fragment) + "' (contains '..')");
  }
  // Reject Windows drive prefixes and UNC paths even on non-Windows builds —
  // a package authored on Windows must not embed drive-rooted names.
  if (fragment.size() >= 2 && fragment[1] == ':') {
    throw std::runtime_error(
        std::string("model package: invalid ") + std::string(kind) + " '" +
        std::string(fragment) + "' (drive-qualified)");
  }
}

// ---- variant.json parsing ---------------------------------------------------

std::string JsonScalarToString(const JSON::Document& d,
                               std::string_view key_name,
                               const std::string& parent_key) {
  if (d.IsString()) return d.AsString();
  if (d.IsNumber()) return JSON::SerializeDocument(d);
  if (d.IsBool()) return d.AsBool() ? "true" : "false";

  throw std::runtime_error(
      "variant.json: '" + std::string(key_name) + "' under '" + parent_key +
      "' must contain scalar (string/number/bool) values");
}

std::map<std::string, std::string> ParseFlatOptionsObject(const JSON::Object& parent,
                                                          std::string_view key_name) {
  auto it = parent.find(std::string(key_name));
  if (it == parent.end() || it->second.IsNull()) return {};
  if (!it->second.IsObject()) {
    throw std::runtime_error("variant.json: '" + std::string(key_name) + "' must be an object");
  }

  std::map<std::string, std::string> result;
  for (const auto& [key, value] : it->second.AsObject()) {
    result.emplace(key, JsonScalarToString(value, key_name, key));
  }
  return result;
}

VariantManifest ParseVariantManifestFromText(const std::string& json_text,
                                             const std::filesystem::path& origin) {
  JSON::Document doc;
  try {
    doc = JSON::ParseDocument(json_text);
  } catch (const std::exception& e) {
    throw std::runtime_error("variant.json: parse error in " + origin.string() + ": " + e.what());
  }
  if (!doc.IsObject()) {
    throw std::runtime_error("variant.json: root must be a JSON object (" + origin.string() + ")");
  }
  const auto& root = doc.AsObject();

  VariantManifest result;
  auto files_it = root.find("files");
  if (files_it == root.end()) {
    throw std::runtime_error("variant.json: missing required 'files' array (" + origin.string() + ")");
  }
  if (!files_it->second.IsArray()) {
    throw std::runtime_error("variant.json: 'files' must be an array (" + origin.string() + ")");
  }
  const auto& files_array = files_it->second.AsArray();
  if (files_array.empty()) {
    throw std::runtime_error("variant.json: 'files' must contain at least one entry (" + origin.string() + ")");
  }

  result.files.reserve(files_array.size());
  std::unordered_set<std::string> filenames_seen;
  for (const auto& file_doc : files_array) {
    if (!file_doc.IsObject()) {
      throw std::runtime_error("variant.json: each entry of 'files' must be an object (" + origin.string() + ")");
    }
    const auto& file_obj = file_doc.AsObject();

    VariantFile vf;

    auto fname_it = file_obj.find("filename");
    if (fname_it == file_obj.end() || !fname_it->second.IsString()) {
      throw std::runtime_error("variant.json: each file requires a string 'filename' (" + origin.string() + ")");
    }
    vf.filename = fname_it->second.AsString();
    ValidatePathFragment(vf.filename, "variant file filename");
    if (!filenames_seen.insert(vf.filename).second) {
      throw std::runtime_error(
          "variant.json: duplicate file identifier '" + vf.filename + "' (" + origin.string() + ")");
    }

    vf.session_options = ParseFlatOptionsObject(file_obj, "session_options");
    vf.provider_options = ParseFlatOptionsObject(file_obj, "provider_options");
    if (auto sf_it = file_obj.find("shared_files"); sf_it != file_obj.end()) {
      std::map<std::string, std::string> shared_files = ParseFlatOptionsObject(file_obj, "shared_files");
      for (const auto& [graph_filename, checksum] : shared_files) {
        ValidatePathFragment(checksum, "shared-weight checksum");
        vf.shared_files.emplace(graph_filename, checksum);
      }
    }

    result.files.push_back(std::move(vf));
  }

  return result;
}

bool FileExistsStrict(const std::filesystem::path& p) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(p, ec);
  if (ec) {
    throw std::runtime_error(
        "model package: cannot stat " + p.string() + ": " + ec.message());
  }
  return exists;
}

#if ORT_API_VERSION >= 27

template <typename T>
struct OrtReleaser;

template <>
struct OrtReleaser<OrtSessionOptions> {
  void operator()(OrtSessionOptions* p) const {
    if (p != nullptr) Ort::api->ReleaseSessionOptions(p);
  }
};

template <>
struct OrtReleaser<OrtModelPackageOptions> {
  explicit OrtReleaser(const OrtModelPackageApi* api = nullptr) : api_{api} {}
  void operator()(OrtModelPackageOptions* p) const {
    if (p != nullptr) api_->ReleaseModelPackageOptions(p);
  }
  const OrtModelPackageApi* api_;
};

template <>
struct OrtReleaser<OrtModelPackageContext> {
  explicit OrtReleaser(const OrtModelPackageApi* api = nullptr) : api_{api} {}
  void operator()(OrtModelPackageContext* p) const {
    if (p != nullptr) api_->ReleaseModelPackageContext(p);
  }
  const OrtModelPackageApi* api_;
};

template <>
struct OrtReleaser<OrtModelPackageComponentContext> {
  explicit OrtReleaser(const OrtModelPackageApi* api = nullptr) : api_{api} {}
  void operator()(OrtModelPackageComponentContext* p) const {
    if (p != nullptr) api_->ReleaseModelPackageComponentContext(p);
  }
  const OrtModelPackageApi* api_;
};

template <typename T>
using OrtUniquePtr = std::unique_ptr<T, OrtReleaser<T>>;

const OrtModelPackageApi* GetModelPackageApiOrThrow() {
  const OrtModelPackageApi* api = Ort::api->GetModelPackageApi();
  if (api == nullptr) {
    throw std::runtime_error(
        "v4 model package detected, but the loaded ONNX Runtime does not expose "
        "OrtModelPackageApi. Rebuild/use ONNX Runtime 1.27+ with model-package support.");
  }
  return api;
}

OrtUniquePtr<OrtSessionOptions> CreateSessionOptionsForEp(const EpSelection& ep) {
  OrtSessionOptions* raw_options = nullptr;
  Ort::ThrowOnError(Ort::api->CreateSessionOptions(&raw_options));
  OrtUniquePtr<OrtSessionOptions> options(raw_options);

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t ep_device_count = 0;
  Ort::ThrowOnError(Ort::api->GetEpDevices(&GetOrtEnv(), &ep_devices, &ep_device_count));

  std::vector<const OrtEpDevice*> matching_devices;
  for (size_t i = 0; i < ep_device_count; ++i) {
    const char* ep_name = Ort::api->EpDevice_EpName(ep_devices[i]);
    if (ep_name != nullptr && ep.ep_name == ep_name) {
      matching_devices.push_back(ep_devices[i]);
    }
  }

  if (!matching_devices.empty()) {
    Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider_V2(
        options.get(), &GetOrtEnv(), matching_devices.data(), matching_devices.size(),
        nullptr, nullptr, 0));
    return options;
  }

  if (ep.ep_name == "CPUExecutionProvider") {
    return options;
  }

  const char* const* option_keys = nullptr;
  const char* const* option_values = nullptr;
  Ort::ThrowOnError(Ort::api->SessionOptionsAppendExecutionProvider(
      options.get(), ep.ep_name.c_str(), option_keys, option_values, 0));
  return options;
}

OrtUniquePtr<OrtModelPackageOptions> CreatePackageOptionsForEp(
    const OrtModelPackageApi* api, const EpSelection& ep) {
  auto session_options = CreateSessionOptionsForEp(ep);

  OrtModelPackageOptions* raw_options = nullptr;
  Ort::ThrowOnError(api->CreateModelPackageOptionsFromSessionOptions(
      &GetOrtEnv(), session_options.get(), &raw_options));
  return OrtUniquePtr<OrtModelPackageOptions>(raw_options, OrtReleaser<OrtModelPackageOptions>(api));
}

fs::path OrtCharPathToFsPath(const ORTCHAR_T* path) {
#ifdef _WIN32
  return fs::path(path);
#else
  return fs::path(path == nullptr ? "" : path);
#endif
}

std::vector<std::string> CopyStringArray(const char* const* values, size_t count) {
  std::vector<std::string> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(values[i] == nullptr ? "" : values[i]);
  }
  return result;
}

fs::path ResolveSharedWeightFromComponentDir(const fs::path& component_dir,
                                             std::string_view checksum) {
  ValidatePathFragment(checksum, "shared-weight checksum");
  const std::filesystem::path sw_dir =
      ToStd(component_dir) / "shared_weights" / std::string(checksum);
  std::error_code ec;
  if (!std::filesystem::is_directory(sw_dir, ec) || ec) {
    throw std::runtime_error(
        "model package: shared-weight directory not found: " + sw_dir.string() +
        (ec ? (" (" + ec.message() + ")") : std::string{}));
  }

  std::vector<std::filesystem::path> blobs;
  std::filesystem::directory_iterator it(sw_dir, ec);
  if (ec) {
    throw std::runtime_error(
        "model package: cannot iterate " + sw_dir.string() + ": " + ec.message());
  }
  for (; it != std::filesystem::directory_iterator(); it.increment(ec)) {
    if (ec) {
      throw std::runtime_error(
          "model package: error iterating " + sw_dir.string() + ": " + ec.message());
    }
    const bool is_regular = it->is_regular_file(ec);
    if (ec) {
      throw std::runtime_error(
          "model package: cannot stat " + it->path().string() + ": " + ec.message());
    }
    if (is_regular) {
      blobs.push_back(it->path());
    }
  }
  if (blobs.empty()) {
    throw std::runtime_error(
        "model package: shared-weight directory contains no blob: " + sw_dir.string());
  }
  if (blobs.size() > 1) {
    throw std::runtime_error(
        "model package: shared-weight directory contains multiple blobs: " + sw_dir.string());
  }
  return FromStd(blobs.front());
}

struct OrtApiComponentInstance : ComponentInstance {
  OrtApiComponentInstance(const OrtModelPackageApi* api,
                          OrtUniquePtr<OrtModelPackageOptions> options,
                          OrtModelPackageComponentContext* component_context,
                          std::string selected_ep)
      : api_{api},
        options_{std::move(options)},
        component_context_{component_context, OrtReleaser<OrtModelPackageComponentContext>(api)},
        selected_ep_{std::move(selected_ep)} {
    const ORTCHAR_T* variant_folder_path = nullptr;
    Ort::ThrowOnError(api_->ModelPackageComponent_GetSelectedVariantFolderPath(
        component_context_.get(), &variant_folder_path));
    variant_folder_ = OrtCharPathToFsPath(variant_folder_path);
    component_dir_ = FromStd(ToStd(variant_folder_).parent_path());

    Ort::ThrowOnError(api_->ModelPackageComponent_GetSelectedVariantFileCount(
        component_context_.get(), &file_count_));

    const char* consumer_metadata = nullptr;
    Ort::ThrowOnError(api_->ModelPackageComponent_GetSelectedVariantConsumerMetadata(
        component_context_.get(), &consumer_metadata));
    consumer_metadata_ = consumer_metadata == nullptr ? "" : consumer_metadata;
  }

  fs::path VariantFolderPath() const override { return variant_folder_; }
  std::size_t FileCount() const override { return file_count_; }
  std::string ConsumerMetadata() const override { return consumer_metadata_; }
  std::string SelectedEp() const override { return selected_ep_; }

  fs::path ResolveSharedWeight(std::string_view checksum) const override {
    return ResolveSharedWeightFromComponentDir(component_dir_, checksum);
  }

  const OrtModelPackageApi* api_;
  OrtUniquePtr<OrtModelPackageOptions> options_;
  OrtUniquePtr<OrtModelPackageComponentContext> component_context_;
  fs::path component_dir_;
  fs::path variant_folder_;
  std::size_t file_count_{};
  std::string consumer_metadata_;
  std::string selected_ep_;
};

struct OrtApiModelPackageContext : ModelPackageContext {
  OrtApiModelPackageContext(const OrtModelPackageApi* api, const fs::path& package_root)
      : api_{api},
        package_root_{package_root} {
    OrtModelPackageContext* raw_context = nullptr;
    Ort::ThrowOnError(api_->CreateModelPackageContext(package_root.c_str(), &raw_context));
    context_ = OrtUniquePtr<OrtModelPackageContext>(
        raw_context, OrtReleaser<OrtModelPackageContext>(api_));

    const char* const* names = nullptr;
    size_t count = 0;
    Ort::ThrowOnError(api_->ModelPackage_GetComponentNames(context_.get(), &names, &count));
    component_names_ = CopyStringArray(names, count);
    variant_names_.resize(component_names_.size());
    variant_ep_compatibility_.resize(component_names_.size());
    variants_loaded_.assign(component_names_.size(), false);
  }

  std::size_t NumComponents() const override { return component_names_.size(); }

  std::string ComponentName(std::size_t cix) const override {
    return component_names_.at(cix);
  }

  std::size_t NumVariants(std::size_t cix) const override {
    EnsureVariants(cix);
    return variant_names_.at(cix).size();
  }

  std::string VariantName(std::size_t cix, std::size_t vix) const override {
    EnsureVariants(cix);
    return variant_names_.at(cix).at(vix);
  }

  std::span<const EpCompatibilityEntry> VariantEpCompatibility(
      std::size_t cix, std::size_t vix) const override {
    EnsureVariants(cix);
    const auto& entries = variant_ep_compatibility_.at(cix).at(vix);
    return {entries.data(), entries.size()};
  }

  std::vector<std::string> EpsCompatibleWith(std::size_t cix) const override {
    EnsureVariants(cix);
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    for (const auto& variant_entries : variant_ep_compatibility_.at(cix)) {
      for (const auto& entry : variant_entries) {
        if (seen.insert(entry.ep).second) {
          result.push_back(entry.ep);
        }
      }
    }
    return result;
  }

  std::unique_ptr<ComponentInstance> SelectComponent(
      std::size_t cix, const ModelPackageSelectionOptions& options) const override {
    if (options.ep_priority.empty()) return nullptr;

    const EpSelection& ep = options.ep_priority.front();
    auto package_options = CreatePackageOptionsForEp(api_, ep);

    OrtModelPackageComponentContext* component_context = nullptr;
    Ort::ThrowOnError(api_->SelectComponent(
        context_.get(), component_names_.at(cix).c_str(), package_options.get(), &component_context));

    return std::make_unique<OrtApiComponentInstance>(
        api_, std::move(package_options), component_context, ep.ep_name);
  }

  fs::path SharedAssetsPath() const override {
    return package_root_.join("configs");
  }

  void EnsureVariants(std::size_t cix) const {
    (void)component_names_.at(cix);
    if (variants_loaded_.at(cix)) return;

    const char* const* names = nullptr;
    size_t count = 0;
    Ort::ThrowOnError(api_->ModelPackage_GetVariantNames(
        context_.get(), component_names_[cix].c_str(), &names, &count));
    variant_names_[cix] = CopyStringArray(names, count);

    auto& component_compatibility = variant_ep_compatibility_[cix];
    component_compatibility.clear();
    component_compatibility.resize(variant_names_[cix].size());

    for (size_t vix = 0; vix < variant_names_[cix].size(); ++vix) {
      size_t compat_count = 0;
      Ort::ThrowOnError(api_->ModelPackage_GetVariantEpCompatibilityCount(
          context_.get(), component_names_[cix].c_str(), variant_names_[cix][vix].c_str(),
          &compat_count));

      auto& entries = component_compatibility[vix];
      entries.reserve(compat_count);
      for (size_t eix = 0; eix < compat_count; ++eix) {
        const char* ep = nullptr;
        const char* device = nullptr;
        const char* compatibility_string = nullptr;
        Ort::ThrowOnError(api_->ModelPackage_GetVariantEpCompatibility(
            context_.get(), component_names_[cix].c_str(), variant_names_[cix][vix].c_str(), eix,
            &ep, &device, &compatibility_string));

        EpCompatibilityEntry entry;
        entry.ep = ep == nullptr ? "" : ep;
        if (device != nullptr) {
          entry.device = device;
        }
        if (compatibility_string != nullptr) {
          entry.compatibility_string = compatibility_string;
        }
        entries.push_back(std::move(entry));
      }
    }

    variants_loaded_[cix] = true;
  }

  const OrtModelPackageApi* api_;
  fs::path package_root_;
  OrtUniquePtr<OrtModelPackageContext> context_{nullptr, OrtReleaser<OrtModelPackageContext>{}};
  std::vector<std::string> component_names_;
  mutable std::vector<std::vector<std::string>> variant_names_;
  mutable std::vector<std::vector<std::vector<EpCompatibilityEntry>>> variant_ep_compatibility_;
  mutable std::vector<bool> variants_loaded_;
};

#endif  // ORT_API_VERSION >= 27

}  // namespace

VariantManifest ParseVariantManifest(const fs::path& variant_folder) {
  const std::filesystem::path origin = ToStd(variant_folder) / "variant.json";
  if (!FileExistsStrict(origin)) {
    throw std::runtime_error("model package: missing variant.json at " + origin.string());
  }
  return ParseVariantManifestFromText(ReadFile(origin), origin);
}

std::unique_ptr<ModelPackageContext> ModelPackageContext::Open(const fs::path& path) {
  const std::filesystem::path root = ToStd(path);

  // Detection.
  const std::filesystem::path root_metadata_path = root / "metadata.json";
  const std::filesystem::path manifest_path = root / "manifest.json";
  const bool has_root_metadata = FileExistsStrict(root_metadata_path);
  const bool has_manifest = FileExistsStrict(manifest_path);
  if (!has_root_metadata && !has_manifest) {
    return nullptr;
  }

#if ORT_API_VERSION >= 27
  (void)GetOrtEnv();
  return std::make_unique<OrtApiModelPackageContext>(GetModelPackageApiOrThrow(), path);
#else
  throw std::runtime_error(
      "v4 model package detected, but this ONNX Runtime GenAI build was compiled "
      "against ONNX Runtime headers without OrtModelPackageApi (requires ORT API "
      "version 27 / ONNX Runtime 1.27+). GenAI no longer falls back to a private "
      "model-package parser; rebuild GenAI against an ONNX Runtime that provides "
      "OrtApi::GetModelPackageApi().");
#endif
}

}  // namespace Generators
