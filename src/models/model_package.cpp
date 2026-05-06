// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../json.h"
#include "model_package.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <unordered_set>

namespace Generators {

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

JSON::Object DocumentToObject(const JSON::Document& d) {
  if (d.IsObject()) return d.AsObject();
  if (d.IsNull()) return {};
  throw std::runtime_error("variant.json: expected object");
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
  result.files.reserve(files_array.size());
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

    if (auto so_it = file_obj.find("session_options"); so_it != file_obj.end()) {
      vf.session_options = DocumentToObject(so_it->second);
    }
    if (auto po_it = file_obj.find("provider_options"); po_it != file_obj.end()) {
      vf.provider_options = DocumentToObject(po_it->second);
    }
    if (auto sf_it = file_obj.find("shared_files"); sf_it != file_obj.end()) {
      if (!sf_it->second.IsObject()) {
        throw std::runtime_error("variant.json: 'shared_files' must be an object (" + origin.string() + ")");
      }
      for (const auto& [graph_filename, checksum_doc] : sf_it->second.AsObject()) {
        if (!checksum_doc.IsString()) {
          throw std::runtime_error("variant.json: 'shared_files' values must be strings (" + origin.string() + ")");
        }
        const std::string& checksum = checksum_doc.AsString();
        ValidatePathFragment(checksum, "shared-weight checksum");
        vf.shared_files.emplace(graph_filename, checksum);
      }
    }

    result.files.push_back(std::move(vf));
  }

  return result;
}

std::string ExtractConsumerMetadataFromText(const std::string& json_text,
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
  auto cm_it = root.find("consumer_metadata");
  if (cm_it == root.end()) {
    return {};
  }
  return JSON::SerializeDocument(cm_it->second);
}

// ---- metadata.json streaming parser (preserves variant declaration order) ---
//
// Spec shape:
//   {
//     "variants": {
//       "<name>": {
//         "ep_compatibility": [
//           { "ep": "...", "device": "...?", "compatibility": ["..."] }
//         ]
//       }
//     }
//   }
//
// `JSON::Object` is `std::map`, which doesn't preserve insertion order. The
// streaming parser DOES — `OnObject(name)` fires per key in source order — so
// we use the streaming Element interface here even though the rest of the
// model_package code uses the DOM.

struct CompatStrings_Element : JSON::Element {
  std::vector<std::string>* out = nullptr;
  void OnValue(std::string_view /*name*/, JSON::Value value) override {
    out->push_back(std::string(JSON::Get<std::string_view>(value)));
  }
};

struct EpCompatibilityEntry_Element : JSON::Element {
  EpCompatibilityEntry entry;
  CompatStrings_Element compat_;

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "ep") {
      entry.ep = std::string(JSON::Get<std::string_view>(value));
    } else if (name == "device") {
      entry.device = std::string(JSON::Get<std::string_view>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  JSON::Element& OnArray(std::string_view name) override {
    if (name == "compatibility") {
      compat_.out = &entry.compatibility;
      return compat_;
    }
    throw JSON::unknown_value_error{};
  }
};

struct EpCompatibilityArray_Element : JSON::Element {
  std::vector<EpCompatibilityEntry>* out = nullptr;
  std::vector<std::unique_ptr<EpCompatibilityEntry_Element>> sub_;

  JSON::Element& OnObject(std::string_view /*name*/) override {
    sub_.emplace_back(std::make_unique<EpCompatibilityEntry_Element>());
    return *sub_.back();
  }
  void OnComplete(bool /*empty*/) override {
    out->reserve(sub_.size());
    for (auto& entry : sub_) {
      if (entry->entry.ep.empty()) {
        throw std::runtime_error("metadata.json: ep_compatibility entry missing 'ep'");
      }
      out->push_back(std::move(entry->entry));
    }
  }
};

struct VariantEntry_Element : JSON::Element {
  std::vector<EpCompatibilityEntry> ep_compatibility;
  EpCompatibilityArray_Element compat_array_;

  JSON::Element& OnArray(std::string_view name) override {
    if (name == "ep_compatibility") {
      compat_array_.out = &ep_compatibility;
      return compat_array_;
    }
    throw JSON::unknown_value_error{};
  }
};

struct ParsedVariantInfo {
  std::string name;
  std::vector<EpCompatibilityEntry> ep_compatibility;
};

struct VariantsObject_Element : JSON::Element {
  std::vector<ParsedVariantInfo>* out = nullptr;
  std::vector<std::unique_ptr<VariantEntry_Element>> sub_;
  std::vector<std::string> names_;

  JSON::Element& OnObject(std::string_view name) override {
    ValidatePathFragment(name, "variant name");
    sub_.emplace_back(std::make_unique<VariantEntry_Element>());
    names_.emplace_back(name);
    return *sub_.back();
  }
  void OnComplete(bool /*empty*/) override {
    out->reserve(sub_.size());
    std::unordered_set<std::string> seen;
    for (std::size_t i = 0; i < sub_.size(); ++i) {
      const std::string& nm = names_[i];
      if (!seen.insert(nm).second) {
        throw std::runtime_error("metadata.json: duplicate variant name '" + nm + "'");
      }
      ParsedVariantInfo info;
      info.name = nm;
      info.ep_compatibility = std::move(sub_[i]->ep_compatibility);
      out->push_back(std::move(info));
    }
  }
};

struct MetadataRoot_Element : JSON::Element {
  std::vector<ParsedVariantInfo>* out = nullptr;
  VariantsObject_Element variants_;

  JSON::Element& OnObject(std::string_view name) override {
    if (name == "variants") {
      variants_.out = out;
      return variants_;
    }
    throw JSON::unknown_value_error{};
  }
};

std::vector<ParsedVariantInfo> ParseMetadataJson(const std::string& json_text,
                                                 const std::filesystem::path& origin) {
  std::vector<ParsedVariantInfo> variants;
  MetadataRoot_Element root;
  root.out = &variants;
  try {
    JSON::Parse(root, json_text);
  } catch (const std::exception& e) {
    throw std::runtime_error("metadata.json: parse error in " + origin.string() + ": " + e.what());
  }
  if (variants.empty()) {
    throw std::runtime_error("metadata.json: 'variants' map is empty or missing (" + origin.string() + ")");
  }
  return variants;
}

// ---- manifest.json parsing --------------------------------------------------

struct ParsedManifest {
  std::optional<std::vector<std::string>> components;  // nullopt if not declared
};

ParsedManifest ParseManifestJson(const std::string& json_text,
                                 const std::filesystem::path& origin) {
  JSON::Document doc;
  try {
    doc = JSON::ParseDocument(json_text);
  } catch (const std::exception& e) {
    throw std::runtime_error("manifest.json: parse error in " + origin.string() + ": " + e.what());
  }
  if (!doc.IsObject()) {
    throw std::runtime_error("manifest.json: root must be a JSON object (" + origin.string() + ")");
  }
  const auto& root = doc.AsObject();

  // Reject schema versions we don't recognize. Absent is fine — older
  // packages simply omit the field. The spec defines `schema_version` as a
  // string, but historic producers wrote a number; we coerce numbers into
  // their canonical decimal-string form so both spellings are accepted.
  if (auto sv_it = root.find("schema_version"); sv_it != root.end()) {
    std::string sv_str;
    if (sv_it->second.IsString()) {
      sv_str = sv_it->second.AsString();
    } else if (sv_it->second.IsNumber()) {
      const double n = sv_it->second.AsNumber();
      // 1 / 1.0 both mean schema v1. Reject anything that isn't an integer
      // value to keep the recognised set tight.
      if (std::floor(n) != n || n < 0) {
        throw std::runtime_error(
            "manifest.json: unsupported schema_version " + std::to_string(n) +
            " (this build supports \"1\")");
      }
      sv_str = std::to_string(static_cast<long long>(n));
    } else {
      throw std::runtime_error(
          "manifest.json: 'schema_version' must be a string (" + origin.string() + ")");
    }
    // Tolerate "1" and "1.0" — both refer to the v1 format.
    if (sv_str != "1" && sv_str != "1.0") {
      throw std::runtime_error(
          "manifest.json: unsupported schema_version \"" + sv_str +
          "\" (this build supports \"1\")");
    }
  }

  ParsedManifest result;
  if (auto comps_it = root.find("components"); comps_it != root.end()) {
    if (!comps_it->second.IsArray()) {
      throw std::runtime_error("manifest.json: 'components' must be an array of strings (" + origin.string() + ")");
    }
    std::vector<std::string> names;
    std::unordered_set<std::string> seen;
    names.reserve(comps_it->second.AsArray().size());
    for (const auto& comp : comps_it->second.AsArray()) {
      if (!comp.IsString()) {
        throw std::runtime_error(
            "manifest.json: 'components' entries must be strings (e.g. \"decoder\"). "
            "Object-form entries (e.g. {\"name\":\"decoder\"}) are not supported — "
            "the on-disk layout is conventional, no per-component metadata path is needed (" +
            origin.string() + ")");
      }
      const std::string& name = comp.AsString();
      ValidatePathFragment(name, "component name");
      if (!seen.insert(name).second) {
        throw std::runtime_error("manifest.json: duplicate component name '" + name + "'");
      }
      names.push_back(name);
    }
    result.components = std::move(names);
  }
  return result;
}

// ---- stub backend -----------------------------------------------------------

struct StubVariant {
  std::string name;
  fs::path folder;  // <package>/<component>/<variant>
  std::vector<EpCompatibilityEntry> ep_compatibility;
  std::size_t file_count = 0;        // from variant.json `files[]`
  std::string consumer_metadata;     // serialized JSON, "" if absent
};

struct StubComponent {
  std::string name;
  fs::path component_dir;            // <package>/<component>
  std::vector<StubVariant> variants;
};

struct StubComponentInstance : ComponentInstance {
  fs::path component_dir;
  fs::path variant_folder;
  std::size_t file_count = 0;
  std::string consumer_metadata_blob;
  std::string selected_ep;

  fs::path VariantFolderPath() const override { return variant_folder; }
  std::size_t FileCount() const override { return file_count; }
  std::string ConsumerMetadata() const override { return consumer_metadata_blob; }
  std::string SelectedEp() const override { return selected_ep; }

  fs::path ResolveSharedWeight(std::string_view checksum) const override {
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
};

struct StubModelPackageContext : ModelPackageContext {
  fs::path package_root;
  std::vector<StubComponent> components;

  std::size_t NumComponents() const override { return components.size(); }
  std::string ComponentName(std::size_t cix) const override { return components.at(cix).name; }
  std::size_t NumVariants(std::size_t cix) const override { return components.at(cix).variants.size(); }
  std::string VariantName(std::size_t cix, std::size_t vix) const override {
    return components.at(cix).variants.at(vix).name;
  }
  std::span<const EpCompatibilityEntry> VariantEpCompatibility(
      std::size_t cix, std::size_t vix) const override {
    const auto& entries = components.at(cix).variants.at(vix).ep_compatibility;
    return {entries.data(), entries.size()};
  }

  std::vector<std::string> EpsCompatibleWith(std::size_t cix) const override {
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    for (const auto& variant : components.at(cix).variants) {
      for (const auto& entry : variant.ep_compatibility) {
        if (seen.insert(entry.ep).second) {
          result.push_back(entry.ep);
        }
      }
    }
    return result;
  }

  std::unique_ptr<ComponentInstance> SelectComponent(
      std::size_t cix, const ModelPackageSelectionOptions& options) const override {
    const StubComponent& component = components.at(cix);

    // Spec: empty captured EP list defaults to [{CPUExecutionProvider}].
    static const std::vector<EpSelection> kCpuFallback = {{"CPUExecutionProvider", std::nullopt}};
    const std::vector<EpSelection>& priority =
        options.ep_priority.empty() ? kCpuFallback : options.ep_priority;

    // Score = (priority index of best matching EP, declaration index of variant).
    // Smaller score wins.
    constexpr std::size_t kNoMatch = static_cast<std::size_t>(-1);
    std::size_t best_priority = kNoMatch;
    std::size_t best_variant_index = kNoMatch;

    for (std::size_t vix = 0; vix < component.variants.size(); ++vix) {
      const auto& variant = component.variants[vix];
      // Find the best (= smallest-index) priority entry that matches any
      // of this variant's compat entries.
      std::size_t variant_priority = kNoMatch;
      for (const auto& compat : variant.ep_compatibility) {
        for (std::size_t pi = 0; pi < priority.size(); ++pi) {
          if (priority[pi].ep_name != compat.ep) continue;
          // If the package pins a device for this entry, the caller must
          // request the same device. An unpinned caller does NOT match a
          // device-pinned compat entry — selecting an NPU-only variant
          // when the caller didn't ask for NPU would be unsafe.
          if (compat.device.has_value()) {
            if (!priority[pi].device.has_value() || *compat.device != *priority[pi].device) {
              continue;
            }
          }
          if (pi < variant_priority) {
            variant_priority = pi;
          }
        }
      }
      if (variant_priority == kNoMatch) continue;
      if (variant_priority < best_priority) {
        best_priority = variant_priority;
        best_variant_index = vix;
      }
      // Equal priority: tie-break by declaration order, which means we
      // keep the earlier variant (already in best_variant_index).
    }

    if (best_variant_index == kNoMatch) return nullptr;

    const auto& chosen = component.variants[best_variant_index];
    auto inst = std::make_unique<StubComponentInstance>();
    inst->component_dir = component.component_dir;
    inst->variant_folder = chosen.folder;
    inst->file_count = chosen.file_count;
    inst->consumer_metadata_blob = chosen.consumer_metadata;
    inst->selected_ep = priority[best_priority].ep_name;
    return inst;
  }

  fs::path SharedAssetsPath() const override {
    return package_root.join("configs");
  }
};

// ---- Open() detection + loading ---------------------------------------------

// Direct child names a v4 package treats as reserved (i.e. NOT components).
bool IsReservedChild(const std::string& name) {
  if (name == "configs") return true;
  if (!name.empty() && (name[0] == '.' || name[0] == '_')) return true;
  return false;
}

// Returns the list of immediate child directories of `root`, lexicographically
// sorted, that are NOT reserved. Surfaces filesystem iteration errors as
// exceptions (we cannot tell a transient IO error apart from "no candidates"
// otherwise).
std::vector<std::filesystem::path> EnumerateComponentCandidates(
    const std::filesystem::path& root) {
  std::vector<std::filesystem::path> result;
  std::error_code ec;
  std::filesystem::directory_iterator it(root, ec);
  if (ec) {
    throw std::runtime_error(
        "model package: cannot iterate " + root.string() + ": " + ec.message());
  }
  for (; it != std::filesystem::directory_iterator(); it.increment(ec)) {
    if (ec) {
      throw std::runtime_error(
          "model package: error iterating " + root.string() + ": " + ec.message());
    }
    const bool is_dir = it->is_directory(ec);
    if (ec) {
      throw std::runtime_error(
          "model package: cannot stat " + it->path().string() + ": " + ec.message());
    }
    if (!is_dir) continue;
    const std::string fname = it->path().filename().string();
    if (IsReservedChild(fname)) continue;
    result.push_back(it->path());
  }
  std::sort(result.begin(), result.end());
  return result;
}

// True iff at least one immediate non-reserved child of `root` contains a
// `metadata.json` file. Used as a fallback v4-detection signal when there is
// no `manifest.json`.
bool AnyChildHasMetadata(const std::filesystem::path& root) {
  std::error_code ec;
  std::filesystem::directory_iterator it(root, ec);
  if (ec) return false;  // Treat IO error as "not detected" — exists() check below catches real breakage.
  for (; it != std::filesystem::directory_iterator(); it.increment(ec)) {
    if (ec) return false;
    bool is_dir = it->is_directory(ec);
    if (ec || !is_dir) continue;
    const std::string fname = it->path().filename().string();
    if (IsReservedChild(fname)) continue;
    if (std::filesystem::exists(it->path() / "metadata.json", ec) && !ec) {
      return true;
    }
  }
  return false;
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
  const std::filesystem::path manifest_path = root / "manifest.json";
  const bool has_manifest = FileExistsStrict(manifest_path);
  const bool has_component_metadata = !has_manifest && AnyChildHasMetadata(root);
  if (!has_manifest && !has_component_metadata) {
    return nullptr;
  }

  auto ctx = std::make_unique<StubModelPackageContext>();
  ctx->package_root = path;

  // Determine the component name list.
  std::vector<std::string> component_names;
  if (has_manifest) {
    ParsedManifest manifest = ParseManifestJson(ReadFile(manifest_path), manifest_path);
    if (manifest.components) {
      component_names = std::move(*manifest.components);
    } else {
      // Manifest present but components absent — discover by directory scan.
      for (const auto& child : EnumerateComponentCandidates(root)) {
        component_names.push_back(child.filename().string());
      }
    }
  } else {
    for (const auto& child : EnumerateComponentCandidates(root)) {
      component_names.push_back(child.filename().string());
    }
  }

  if (component_names.empty()) {
    throw std::runtime_error(
        "model package: package recognized at " + path.string() +
        " but no components were found");
  }

  // Load each component.
  for (const auto& component_name : component_names) {
    StubComponent component;
    component.name = component_name;
    component.component_dir = path.join(component_name);

    const std::filesystem::path component_dir = root / component_name;
    const std::filesystem::path metadata_path = component_dir / "metadata.json";
    if (!FileExistsStrict(metadata_path)) {
      throw std::runtime_error(
          "model package: component '" + component_name +
          "' is missing required metadata.json at " + metadata_path.string());
    }

    std::vector<ParsedVariantInfo> variants =
        ParseMetadataJson(ReadFile(metadata_path), metadata_path);

    // Validate that every metadata-declared variant has a directory with a
    // variant.json on disk, and parse it to capture file_count + consumer_metadata.
    for (auto& parsed : variants) {
      const std::filesystem::path variant_dir = component_dir / parsed.name;
      const std::filesystem::path variant_json = variant_dir / "variant.json";
      if (!FileExistsStrict(variant_json)) {
        throw std::runtime_error(
            "model package: variant '" + parsed.name + "' of component '" +
            component_name + "' is missing variant.json at " + variant_json.string());
      }
      const std::string variant_text = ReadFile(variant_json);
      VariantManifest vm = ParseVariantManifestFromText(variant_text, variant_json);
      std::string consumer = ExtractConsumerMetadataFromText(variant_text, variant_json);

      StubVariant sv;
      sv.name = std::move(parsed.name);
      sv.folder = FromStd(variant_dir);
      sv.ep_compatibility = std::move(parsed.ep_compatibility);
      sv.file_count = vm.files.size();
      sv.consumer_metadata = std::move(consumer);
      component.variants.push_back(std::move(sv));
    }

    if (component.variants.empty()) {
      throw std::runtime_error(
          "model package: component '" + component_name + "' has no variants");
    }

    ctx->components.push_back(std::move(component));
  }

  return ctx;
}

}  // namespace Generators
