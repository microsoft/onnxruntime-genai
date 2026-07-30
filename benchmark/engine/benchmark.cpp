#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ConfigEntry {
  std::string scenario;
  int concurrency = 0;
  int prompt_k = 0;
  bool synthetic = false;
  std::string model_path;
  std::string execution_provider = "cuda";
  int generation_tokens = 128;
  int measured_runs = 2;
};

std::string EscapeJson(std::string text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

std::string ReadAllText(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open config file: " + path);
  }

  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<std::string> SplitObjects(const std::string& json_array_text) {
  std::vector<std::string> objects;

  // This lightweight parser is intentionally simple for the prototype config shape.
  int depth = 0;
  size_t start = std::string::npos;
  for (size_t i = 0; i < json_array_text.size(); ++i) {
    if (json_array_text[i] == '{') {
      if (depth == 0) {
        start = i;
      }
      ++depth;
    } else if (json_array_text[i] == '}') {
      --depth;
      if (depth == 0 && start != std::string::npos) {
        objects.push_back(json_array_text.substr(start, i - start + 1));
        start = std::string::npos;
      }
    }
  }

  return objects;
}

std::string EscapeRegex(const std::string& text) {
  std::string out;
  out.reserve(text.size() * 2);
  for (char c : text) {
    switch (c) {
      case '\\':
      case '^':
      case '$':
      case '.':
      case '|':
      case '?':
      case '*':
      case '+':
      case '(':
      case ')':
      case '[':
      case ']':
      case '{':
      case '}':
        out.push_back('\\');
        out.push_back(c);
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::string ExtractString(const std::string& object_text, const std::string& key) {
  const std::regex pattern("\\\"" + EscapeRegex(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(object_text, match, pattern)) {
    throw std::runtime_error("Missing string key in config entry: " + key);
  }
  return match[1].str();
}

int ExtractInt(const std::string& object_text, const std::string& key) {
  const std::regex pattern("\\\"" + EscapeRegex(key) + "\\\"\\s*:\\s*(-?\\d+)");
  std::smatch match;
  if (!std::regex_search(object_text, match, pattern)) {
    throw std::runtime_error("Missing integer key in config entry: " + key);
  }
  return std::stoi(match[1].str());
}

bool ExtractBool(const std::string& object_text, const std::string& key) {
  const std::regex pattern("\\\"" + EscapeRegex(key) + "\\\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(object_text, match, pattern)) {
    throw std::runtime_error("Missing bool key in config entry: " + key);
  }
  return match[1].str() == "true";
}

std::string ExtractOptionalString(const std::string& object_text,
                                  const std::string& key,
                                  const std::string& default_value) {
  const std::regex pattern("\\\"" + EscapeRegex(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(object_text, match, pattern)) {
    return default_value;
  }
  return match[1].str();
}

int ExtractOptionalInt(const std::string& object_text,
                       const std::string& key,
                       int default_value) {
  const std::regex pattern("\\\"" + EscapeRegex(key) + "\\\"\\s*:\\s*(-?\\d+)");
  std::smatch match;
  if (!std::regex_search(object_text, match, pattern)) {
    return default_value;
  }
  return std::stoi(match[1].str());
}

std::vector<ConfigEntry> LoadConfig(const std::string& config_path) {
  const std::string text = ReadAllText(config_path);
  const auto objects = SplitObjects(text);

  std::vector<ConfigEntry> entries;
  entries.reserve(objects.size());

  for (const auto& obj : objects) {
    ConfigEntry e;
    e.scenario = ExtractString(obj, "scenario");
    e.concurrency = ExtractInt(obj, "concurrency");
    e.prompt_k = ExtractInt(obj, "prompt length (1000s)");
    e.synthetic = ExtractBool(obj, "synthetic");
    e.model_path = ExtractOptionalString(obj, "model_path", "");
    e.execution_provider = ExtractOptionalString(obj, "execution_provider", "cuda");
    e.generation_tokens = ExtractOptionalInt(obj, "generation_tokens", 128);
    e.measured_runs = ExtractOptionalInt(obj, "measured_runs", 2);
    entries.push_back(std::move(e));
  }

  if (entries.empty()) {
    throw std::runtime_error("Config did not contain any benchmark entries.");
  }

  return entries;
}

void WriteFile(const std::filesystem::path& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to write file: " + path.string());
  }
  out << content;
}

std::string BuildVisualizerHtml() {
  // The visualizer is intentionally dependency-free so it can be opened directly in a browser.
  return R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Visualizer</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f6f8fb; color: #1f2937; }
    h1 { margin-bottom: 8px; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 14px; }
    th { background: #f9fafb; }
    .bar { height: 10px; background: linear-gradient(90deg, #0ea5e9, #2563eb); border-radius: 999px; }
    .muted { color: #6b7280; font-size: 13px; }
    .error { color: #b91c1c; }
  </style>
</head>
<body>
  <h1>Benchmark Results</h1>
  <p class="muted">Source: results.json (same folder as this HTML file)</p>
  <div id="content" class="card">Loading...</div>

  <script>
    async function main() {
      const root = document.getElementById('content');
      try {
        const resp = await fetch('./results.json');
        if (!resp.ok) {
          throw new Error('Failed to fetch results.json');
        }
        const data = await resp.json();
        const runs = data.runs || [];

        if (!runs.length) {
          root.innerHTML = '<p>No runs found.</p>';
          return;
        }

        const baselineRuns = runs.filter(r => r.status === 'ok');
        const maxP95 = Math.max(...baselineRuns.map(r => (r.summary && r.summary.ttft_p95_ms) || 0), 1);

        let html = '';
        html += '<h2>Summary Table</h2>';
        html += '<table><thead><tr><th>Scenario</th><th>Concurrency</th><th>Prompt(K)</th><th>Synthetic</th><th>TTFT p95 (ms)</th><th>E2E p95 (ms)</th><th>Tokens/s p50</th><th>Status</th></tr></thead><tbody>';

        for (const r of runs) {
          const s = r.summary || {};
          html += `<tr>
            <td>${r.scenario}</td>
            <td>${r.concurrency}</td>
            <td>${r.prompt_length_k}</td>
            <td>${r.synthetic}</td>
            <td>${s.ttft_p95_ms ?? '-'}</td>
            <td>${s.e2e_p95_ms ?? '-'}</td>
            <td>${s.tokens_per_s_p50 ?? '-'}</td>
            <td>${r.status}</td>
          </tr>`;
        }
        html += '</tbody></table>';

        html += '<h2 style="margin-top:18px;">TTFT p95 Chart</h2>';
        html += '<div class="muted" style="margin-bottom:8px;">Each bar is one run entry.</div>';
        for (const r of baselineRuns) {
          const ttft = (r.summary && r.summary.ttft_p95_ms) || 0;
          const pct = Math.max(2, (ttft / maxP95) * 100);
          html += `<div style="margin-bottom:8px;">
            <div class="muted">${r.scenario} | c=${r.concurrency} | p=${r.prompt_length_k}K | ttft_p95=${ttft}ms</div>
            <div class="bar" style="width:${pct}%;"></div>
          </div>`;
        }

        root.innerHTML = html;
      } catch (err) {
        root.innerHTML = `<p class="error">Unable to load results.json. If opened with file://, some browsers block fetch. Use a local static server.</p><pre>${err}</pre>`;
      }
    }

    main();
  </script>
</body>
</html>
)HTML";
}

std::string BuildIsoUtcNow() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[64]{};
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string(buf);
}

struct Args {
  std::string config_path;
  std::string output_dir = "out";
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string current = argv[i];
    if (current == "--config" && i + 1 < argc) {
      args.config_path = argv[++i];
    } else if (current == "--output" && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (current == "-h" || current == "--help") {
      std::cout << "Usage: benchmark --config <config.json> [--output <output_dir>]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + current);
    }
  }

  if (args.config_path.empty()) {
    throw std::runtime_error("Missing --config argument");
  }

  return args;
}

}  // namespace

bool RunDecodeBaseline(int concurrency,
                       int prompt_k,
                       bool synthetic,
                       const std::string& model_path,
                       const std::string& execution_provider,
                       int generation_tokens,
                       int measured_runs,
                       std::string* run_json,
                       std::string* error_message);

int main(int argc, char** argv) {
  try {
    const auto args = ParseArgs(argc, argv);
    const auto entries = LoadConfig(args.config_path);

    std::filesystem::create_directories(args.output_dir);

    std::vector<std::string> run_json_entries;
    run_json_entries.reserve(entries.size());
    bool had_failure = false;

    for (const auto& entry : entries) {
      std::string run_json;
      std::string error;

      if (entry.scenario == "decode_baseline") {
        const bool ok = RunDecodeBaseline(entry.concurrency,
                                          entry.prompt_k,
                                          entry.synthetic,
                                          entry.model_path,
                                          entry.execution_provider,
                                          entry.generation_tokens,
                                          entry.measured_runs,
                                          &run_json,
                                          &error);
        if (!ok) {
          had_failure = true;
          run_json = "{\"scenario\":\"" + EscapeJson(entry.scenario) +
                     "\",\"concurrency\":" + std::to_string(entry.concurrency) +
                     ",\"prompt_length_k\":" + std::to_string(entry.prompt_k) +
                     ",\"synthetic\":" + std::string(entry.synthetic ? "true" : "false") +
                     ",\"model_path\":\"" + EscapeJson(entry.model_path) + "\"" +
                     ",\"execution_provider\":\"" + EscapeJson(entry.execution_provider) + "\"" +
                     ",\"status\":\"error\",\"error\":\"" + EscapeJson(error) + "\"}";
        }
      } else {
        had_failure = true;
        run_json = "{\"scenario\":\"" + EscapeJson(entry.scenario) +
                   "\",\"concurrency\":" + std::to_string(entry.concurrency) +
                   ",\"prompt_length_k\":" + std::to_string(entry.prompt_k) +
                   ",\"synthetic\":" + std::string(entry.synthetic ? "true" : "false") +
                   ",\"model_path\":\"" + EscapeJson(entry.model_path) + "\"" +
                   ",\"execution_provider\":\"" + EscapeJson(entry.execution_provider) + "\"" +
                   ",\"status\":\"unsupported\",\"error\":\"Scenario not yet implemented in prototype\"}";
      }

      run_json_entries.push_back(std::move(run_json));
    }

    std::ostringstream results;
    results << "{\n";
    results << "  \"generated_utc\": \"" << BuildIsoUtcNow() << "\",\n";
    results << "  \"runs\": [\n";
    for (size_t i = 0; i < run_json_entries.size(); ++i) {
      results << "    " << run_json_entries[i] << (i + 1 == run_json_entries.size() ? "\n" : ",\n");
    }
    results << "  ]\n";
    results << "}\n";

    WriteFile(std::filesystem::path(args.output_dir) / "results.json", results.str());
    WriteFile(std::filesystem::path(args.output_dir) / "visualize.html", BuildVisualizerHtml());

    std::cout << "Wrote " << run_json_entries.size() << " run(s) to " << args.output_dir << "\n";
    return had_failure ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
