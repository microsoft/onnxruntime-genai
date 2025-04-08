// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "log_sink.h"

#include <fstream>
#include <iostream>

#include "../logging.h"
#include "../make_string.h"

#if defined(__ANDROID__)
#include "android_log_sink.h"
#endif

namespace Generators {

namespace {

class OStreamLogSink : public LogSink {
 public:
  OStreamLogSink(std::ostream& output_stream)
      : output_stream_{output_stream} {
  }

  void Send(const LogCapture& capture) override {
    const auto label = capture.Label();
    const auto message = capture.Message();

    // Warnings will be yellow, all other labels will be blue
    const SGR label_background = capture.Severity() == LogSeverity::Warning ? SGR::Bg_Yellow : SGR::Bg_Blue;
    output_stream_ << SGR::Bold << label_background << "  " << label << "  " << SGR::Reset;

    if (!message.empty()) {
      output_stream_ << ' ' << message;
    }

    output_stream_ << "\n";
  }

 private:
  std::ostream& output_stream_;
};

class FileLogSink : public LogSink {
 public:
  FileLogSink(const fs::path& log_file)
      : file_stream_{log_file.open_for_write()}, ostream_sink_{file_stream_} {
    if (!file_stream_) {
      throw std::runtime_error(MakeString("Failed to open log file: ", log_file.string()));
    }
  }

  void Send(const LogCapture& capture) override {
    ostream_sink_.Send(capture);
  }

 private:
  std::ofstream file_stream_;
  OStreamLogSink ostream_sink_;
};

}  // namespace

std::unique_ptr<LogSink> MakeDefaultLogSink() {
#if defined(__ANDROID__)
  return MakeAndroidLogSink();
#else
  return std::make_unique<OStreamLogSink>(std::cerr);
#endif
}

std::unique_ptr<LogSink> MakeFileLogSink(const fs::path& log_file) {
  return std::make_unique<FileLogSink>(log_file);
}

}  // namespace Generators
