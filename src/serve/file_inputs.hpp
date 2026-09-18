#pragma once

#include <cstdint>
#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "loaders/minijson.hpp"
#include "serve/http_server.hpp"

namespace dgpp::serve {

struct FileInputConfig {
  std::string directory;  // empty: private, ephemeral store for this process
  std::string pdf_command = "pdftotext";
  uint64_t max_file_bytes = 50ull * 1024 * 1024;
  uint64_t max_request_bytes = 50ull * 1024 * 1024;
  uint64_t max_text_bytes = 128ull * 1024 * 1024;
  uint64_t max_storage_bytes = 1024ull * 1024 * 1024;
  int pdf_timeout_ms = 120000;
  int workers = 4;
};

struct FileInputError : std::runtime_error {
  FileInputError(std::string message, std::string param = "file", int status = 400,
                 std::string code = "invalid_file")
      : std::runtime_error(std::move(message)), param(std::move(param)), status(status), code(std::move(code)) {}
  std::string param;
  int status;
  std::string code;
};

struct FileResponse { int status = 200; std::string content_type = "application/json", body; };

// Called on preprocessing workers. No HTTP writers, engine state or GPU
// calls are touched; a disconnected client's result can simply be dropped.
class FileInputs {
 public:
  explicit FileInputs(FileInputConfig config);
  ~FileInputs();
  static bool needed(const HttpRequest& request);
  HttpRequest prepare(HttpRequest request);
  FileResponse route(const HttpRequest& request);
  void stop() { stopping_.store(true, std::memory_order_relaxed); }
  const FileInputConfig& config() const { return config_; }
 private:
  std::string extract(const std::string& data, const std::string& filename);
  std::string resolve(const minijson::Value& file, std::string* filename);
  FileInputConfig config_;
  bool ephemeral_ = false;
  std::atomic<bool> stopping_{false};
  std::mutex mutex_;
  std::map<std::string, minijson::Value> files_;
  uint64_t stored_bytes_ = 0;
};

}  // namespace dgpp::serve
