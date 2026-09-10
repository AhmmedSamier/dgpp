#include "loaders/architecture.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace dgpp {

ModelArchitecture detect_architecture(const minijson::Value& root) {
  if (!root.is_object())
    throw std::runtime_error("config.json: root is not an object");
  std::string arch;
  if (const minijson::Value* a = root.find("architectures")) {
    if (!a->is_array() || a->items().empty() || !a->items()[0].is_string())
      throw std::runtime_error("config.json: architectures must be a non-empty string array");
    arch = std::string(a->items()[0].as_string());
  }
  std::string type;
  if (const minijson::Value* t = root.find("model_type"))
    if (t->is_string()) type = std::string(t->as_string());
  if (arch.rfind("Glm5", 0) == 0 || (arch.empty() && type == "glm_moe_dsa"))
    return ModelArchitecture::Glm5;
  if (arch.rfind("Qwen4Exp", 0) == 0 || (arch.empty() && type == "qwen4_exp"))
    return ModelArchitecture::Qwen4Exp;
  if (arch.rfind("Glm4Moe", 0) == 0 || (arch.empty() && type == "glm4_moe"))
    return ModelArchitecture::Glm4Moe;
  throw std::runtime_error(
      "config.json: unsupported architecture '" + arch + "' (model_type '" +
      type + "'); the engine implements Glm5*, Qwen4Exp* and Glm4Moe*");
}

ModelArchitecture detect_architecture_file(const std::string& path) {
  namespace fs = std::filesystem;
  const fs::path p = fs::is_directory(path) ? fs::path(path) / "config.json"
                                            : fs::path(path);
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f)
    throw std::runtime_error("cannot open config " + p.string() + ": " +
                             std::strerror(errno));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  const auto parsed = minijson::parse(text);
  return detect_architecture(parsed.root);
}

}  // namespace dgpp
