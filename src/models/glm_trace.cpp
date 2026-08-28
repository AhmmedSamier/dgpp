#include "models/glm_trace.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace dgpp {

namespace {

constexpr char kMagic[8] = {'D', 'G', 'P', 'P', 'T', 'C', '1', '\x01'};

void write_u32(std::FILE* f, uint32_t v) {
  if (std::fwrite(&v, 4, 1, f) != 1)
    throw std::runtime_error("glm_trace: short write");
}
void write_u64(std::FILE* f, uint64_t v) {
  if (std::fwrite(&v, 8, 1, f) != 1)
    throw std::runtime_error("glm_trace: short write");
}
void read_exact(std::FILE* f, void* dst, size_t n, const char* what) {
  if (std::fread(dst, 1, n, f) != n)
    throw std::runtime_error(std::string("glm_trace: short read (") + what +
                             ")");
}

}  // namespace

void glm_trace_write(const std::string& path,
                     const std::vector<GlmRouteTraceLayer>& layers) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("glm_trace: cannot open " + path);
  try {
    if (std::fwrite(kMagic, 8, 1, f) != 1)
      throw std::runtime_error("glm_trace: short write (magic)");
    write_u32(f, static_cast<uint32_t>(layers.size()));
    for (const auto& l : layers) {
      write_u32(f, l.layer_idx);
      write_u32(f, l.top_k);
      write_u64(f, l.tokens);
      const size_t n = static_cast<size_t>(l.tokens) * l.top_k;
      if (l.ids.size() != n || l.weights.size() != n)
        throw std::runtime_error("glm_trace: layer payload size mismatch");
      if (n && std::fwrite(l.ids.data(), 4, n, f) != n)
        throw std::runtime_error("glm_trace: short write (ids)");
      if (n && std::fwrite(l.weights.data(), 4, n, f) != n)
        throw std::runtime_error("glm_trace: short write (weights)");
    }
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
}

std::vector<GlmRouteTraceLayer> glm_trace_read(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("glm_trace: cannot open " + path);
  try {
    char magic[8];
    read_exact(f, magic, 8, "magic");
    if (std::memcmp(magic, kMagic, 8) != 0)
      throw std::runtime_error("glm_trace: bad magic");
    uint32_t n_layers = 0;
    read_exact(f, &n_layers, 4, "num_layers");
    if (n_layers > 4096)
      throw std::runtime_error("glm_trace: implausible layer count");
    std::vector<GlmRouteTraceLayer> layers(n_layers);
    for (auto& l : layers) {
      read_exact(f, &l.layer_idx, 4, "layer_idx");
      read_exact(f, &l.top_k, 4, "top_k");
      read_exact(f, &l.tokens, 8, "tokens");
      if (l.top_k == 0 || l.top_k > 64 ||
          l.tokens > (1ull << 32))
        throw std::runtime_error("glm_trace: implausible record header");
      const size_t n = static_cast<size_t>(l.tokens) * l.top_k;
      l.ids.resize(n);
      l.weights.resize(n);
      if (n) {
        read_exact(f, l.ids.data(), n * 4, "ids");
        read_exact(f, l.weights.data(), n * 4, "weights");
      }
    }
    if (std::fgetc(f) != EOF)
      throw std::runtime_error("glm_trace: trailing bytes");
    return layers;
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
}

}  // namespace dgpp
