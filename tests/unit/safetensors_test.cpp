#include <cstdio>
#include <cstring>
#include <filesystem>

#include "common/test.hpp"
#include "loaders/safetensors.hpp"

namespace {
namespace fs = std::filesystem;

// Builds a tiny safetensors file with one F32, one BF16, one F8_E4M3 tensor.
fs::path make_fixture() {
  fs::path dir = fs::temp_directory_path() / "dgpp_st_test";
  fs::create_directories(dir);
  fs::path file = dir / "fixture.safetensors";

  const char* header =
      R"({"a_f32":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
      R"("b_bf16":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]},)"
      R"("c_fp8":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[32,36]}})";
  uint64_t hlen = static_cast<uint64_t>(std::strlen(header));

  std::FILE* f = std::fopen(file.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write fixture");
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header, hlen, 1, f);

  float f32v[6] = {1.f, -2.5f, 3.25f, 4.f, -5.f, 0.125f};
  uint16_t bf16v[4];
  for (int i = 0; i < 4; ++i)
    bf16v[i] = dgpp::float_to_bf16_bits(0.5f * i);
  uint8_t fp8v[4] = {0x38, 0x40, 0x7E, 0x00};  // .28125?, 0.5, 448, 0

  std::fwrite(f32v, sizeof(f32v), 1, f);
  std::fwrite(bf16v, sizeof(bf16v), 1, f);
  std::fwrite(fp8v, sizeof(fp8v), 1, f);
  std::fclose(f);
  return file;
}
}  // namespace

DGPP_TEST(safetensors_mmap_reader_roundtrip) {
  fs::path p = make_fixture();
  auto sf = dgpp::SafetensorsFile::open(p.string());

  const dgpp::TensorInfo* a = sf->find("a_f32");
  if (!a || a->dtype != dgpp::DType::F32) throw std::runtime_error("a meta");
  // This fixture deliberately has an unpadded JSON header. Read its byte
  // storage without imposing the alignment of float/uint16_t on the mapping.
  float af[6];
  std::memcpy(af, a->data, sizeof(af));
  if (af[0] != 1.f || af[1] != -2.5f || af[5] != 0.125f)
    throw std::runtime_error("a data");

  const dgpp::TensorInfo& b = sf->at("b_bf16");
  if (b.dtype != dgpp::DType::BF16 || b.numel() != 4)
    throw std::runtime_error("b meta");
  uint16_t bb[4];
  std::memcpy(bb, b.data, sizeof(bb));
  if (bb[2] != dgpp::float_to_bf16_bits(1.0f))
    throw std::runtime_error("b data");

  const dgpp::TensorInfo& c = sf->at("c_fp8");
  if (c.dtype != dgpp::DType::F8_E4M3) throw std::runtime_error("c meta");
  const uint8_t* cb = static_cast<const uint8_t*>(c.data);
  // fixture bytes: 0x38 -> +1.0, 0x40 -> +2.0, 0x7E -> max finite 448,
  // 0x00 -> +0
  if (cb[2] != 0x7E) throw std::runtime_error("c saturate byte");
  if (dgpp::fp8_e4m3_bits_to_float(cb[0]) != 1.0f)
    throw std::runtime_error("c[0] decode");
  if (dgpp::fp8_e4m3_bits_to_float(cb[1]) != 2.0f)
    throw std::runtime_error("c[1] decode");

  if (sf->find("missing") != nullptr) throw std::runtime_error("ghost tensor");

  std::error_code ec;
  fs::remove(p, ec);
}
