// Compare port candidates against the original path on identical paged state.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/bf16_residency.hpp"
#include "models/mimo/forward.hpp"
#include "mimo_fixture.hpp"
using namespace dgpp;
static void require(bool v, const char* message) { if (!v) throw std::runtime_error(message); }
static std::vector<MimoModel::Outputs> run(const MimoTextConfig& cfg, const std::string& dir,
                                         LatentFormat format, bool head, bool cache) {
  setenv("DGPP_MIMO_PREFILL_LAST_HEAD", head ? "1" : "0", 1);
  setenv("DGPP_MIMO_MTP_CACHE_ONLY", cache ? "1" : "0", 1);
  MimoModel m(cfg, dir, 64, 512, MimoResidency::Resident, nullptr, 0, 1, 2, true, 0, format);
  m.session_graph_prepare();
  std::vector<MimoModel::Outputs> out;
  for (int n : {23, 97}) {
    std::vector<int64_t> prompt;
    for (int i = 0; i < n; ++i) prompt.push_back((i * 13 + 7) % cfg.vocab_size);
    for (int req : {0, 1}) out.push_back(m.session_prefill(req, prompt));
    for (int step = 0; step < 3; ++step) {
      for (int req : {1, 0}) {
        out.push_back(m.session_draft(req, {int64_t(7 + step)}));
        out.push_back(m.session_step(req, 7 + step));
      }
    }
    m.session_close(0); m.session_close(1);
  }
  return out;
}
int main() {
  try {
    auto cfg = mimofx::tiny_config();
    auto dir = (std::filesystem::current_path() / "mimo_port_fixture").string();
    mimofx::write_fixture(cfg, dir);
    for (auto residency : {Bf16Residency::Checkpoint, Bf16Residency::Bf12}) {
      set_bf16_residency(residency);
      require(bf16_residency() == residency, "external BF12 override invalidates the test");
      std::printf("Weight residency: %s\n", bf16_residency_name(residency));
      for (auto format : {LatentFormat::kBf16, LatentFormat::kFp8}) {
        auto ref = run(cfg, dir, format, false, false);
        for (int mode : {1, 2, 3}) {
          auto got = run(cfg, dir, format, mode & 1, mode & 2);
          require(got.size() == ref.size(), "output count");
          double worst = 0;
          for (size_t i = 0; i < got.size(); ++i) {
            require(got[i].final_hidden_bits == ref[i].final_hidden_bits, "hidden state changed");
            const auto& a = got[i].logits; const auto& b = ref[i].logits;
            require(a.size() == b.size(), "logit shape changed");
            if (!(mode & 1)) require(std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0,
                                    "cache-only logits must be bitwise identical");
            double d2=0, b2=0;
            for (size_t j=0; j<a.size(); ++j) {
              require(std::isfinite(a[j]) && std::isfinite(b[j]), "non-finite logit");
              d2 += (double(a[j])-b[j])*(double(a[j])-b[j]); b2 += double(b[j])*b[j];
            }
            double l2=std::sqrt(d2/std::max(b2,1e-30)); worst=std::max(worst,l2);
            require(l2 < 1e-5, "head relative L2 exceeds 1e-5");
            require(std::max_element(a.begin(),a.end())-a.begin() == std::max_element(b.begin(),b.end())-b.begin(),
                    "top-1 changed");
          }
          std::printf("PASS format=%d mode=%d outputs=%zu worst_l2=%.9g\n", int(format),mode,got.size(),worst);
        }
      }
    }
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
