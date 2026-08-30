#pragma once
// Deterministic test RNG, shared by the fixture writer (glm_fixture.hpp)
// and test-local value generation (tokens, isolated-mode stream states).
// Split from glm_fixture.hpp so .cu tests can use the RNG without
// pulling minijson through nvcc, which does not accept the header's
// vector-of-incomplete-Member pattern (g++ is fine with it).
#include <cstdint>
#include <string>

namespace glmrng {

// xorshift64 seeded per tensor name; every tensor is reproducible and
// independent of load order.
inline uint64_t seed_for(const std::string& name) {
  uint64_t h = 1469598103934665603ull;
  for (char c : name) {
    h ^= static_cast<uint8_t>(c);
    h *= 1099511628211;
  }
  return h ^ 0x9e3779b97f4a7c15ull;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float unit() {  // [-1, 1)
    return static_cast<float>(next() >> 11) / static_cast<float>(1ull << 52) -
           1.0f;
  }
  // CLT-ish near-normal: three uniforms, mean 0, sigma ~ unit*sqrt(3).
  float normal3() {
    return (unit() + unit() + unit()) * (1.0f / 3.0f);
  }
};

}  // namespace glmrng
