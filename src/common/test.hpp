// Micro test harness (dependency-free). Add framework later if desired.
#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace dgpp::test {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline int run_all() {
  int failed = 0;
  for (auto& tc : registry()) {
    try {
      tc.fn();
      std::printf("[ OK ] %s\n", tc.name);
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[FAIL] %s: %s\n", tc.name, e.what());
    }
  }
  std::printf("%zu tests, %d failed\n", registry().size(), failed);
  return failed == 0 ? 0 : 1;
}

}  // namespace dgpp::test

#define DGPP_TEST(name)                                                        \
  static void dgpp_test_##name();                                             \
  static ::dgpp::test::Registrar dgpp_reg_##name{#name, dgpp_test_##name};     \
  static void dgpp_test_##name()
