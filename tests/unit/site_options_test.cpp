#include "common/test.hpp"
#include "net/site_options.hpp"

#include <optional>

namespace {
struct SavedEnvironment {
  const char* key;
  std::optional<std::string> previous;
  SavedEnvironment(const char* name, const char* value) : key(name) {
    if (const char* old = std::getenv(key)) previous = old;
    ::setenv(key, value, 1);
  }
  ~SavedEnvironment() {
    if (previous) ::setenv(key, previous->c_str(), 1);
    else ::unsetenv(key);
  }
};
}

DGPP_TEST(site_options_preserve_explicit_lane_order_and_gid_mapping) {
  SavedEnvironment devices("DGPP_ROCE_DEVICES", "nic_b  nic_a");
  SavedEnvironment indices("DGPP_ROCE_GID_INDICES", "5 3");
  const auto lanes = dgpp::net::configured_lane_devices();
  if (lanes != std::vector<std::string>{"nic_b", "nic_a"} ||
      dgpp::net::configured_gid_index("nic_b") != 5 ||
      dgpp::net::configured_gid_index("nic_a") != 3)
    throw std::runtime_error("explicit lane order or GID mapping changed");
}

DGPP_TEST(site_options_reject_invalid_gid_mapping) {
  SavedEnvironment devices("DGPP_ROCE_DEVICES", "nic_a nic_b");
  for (const char* value : {"1", "256 3", "3x 4", "-1 2"}) {
    SavedEnvironment indices("DGPP_ROCE_GID_INDICES", value);
    bool rejected = false;
    try { (void)dgpp::net::configured_gid_index("nic_a"); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid GID mapping accepted");
  }
  SavedEnvironment indices("DGPP_ROCE_GID_INDICES", "");
  if (dgpp::net::configured_gid_index("nic_a") != -1)
    throw std::runtime_error("unset GID indices must request automatic selection");
}
