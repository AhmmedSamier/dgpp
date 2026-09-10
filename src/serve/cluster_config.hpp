// The cluster config (2026-09-06, the productionizing pass): ONE JSON file
// that every rank and the launcher read, replacing the launcher's
// hard-coded addresses, ports and model and the knob string each rank was
// handed on its command line.
//
//   {
//     "model": "unsloth/GLM-5.3-Flash-FP8",
//     "nodes": ["192.0.2.11", "192.0.2.12", ...],         // rank = index; [0] is the head
//     "ssh_user": "<login>",                               // the launcher's ssh user
//     "ports": {"http": 18080, "fabric": 29970, "journal": 29971},
//     "engine": {"max_concurrency": 4, "kv_capacity": 8192, "kv_dtype": "bf16", ... },
//     "paths": {"log_dir": "~/dgpp/log", "stage_dir": "/tmp/bus4",
//               "release_dir": "~/dgpp/releases", "resident_cache": ""}
//   }
//
// `dgpp-serve --config PATH --rank R` takes the model, the world (the node
// count), this rank's peer (node 0), the ports and every engine knob from
// the file; flags given after it override (the evidence scripts' knob
// strings still win). Every key is checked by name: an unknown key or a
// wrong type is an error, never a silent default — a misspelt knob must
// not deploy. The engine defaults here are the binary's own flag defaults,
// so there is exactly one set of defaults; the committed deploy/cluster.json
// states the production values explicitly.
//
// The EFFECTIVE configuration (what the rank actually runs, after flags) is
// canonicalized and digested by the app; rank 0 puts the digest on the
// journal's warm record and every peer compares its own before serving —
// a world whose ranks disagree on the model, the world size, the fabric
// ports or an engine knob refuses to form (the drift a shared knob string
// prevented by convention is now checked).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dgpp::serve {

struct ClusterConfig {
  std::string model;
  std::vector<std::string> nodes;  // rank = index; nodes[0] is the head
  std::string ssh_user;            // empty: the launcher's own user
  std::string release;             // the installed release the launcher runs (launcher-only; empty: the development binary)
  int http_port = 18080;
  int fabric_port = 29970;
  int journal_port = 29971;
  struct Engine {
    int max_concurrency = 8;
    int64_t kv_capacity = 8192;
    std::string kv_dtype = "bf16";  // the latent cache's format: bf16 | fp8 | fp4
    // The Qwen n-gram table's residency (2026-09-10): "resident" copies it
    // to the device (the default; 47.7 GiB at world 1), "mmap" leaves it
    // on the NVMe behind the page cache and gathers each step's rows on
    // the host — the single-Spark deployment.
    std::string ngram_table = "resident";
    // The Qwen dense stack's form (2026-09-10): "checkpoint" (the default:
    // the BF16 the checkpoint ships) or "fp8" (every dense projection
    // encoded to block FP8 at load — the same recipe as the FP8 releases;
    // docs/qwen38_single_spark.md).
    std::string dense_weights = "checkpoint";
    int default_max_tokens = 256;
    int queue_limit = 64;
    int max_connections = 64;
    bool no_eos = false;
    bool decode_graph = false;
    bool mtp = false;
    int mtp_depth = 1;             // draft tokens per step (1..3); needs mtp
    int graph_batch_min_live = 0;  // 0 = min(2, max_concurrency) (the batch family, 2026-09-07)
    int sampling_candidates = 128;
    double prefix_cache_gib = 1.5;
    std::string admission = "full";
    int admission_window = 256;
    double bulk_pace_gbps = -1.0;  // derived from the port rate
    int bulk_inflight = -1;
    int rendezvous_timeout_ms = 120000;
    double stats_interval_s = 10.0;
    bool reasoning_in_content = false;
  } engine;
  struct Paths {
    std::string log_dir = "~/dgpp/log";
    std::string stage_dir = "/tmp/bus4";     // where the launcher puts the peers' binary and config
    std::string release_dir = "~/dgpp/releases";
    std::string resident_cache;              // empty: the binary's default (~/.cache/dgpp/resident)
  } paths;

  int world() const { return static_cast<int>(nodes.size()); }
};

// Parses the JSON text; `what` names it in errors. Throws
// std::runtime_error naming the offending key.
ClusterConfig parse_cluster_config(const std::string& json, const std::string& what);
// Reads and parses the file.
ClusterConfig load_cluster_config(const std::string& path);

// "~" and "~/..." to $HOME; anything else unchanged.
std::string expand_home(const std::string& path);

// The 64-bit FNV-1a of a canonical configuration string, as 16 hex digits.
std::string config_digest(const std::string& canonical);

}  // namespace dgpp::serve
