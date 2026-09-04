// graph_queue_repro: the synthetic model that root-caused the batched-MTP
// loopback graph stall (docs/batched_mtp_graph_stall.md, 2026-09-03). Two
// "ranks" share one process/context. Each rank captures a decode-step-shaped
// graph: a main chain (high-priority stream) of small kernels with a
// collective node per boundary, and a low-priority side chain of prefetch
// kernels forked from the main chain before every collective and once inside
// every layer, joined ONCE at the end — the captured shape of the model's
// decode row under WeightPrefetcher. A collective node publishes its
// generation to pinned memory and waits for the peer's same-node publication
// (the real kernel waits for the peer's NIC doorbell, which needs the peer's
// kernel to have run — the same cross-stream dependency, invisible to CUDA).
//
// What it measured: a kernels-only graph never stalls, even at
// CUDA_DEVICE_MAX_CONNECTIONS=1 (a spinning kernel does not block another
// stream's kernels); one 4-byte memset node per second collective (MEMSET=1)
// deadlocks the first replay at 1 connection — memset/memcpy nodes ride the
// process-shared, in-order copy-engine queue; a synchronous legacy-stream
// cudaMemset between replays (LEGACY_OP=1) deadlocks at any connection count
// (the model stream is a blocking stream). Hence the decode graph's
// kernels-only contract (glm_check_decode_graph).
//
// Knobs (environment): MODE=resident|while (the collective as a resident
// spinner, or a finite stage kernel + a conditional WHILE poll node),
// NODES, EXECS, ITERS, PREFETCH=0|1, LAYERWIN=0|1, WORK, SPINS, PFMB,
// POLL_US, STALL_MS, MEMSET=0|1|2 (2 adds an 8-byte H2D memcpy node),
// LEGACY_OP=0|1|2, VERBOSE. Exit 0 = every replay completed, 1 = a stall.
#include <cuda_runtime.h>
#include <unistd.h>
#include <cuda/atomic>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                 \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,             \
              cudaGetErrorString(e_));                                        \
      fflush(stderr);                                                         \
      _exit(2);                                                               \
    }                                                                         \
  } while (0)

static int env_int(const char* n, int d) {
  const char* v = getenv(n);
  return v ? atoi(v) : d;
}
static std::string env_str(const char* n, const char* d) {
  const char* v = getenv(n);
  return v ? v : d;
}

constexpr int kWorld = 2;
constexpr int kThreads = 256;

__device__ __forceinline__ uint64_t ld_sys(const volatile uint64_t* p) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> r(
      *const_cast<uint64_t*>(p));
  return r.load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void st_sys(volatile uint64_t* p, uint64_t v) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> r(
      *const_cast<uint64_t*>(p));
  r.store(v, cuda::memory_order_release);
}
__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// The chain's kernels: a dependent-FMA burn.
__global__ void work_kernel(float* buf, int spins) {
  float a = buf[threadIdx.x];
  const float b = 1.0001f;
  for (int i = 0; i < spins; ++i) a = fmaf(a, b, 0.5f);
  if (a == 12345.0f) buf[threadIdx.x] = a;
}

// The prefetch kernel (kernels/l2_prefetch.cu, Light rate: 16 blocks x unroll 2).
__device__ uint32_t g_sink[2];
__global__ __launch_bounds__(kThreads) void prefetch_kernel(const uint4* p,
                                                            size_t vec_count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
  uint32_t fold = 0;
  for (; i + stride < vec_count; i += 2 * stride) {
    const uint4 v0 = __ldcg(p + i);
    const uint4 v1 = __ldcg(p + i + stride);
    fold ^= v0.x ^ v0.w ^ v1.x ^ v1.w;
  }
  for (; i < vec_count; i += stride) {
    const uint4 v = __ldcg(p + i);
    fold ^= v.x ^ v.w;
  }
  if (fold == g_sink[0] + 0x9E3779B9u) g_sink[1] = fold;
}

// ---- resident collective (today's shape) ------------------------------------
__global__ __launch_bounds__(kThreads) void coll_resident(
    volatile uint64_t* my_flag, const volatile uint64_t* peer_flag,
    const volatile uint64_t* gen_cell, const volatile uint64_t* poison,
    uint64_t deadline_ns, uint32_t* status, volatile uint64_t* stamp) {
  __shared__ int s_stop;
  const uint64_t gen = ld_sys(gen_cell);
  if (threadIdx.x == 0) {
    s_stop = 0;
    st_sys(stamp, globaltimer_ns());
    st_sys(my_flag, gen);
  }
  __syncthreads();
  const uint64_t t0 = globaltimer_ns();
  for (;;) {
    if (threadIdx.x == 0) {
      if (ld_sys(peer_flag) >= gen) s_stop = 1;
      else if (ld_sys(poison) != 0 || globaltimer_ns() - t0 > deadline_ns)
        s_stop = 2;
    }
    __syncthreads();
    const int stop = s_stop;
    if (stop) {
      if (threadIdx.x == 0 && stop == 2) atomicAdd(status, 1u);
      return;
    }
    __nanosleep(200);
    __syncthreads();
  }
}

// ---- while-node collective ----------------------------------------------------
__global__ void coll_stage(volatile uint64_t* my_flag,
                           const volatile uint64_t* gen_cell,
                           volatile uint64_t* stamp) {
  if (threadIdx.x != 0) return;
  const uint64_t gen = ld_sys(gen_cell);
  st_sys(stamp, globaltimer_ns());
  st_sys(my_flag, gen);
}
// Finite poll: spins at most `budget_ns`, then yields (leaves the handle set) or
// clears the handle when the peer has published / on poison or deadline.
__global__ void coll_poll(cudaGraphConditionalHandle h,
                          const volatile uint64_t* peer_flag,
                          const volatile uint64_t* gen_cell,
                          const volatile uint64_t* poison,
                          const volatile uint64_t* stamp, uint64_t deadline_ns,
                          uint64_t budget_ns, uint32_t* status,
                          uint32_t* iters) {
  if (threadIdx.x != 0) return;
  const uint64_t gen = ld_sys(gen_cell);
  atomicAdd(iters, 1u);
  const uint64_t t0 = globaltimer_ns();
  for (;;) {
    if (ld_sys(peer_flag) >= gen) {
      cudaGraphSetConditional(h, 0);
      return;
    }
    if (ld_sys(poison) != 0 || globaltimer_ns() - ld_sys(stamp) > deadline_ns) {
      atomicAdd(status, 1u);
      cudaGraphSetConditional(h, 0);
      return;
    }
    if (globaltimer_ns() - t0 > budget_ns) return;  // yield; handle stays 1
    __nanosleep(200);
  }
}

struct Shared {
  // pinned, per rank
  volatile uint64_t* flags = nullptr;   // [nodes] published gen per node
  volatile uint64_t* stamps = nullptr;  // [nodes] publish time (globaltimer)
  volatile uint64_t* gen = nullptr;     // the replay's generation
  volatile uint64_t* poison = nullptr;
  uint32_t* status = nullptr;           // device: deadline/poison exits
  uint32_t* iters = nullptr;            // device: poll iterations
};

struct Config {
  std::string mode = "resident";
  int nodes = 90;
  int execs = 5;
  int iters = 200;
  int prefetch = 1;      // side-chain forks on/off
  int layer_windows = 1; // the intra-layer window too
  int work_kernels = 4;  // per boundary
  int work_spins = 8000;
  int prefetch_mb = 12;
  int poll_budget_us = 0;
  int stall_ms = 3000;
  int verbose = 0;
  int legacy_op = 0;  // sync cudaMemset (legacy stream) before each launch
  int memset_nodes = 0;  // 1: a 4-byte memset node before every other collective; 2: + an 8-byte H2D memcpy node at the start
};

static std::atomic<int> g_barrier_count{0};
static std::atomic<int> g_barrier_gen{0};
static void barrier() {
  const int gen = g_barrier_gen.load();
  if (g_barrier_count.fetch_add(1) + 1 == kWorld) {
    g_barrier_count.store(0);
    g_barrier_gen.store(gen + 1);
  } else {
    while (g_barrier_gen.load() == gen) std::this_thread::yield();
  }
}

static std::atomic<int> g_failed{0};

static void rank_main(int rank, const Config& cfg, Shared* sh) {
  Shared& me = sh[rank];
  Shared& peer = sh[rank ^ 1];
  int least = 0, greatest = 0;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  cudaStream_t main = nullptr, side = nullptr;
  // The model's main stream: cudaStreamDefault flags at the highest priority.
  CK(cudaStreamCreateWithPriority(&main, cudaStreamDefault, greatest));
  CK(cudaStreamCreateWithPriority(&side, cudaStreamNonBlocking, least));
  cudaEvent_t fork = nullptr, join = nullptr;
  CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));

  float* work_buf = nullptr;
  CK(cudaMalloc(&work_buf, kThreads * sizeof(float)));
  CK(cudaMemsetAsync(work_buf, 0, kThreads * sizeof(float), main));
  CK(cudaStreamSynchronize(main));
  uint32_t* ms_buf = nullptr;
  CK(cudaMalloc(&ms_buf, 4096));
  uint64_t* pin_src = nullptr;
  CK(cudaHostAlloc(reinterpret_cast<void**>(&pin_src), 64, cudaHostAllocDefault));
  const size_t pf_bytes = static_cast<size_t>(cfg.prefetch_mb) << 20;
  uint4* pf_buf = nullptr;
  CK(cudaMalloc(&pf_buf, pf_bytes * 2));
  const size_t pf_vecs = pf_bytes / 16;

  const uint64_t deadline_ns = static_cast<uint64_t>(cfg.stall_ms) * 1000000ull;
  const uint64_t budget_ns = static_cast<uint64_t>(cfg.poll_budget_us) * 1000ull;

  auto open_window = [&] {
    if (!cfg.prefetch) return;
    CK(cudaEventRecord(fork, main));
    CK(cudaStreamWaitEvent(side, fork, 0));
    // two adds per window (a small array and the clamped big weight)
    prefetch_kernel<<<16, kThreads, 0, side>>>(pf_buf, pf_vecs / 4);
    prefetch_kernel<<<16, kThreads, 0, side>>>(pf_buf + pf_vecs, pf_vecs * 3 / 4);
    CK(cudaGetLastError());
  };
  auto join_side = [&] {
    if (!cfg.prefetch) return;
    CK(cudaEventRecord(join, side));
    CK(cudaStreamWaitEvent(main, join, 0));
  };
  auto work = [&](int n) {
    for (int i = 0; i < n; ++i)
      work_kernel<<<1, kThreads, 0, main>>>(work_buf, cfg.work_spins);
    CK(cudaGetLastError());
  };

  auto collective = [&](int n) {
    volatile uint64_t* my_flag = me.flags + n;
    const volatile uint64_t* peer_flag = peer.flags + n;
    volatile uint64_t* stamp = me.stamps + n;
    if (cfg.mode == "while") {
      coll_stage<<<1, 32, 0, main>>>(my_flag, me.gen, stamp);
      CK(cudaGetLastError());
      cudaStreamCaptureStatus st;
      cudaGraph_t g = nullptr;
      const cudaGraphNode_t* deps = nullptr;
      const cudaGraphEdgeData* edges = nullptr;
      size_t nd = 0;
      CK(cudaStreamGetCaptureInfo(main, &st, nullptr, &g, &deps, &edges, &nd));
      cudaGraphConditionalHandle h;
      CK(cudaGraphConditionalHandleCreate(&h, g, 1, cudaGraphCondAssignDefault));
      cudaGraphNodeParams p = {};
      p.type = cudaGraphNodeTypeConditional;
      p.conditional.handle = h;
      p.conditional.type = cudaGraphCondTypeWhile;
      p.conditional.size = 1;
      cudaGraphNode_t cnode = nullptr;
      CK(cudaGraphAddNode(&cnode, g, deps, edges, nd, &p));
      cudaGraph_t body = p.conditional.phGraph_out[0];
      cudaKernelNodeParams kp = {};
      kp.func = reinterpret_cast<void*>(coll_poll);
      kp.gridDim = dim3(1);
      kp.blockDim = dim3(32);
      const volatile uint64_t* stamp_c = stamp;
      const volatile uint64_t* gen_c = me.gen;
      const volatile uint64_t* poison_c = me.poison;
      uint64_t dl = deadline_ns, bg = budget_ns;
      uint32_t* status = me.status;
      uint32_t* iters = me.iters;
      void* args[] = {&h, &peer_flag, &gen_c, &poison_c, &stamp_c, &dl, &bg,
                      &status, &iters};
      kp.kernelParams = args;
      cudaGraphNode_t kn = nullptr;
      CK(cudaGraphAddKernelNode(&kn, body, nullptr, 0, &kp));
      CK(cudaStreamUpdateCaptureDependencies(main, &cnode, nullptr, 1,
                                             cudaStreamSetCaptureDependencies));
    } else {
      coll_resident<<<1, kThreads, 0, main>>>(my_flag, peer_flag, me.gen,
                                              me.poison, deadline_ns, me.status,
                                              stamp);
      CK(cudaGetLastError());
    }
  };

  barrier();
  std::vector<cudaGraphExec_t> execs;
  for (int e = 0; e < cfg.execs; ++e) {
    CK(cudaStreamBeginCapture(main, cudaStreamCaptureModeThreadLocal));
    if (cfg.memset_nodes >= 2)
      CK(cudaMemcpyAsync(ms_buf + 8, pin_src, 8, cudaMemcpyHostToDevice, main));
    for (int n = 0; n < cfg.nodes; ++n) {
      work(cfg.work_kernels / 2);
      if (cfg.memset_nodes && (n % 2) == 0) CK(cudaMemsetAsync(ms_buf + (n % 512), 0, 4, main));
      if (cfg.layer_windows && (n & 1) == 0) open_window();  // the layer's window
      work(cfg.work_kernels - cfg.work_kernels / 2);
      open_window();  // the boundary window
      collective(n);
      work(1);
    }
    join_side();
    cudaGraph_t graph = nullptr;
    CK(cudaStreamEndCapture(main, &graph));
    cudaGraphExec_t exec = nullptr;
    CK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    CK(cudaStreamSynchronize(main));
    size_t nn = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &nn));
    if (rank == 0 && e == 0)
      printf("graph: %zu nodes, mode=%s prefetch=%d layer_windows=%d\n", nn,
             cfg.mode.c_str(), cfg.prefetch, cfg.layer_windows);
    CK(cudaGraphDestroy(graph));
    execs.push_back(exec);
  }
  barrier();
  const auto t_start = std::chrono::steady_clock::now();
  for (int it = 1; it <= cfg.iters; ++it) {
    if (g_failed.load()) break;
    __atomic_store_n(const_cast<uint64_t*>(me.gen), static_cast<uint64_t>(it),
                     __ATOMIC_RELEASE);
    barrier();
    if (cfg.legacy_op) {
      if (cfg.legacy_op == 2 && rank == 1) std::this_thread::sleep_for(std::chrono::microseconds(300));
      CK(cudaMemset(work_buf, 0, 4));
    }
    cudaGraphExec_t exec = execs[static_cast<size_t>(it % cfg.execs)];
    CK(cudaGraphLaunch(exec, main));
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = false;
    for (;;) {
      const cudaError_t q = cudaStreamQuery(main);
      if (q == cudaSuccess) {
        ok = true;
        break;
      }
      if (q != cudaErrorNotReady) CK(q);
      if (std::chrono::steady_clock::now() - t0 >
          std::chrono::milliseconds(cfg.stall_ms)) break;
      std::this_thread::yield();
    }
    if (!ok) {
      // Stall. Poison both sides so every spinner exits, then report.
      g_failed.store(1);
      __atomic_store_n(const_cast<uint64_t*>(me.poison), 1ull, __ATOMIC_RELEASE);
      __atomic_store_n(const_cast<uint64_t*>(peer.poison), 1ull, __ATOMIC_RELEASE);
      int mine = 0, theirs = 0;
      for (int n = 0; n < cfg.nodes; ++n) {
        if (__atomic_load_n(const_cast<uint64_t*>(me.flags + n), __ATOMIC_ACQUIRE) >= static_cast<uint64_t>(it)) mine = n + 1;
        if (__atomic_load_n(const_cast<uint64_t*>(peer.flags + n), __ATOMIC_ACQUIRE) >= static_cast<uint64_t>(it)) theirs = n + 1;
      }
      printf("STALL rank %d iter %d (exec %d): my published nodes %d/%d, peer's %d/%d\n",
             rank, it, it % cfg.execs, mine, cfg.nodes, theirs, cfg.nodes);
      fflush(stdout);
      CK(cudaStreamSynchronize(main));
      break;
    }
    if (cfg.verbose && rank == 0 && (it % 50) == 0) {
      printf("iter %d ok\n", it);
      fflush(stdout);
    }
  }
  CK(cudaStreamSynchronize(main));
  barrier();
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
          .count();
  uint32_t status = 0, iters = 0;
  CK(cudaMemcpyAsync(&status, me.status, 4, cudaMemcpyDeviceToHost, main));
  CK(cudaMemcpyAsync(&iters, me.iters, 4, cudaMemcpyDeviceToHost, main));
  CK(cudaStreamSynchronize(main));
  if (!g_failed.load())
    printf("rank %d: %d replays OK in %.2fs (%.1f us/collective), deadline exits %u, poll iters %u (%.2f/collective)\n",
           rank, cfg.iters, secs,
           secs * 1e6 / (static_cast<double>(cfg.iters) * cfg.nodes), status,
           iters,
           static_cast<double>(iters) / (static_cast<double>(cfg.iters) * cfg.nodes));
  fflush(stdout);
  for (cudaGraphExec_t e : execs) cudaGraphExecDestroy(e);
}

int main(int argc, char** argv) {
  Config cfg;
  cfg.mode = env_str("MODE", "resident");
  cfg.nodes = env_int("NODES", 90);
  cfg.execs = env_int("EXECS", 5);
  cfg.iters = env_int("ITERS", 200);
  cfg.prefetch = env_int("PREFETCH", 1);
  cfg.layer_windows = env_int("LAYERWIN", 1);
  cfg.work_kernels = env_int("WORK", 4);
  cfg.work_spins = env_int("SPINS", 8000);
  cfg.prefetch_mb = env_int("PFMB", 12);
  cfg.poll_budget_us = env_int("POLL_US", 0);
  cfg.stall_ms = env_int("STALL_MS", 3000);
  cfg.verbose = env_int("VERBOSE", 0);
  cfg.legacy_op = env_int("LEGACY_OP", 0);
  cfg.memset_nodes = env_int("MEMSET", 0);
  (void)argc;
  (void)argv;

  if (cfg.mode != "resident" && cfg.mode != "while") {
    fprintf(stderr, "MODE must be resident or while\n");
    return 2;
  }
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  Shared sh[kWorld];
  for (int r = 0; r < kWorld; ++r) {
    void* p = nullptr;
    CK(cudaHostAlloc(&p, (2 * cfg.nodes + 8) * sizeof(uint64_t),
                     cudaHostAllocMapped));
    memset(p, 0, (2 * cfg.nodes + 8) * sizeof(uint64_t));
    uint64_t* u = static_cast<uint64_t*>(p);
    sh[r].flags = u;
    sh[r].stamps = u + cfg.nodes;
    sh[r].gen = u + 2 * cfg.nodes;
    sh[r].poison = u + 2 * cfg.nodes + 1;
    CK(cudaMalloc(&sh[r].status, 8));
    CK(cudaMemset(sh[r].status, 0, 8));
    sh[r].iters = sh[r].status + 1;
  }
  std::vector<std::thread> threads;
  for (int r = 0; r < kWorld; ++r)
    threads.emplace_back(rank_main, r, std::cref(cfg), sh);
  for (auto& t : threads) t.join();
  if (g_failed.load()) {
    printf("RESULT: STALL\n");
    return 1;
  }
  printf("RESULT: OK\n");
  return 0;
}
