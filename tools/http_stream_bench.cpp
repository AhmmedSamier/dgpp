// Host-only output-delivery probe; see docs/http-stream-flush.md.
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "serve/http_server.hpp"

using namespace dgpp::serve;
using Clock = std::chrono::steady_clock;

namespace {
int64_t micros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch())
      .count();
}
double cpu_ms(const rusage& usage) {
  return 1000.0 * (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
         (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000.0;
}
void check(bool value, const char* message) {
  if (!value) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

struct Handler : HttpHandler {
  explicit Handler(int streams) : writers(streams), pending(streams) {}
  HttpWakeup wake;
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<HttpResponseWriter*> writers;
  std::vector<std::vector<std::string>> pending;
  size_t connected = 0;
  uint64_t passes = 0;
  void set_wakeup(HttpWakeup value) override { wake = value; }
  void handle(const HttpRequest& request, HttpResponseWriter& writer) override {
    const auto index = std::stoul(request.path.substr(1));
    writer.begin_stream();
    std::lock_guard lock(mutex);
    writers.at(index) = &writer;
    ++connected;
    ready.notify_all();
  }
  void idle() override {
    std::lock_guard lock(mutex);
    ++passes;
    for (size_t i = 0; i < pending.size(); ++i) {
      for (const auto& event : pending[i]) writers[i]->write_event(event);
      pending[i].clear();
    }
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "usage: %s pass25|pass10|wake RATE STREAMS TOKENS_PER_STREAM\n"
                 "RATE=0 measures two seconds of idle streaming connections.\n",
                 argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const int rate = std::stoi(argv[2]), streams = std::stoi(argv[3]), tokens = std::stoi(argv[4]);
  check(mode == "pass25" || mode == "pass10" || mode == "wake", "invalid mode");
  check(rate >= 0 && streams >= 1 && streams <= 256 && tokens >= 1, "invalid dimensions");
  Handler handler(streams);
  HttpServer server(0, &handler, streams + 1, "127.0.0.1", kDefaultHttpMaxBodyBytes, mode == "wake",
                    mode == "pass10" ? 10 : 25);
  double http_cpu = 0;
  std::thread loop([&] {
    rusage before{}, after{};
    getrusage(RUSAGE_THREAD, &before);
    server.serve();
    getrusage(RUSAGE_THREAD, &after);
    http_cpu = cpu_ms(after) - cpu_ms(before);
  });
  std::vector<int> sockets;
  for (int i = 0; i < streams; ++i) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    check(connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
          "connect failed");
    timeval timeout{10, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    const auto request = "GET /" + std::to_string(i) + " HTTP/1.1\r\nHost: localhost\r\n\r\n";
    check(send(fd, request.data(), request.size(), 0) == static_cast<ssize_t>(request.size()),
          "send failed");
    sockets.push_back(fd);
  }
  std::vector<std::vector<double>> samples(streams);
  std::vector<int> reads(streams);
  std::vector<std::thread> clients;
  if (rate > 0)
    for (int i = 0; i < streams; ++i)
      clients.emplace_back([&, i] {
        std::string buffered;
        while (samples[i].size() < static_cast<size_t>(tokens)) {
          char bytes[65536];
          const auto n = recv(sockets[i], bytes, sizeof(bytes), 0);
          const auto received = micros();
          check(n > 0, "client recv failed or timed out");
          buffered.append(bytes, n);
          bool any = false;
          while (true) {
            const auto begin = buffered.find("data: ");
            if (begin == std::string::npos) break;
            const auto end = buffered.find("\n\n", begin);
            if (end == std::string::npos) break;
            const auto sent = std::stoll(buffered.substr(begin + 6, end - begin - 6));
            samples[i].push_back((received - sent) / 1000.0);
            buffered.erase(0, end + 2);
            any = true;
          }
          reads[i] += any;
        }
      });
  {
    std::unique_lock lock(handler.mutex);
    check(handler.ready.wait_for(lock, std::chrono::seconds(5),
                                 [&] { return handler.connected == static_cast<size_t>(streams); }),
          "connections not ready");
  }
  const auto start = Clock::now();
  if (rate == 0)
    std::this_thread::sleep_for(std::chrono::seconds(2));
  else
    for (int64_t i = 0; i < static_cast<int64_t>(tokens) * streams; ++i) {
      // Stagger streams so coalescing cannot depend on synchronized producers.
      std::this_thread::sleep_until(start +
                                    std::chrono::microseconds(1000000 * i / (rate * streams)));
      {
        std::lock_guard lock(handler.mutex);
        handler.pending[i % streams].push_back(std::to_string(micros()) + " " +
                                               std::string(300, 'x'));
      }
      handler.wake();
    }
  for (auto& client : clients) client.join();
  server.stop();
  loop.join();
  const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  for (int fd : sockets) close(fd);
  std::vector<double> all;
  for (const auto& sample : samples) all.insert(all.end(), sample.begin(), sample.end());
  std::sort(all.begin(), all.end());
  const auto percentile = [&](double p) {
    return all.empty() ? 0 : all[static_cast<size_t>(p * (all.size() - 1))];
  };
  std::printf(
      "{\"mode\":\"%s\",\"rate\":%d,\"streams\":%d,\"events\":%zu,\"mean_ms\":%.4f,"
      "\"p99_ms\":%.4f,\"max_ms\":%.4f,\"events_per_read\":%.3f,\"http_cpu_ms\":%.3f,"
      "\"wall_ms\":%.3f,\"http_cpu_percent\":%.3f,\"passes\":%llu}\n",
      mode.c_str(), rate, streams, all.size(),
      all.empty() ? 0 : std::accumulate(all.begin(), all.end(), 0.0) / all.size(), percentile(.99),
      percentile(1),
      all.empty() ? 0 : double(all.size()) / std::accumulate(reads.begin(), reads.end(), 0),
      http_cpu, wall_ms, 100 * http_cpu / wall_ms, static_cast<unsigned long long>(handler.passes));
}
