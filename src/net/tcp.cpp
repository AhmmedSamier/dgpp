#include "net/tcp.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "common/log.hpp"

namespace dgpp::net {
namespace {

// poll() retrying only on EINTR; -1 on any other error.
int poll_one(struct pollfd* p, int timeout_ms) {
  for (;;) {
    int ready = ::poll(p, 1, timeout_ms);
    if (ready >= 0) return ready;
    if (errno == EINTR) continue;
    return -1;
  }
}

void set_socket_blocking(int fd, bool blocking) {
  int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) return;
  if (blocking) {
    flags &= ~O_NONBLOCK;
  } else {
    flags |= O_NONBLOCK;
  }
  ::fcntl(fd, F_SETFL, flags);
}

std::string errno_text() { return std::strerror(errno); }

}  // namespace

TcpConn::TcpConn(int fd) : fd_(fd) {}

TcpConn::TcpConn(TcpConn&& other) noexcept : fd_(other.fd_) {
  other.fd_ = -1;
}

TcpConn& TcpConn::operator=(TcpConn&& other) noexcept {
  if (this != &other) {
    close();
    fd_ = other.fd_;
    other.fd_ = -1;
  }
  return *this;
}

TcpConn::~TcpConn() { close(); }

TcpConn TcpConn::connect(const std::string& host, uint16_t port,
                         int timeout_ms) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_NUMERICSERV;

  const std::string service = std::to_string(port);
  addrinfo* list = nullptr;
  int rc = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &list);
  if (rc != 0 || list == nullptr) {
    throw std::runtime_error("tcp connect: cannot resolve \"" + host +
                             "\": " + (rc == 0 ? "no addresses"
                                              : ::gai_strerror(rc)));
  }

  std::string last_error;
  for (addrinfo* ai = list; ai != nullptr; ai = ai->ai_next) {
    int fd = ::socket(ai->ai_family, ai->ai_socktype | SOCK_CLOEXEC,
                      ai->ai_protocol);
    if (fd < 0) {
      last_error = "socket: " + errno_text();
      continue;
    }
    // Non-blocking connect + poll so the timeout is honored for hosts that
    // are unreachable rather than merely refused.
    set_socket_blocking(fd, false);
    rc = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
    if (rc != 0 && errno != EINPROGRESS) {
      last_error = "connect: " + errno_text();
      ::close(fd);
      continue;
    }
    if (rc != 0) {
      struct pollfd p{};
      p.fd = fd;
      p.events = POLLOUT;
      if (poll_one(&p, timeout_ms) != 1) {
        last_error = "connect: timeout after " + std::to_string(timeout_ms) +
                     " ms";
        ::close(fd);
        continue;
      }
      int so_error = 0;
      socklen_t len = sizeof(so_error);
      if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) < 0 ||
          so_error != 0) {
        last_error = "connect: " +
                     (so_error != 0 ? std::strerror(so_error) : errno_text());
        ::close(fd);
        continue;
      }
    }
    set_socket_blocking(fd, true);

    int nodelay = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    ::freeaddrinfo(list);
    return TcpConn(fd);
  }
  ::freeaddrinfo(list);
  throw std::runtime_error("tcp connect to " + host + ":" + service +
                            " failed: " + last_error);
}

void TcpConn::set_io_deadline_ms(int ms) {
  if (fd_ < 0) return;
  struct timeval tv{};
  tv.tv_sec = ms / 1000;
  tv.tv_usec = (ms % 1000) * 1000;
  ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

bool TcpConn::write_all(const void* data, size_t len) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  size_t done = 0;
  while (done < len) {
    ssize_t n = ::send(fd_, bytes + done, len - done, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) return false;
    done += static_cast<size_t>(n);
  }
  return true;
}

bool TcpConn::read_exact(void* dst, size_t len) {
  auto* bytes = static_cast<unsigned char*>(dst);
  size_t done = 0;
  while (done < len) {
    ssize_t n = ::recv(fd_, bytes + done, len - done, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;  // includes SO_RCVTIMEO expiry
    }
    if (n == 0) return false;  // peer closed
    done += static_cast<size_t>(n);
  }
  return true;
}

int TcpConn::read_some(void* dst, size_t cap) {
  for (;;) {
    ssize_t n = ::recv(fd_, dst, cap, 0);
    if (n >= 0) return static_cast<int>(n);
    if (errno == EINTR) continue;
    return -1;  // SO_RCVTIMEO expiry or error
  }
}

bool TcpConn::wait_readable(int timeout_ms) {
  struct pollfd p{};
  p.fd = fd_;
  p.events = POLLIN;
  return poll_one(&p, timeout_ms) == 1;
}

void TcpConn::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

TcpListener::TcpListener(TcpListener&& other) noexcept
    : fd_(other.fd_), port_(other.port_) {
  other.fd_ = -1;
  other.port_ = 0;
}

TcpListener& TcpListener::operator=(TcpListener&& other) noexcept {
  if (this != &other) {
    close();
    fd_ = other.fd_;
    port_ = other.port_;
    other.fd_ = -1;
    other.port_ = 0;
  }
  return *this;
}

TcpListener::~TcpListener() { close(); }

TcpListener TcpListener::bind(uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, IPPROTO_TCP);
  if (fd < 0) {
    throw std::runtime_error("tcp listen: socket: " + errno_text());
  }
  // Tests rebind frequently; TIME_WAIT sockets must not block the next run.
  int reuse = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = errno_text();
    ::close(fd);
    throw std::runtime_error("tcp listen: bind port " + std::to_string(port) +
                             ": " + err);
  }
  if (::listen(fd, 16) < 0) {
    std::string err = errno_text();
    ::close(fd);
    throw std::runtime_error("tcp listen: " + err);
  }

  TcpListener listener;
  listener.fd_ = fd;
  if (port == 0) {
    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &len) < 0) {
      std::string err = errno_text();
      ::close(fd);
      throw std::runtime_error("tcp listen: getsockname: " + err);
    }
    listener.port_ = ntohs(bound.sin_port);
  } else {
    listener.port_ = port;
  }
  return listener;
}

TcpConn TcpListener::accept(int timeout_ms) const {
  struct pollfd p{};
  p.fd = fd_;
  p.events = POLLIN;
  if (poll_one(&p, timeout_ms) != 1) {
    return TcpConn();
  }
  for (;;) {
    int fd = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd >= 0) {
      int nodelay = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
      return TcpConn(fd);
    }
    if (errno == EINTR || errno == ECONNABORTED) continue;
    DGPP_LOG_ERROR("tcp accept: {}", errno_text());
    return TcpConn();
  }
}

void TcpListener::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
    port_ = 0;
  }
}

}  // namespace dgpp::net
