#pragma once

#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

namespace dgpp::serve {

// Bypass the logger and teardown: either may be the operation that stalled.
// A full supervisor log pipe must not hold up the exit either.
template <std::size_t N>
[[noreturn]] inline void emergency_exit(const char (&message)[N]) {
  const int flags = ::fcntl(STDERR_FILENO, F_GETFL);
  if (flags >= 0 && ::fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK) == 0)
    (void)!::write(STDERR_FILENO, message, N - 1);
  std::_Exit(2);
}

}  // namespace dgpp::serve
