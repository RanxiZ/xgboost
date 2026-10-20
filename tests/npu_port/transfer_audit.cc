// Copyright 2026, XGBoost Contributors
// Test-only LD_PRELOAD audit. Never link this into the production backend.
// Build (CANN headers only; no ACL library is linked):
// c++ -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread
//   -I/usr/local/Ascend/cann-9.1.0/x86_64-linux/include
//   tests/npu_port/transfer_audit.cc -ldl -o <fresh-directory>/libxgb_npu_transfer_audit.so
// Set XGB_NPU_TRANSFER_LOG before process startup; unset/empty disables logging.
// The file is append-only JSON Lines. Addresses and sizes, never buffer contents,
// are recorded. Audit failures terminate with exit 125, after forwarding the call.
// A missing RTLD_NEXT symbol terminates with exit 126 (forwarding is impossible).
// Events describe completed API calls; async success means enqueue success only.
// Other memcpy variants, hidden/internal calls, and non-ACL transfers are outside
// this interceptor's coverage. Correlate records with external stage markers and
// known buffer sizes; these addresses alone do not identify gradients/row IDs.
// Configuration is process-start-only. Fork before starting threads or use exec:
// like ordinary pthread mutexes, the logger is not async-signal/fork-child safe.
#include <acl/acl.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace {
[[noreturn]] void Fail(char const* message, int code) {
  // Fixed diagnostics deliberately omit environment values, paths and payloads.
  auto length = __builtin_strlen(message);
  while (length != 0) {
    auto n = ::write(STDERR_FILENO, message, length);
    if (n < 0 && errno == EINTR) { continue; }
    if (n <= 0) { break; }
    message += n;
    length -= static_cast<std::size_t>(n);
  }
  ::_exit(code);
}

template <typename Function>
Function Resolve(char const* name) {
  int saved_errno = errno;
  auto symbol = ::dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) {
    Fail("XGB_NPU_TRANSFER_AUDIT: missing RTLD_NEXT ACL symbol; cannot forward\n", 126);
  }
  errno = saved_errno;
  return reinterpret_cast<Function>(symbol);
}

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
int log_fd = -1;
bool log_initialized = false;
std::uint64_t sequence = 0;

void Record(char const* api, void const* dst, void const* src, std::size_t bytes,
            std::size_t dst_max, int kind, int policy, void const* stream, aclError result) {
  int saved_errno = errno;
  // Environment configuration is fixed on first intercepted call, as with LD_PRELOAD.
  static char const* const path = std::getenv("XGB_NPU_TRANSFER_LOG");
  if (path == nullptr || path[0] == '\0') {
    errno = saved_errno;
    return;
  }
  if (::pthread_mutex_lock(&log_mutex) != 0) {
    Fail("XGB_NPU_TRANSFER_AUDIT: cannot lock audit log\n", 125);
  }
  if (!log_initialized) {
    log_fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK,
                    S_IRUSR | S_IWUSR);
    struct stat info {};
    if (log_fd < 0 || ::fstat(log_fd, &info) != 0 || !S_ISREG(info.st_mode)) {
      Fail("XGB_NPU_TRANSFER_AUDIT: requested log cannot be opened as a regular file\n", 125);
    }
    log_initialized = true;
  }
  struct timespec now {};
  if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    Fail("XGB_NPU_TRANSFER_AUDIT: cannot read monotonic clock\n", 125);
  }
  auto ns = static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL + now.tv_nsec;
  char line[768];
  int length = std::snprintf(
      line, sizeof(line),
      "{\"audit\":1,\"pid\":%ld,\"tid\":%ld,\"seq\":%" PRIu64
      ",\"mono_ns\":%" PRIu64 ",\"api\":\"%s\",\"dst\":\"%p\",\"src\":\"%p\","
      "\"bytes\":%zu,\"dst_max\":%zu,\"kind\":%d,\"policy\":%d,\"stream\":\"%p\","
      "\"result\":%d,\"success\":%s}\n",
      static_cast<long>(::getpid()), static_cast<long>(::syscall(SYS_gettid)), ++sequence,
      static_cast<std::uint64_t>(ns), api, dst, src, bytes, dst_max, kind, policy, stream,
      static_cast<int>(result), result == ACL_SUCCESS ? "true" : "false");
  if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
    Fail("XGB_NPU_TRANSFER_AUDIT: audit record overflow\n", 125);
  }
  // One O_APPEND write per record; retry only EINTR, never a partial record.
  ssize_t written;
  do { written = ::write(log_fd, line, static_cast<std::size_t>(length)); }
  while (written < 0 && errno == EINTR);
  if (written != length) {
    Fail("XGB_NPU_TRANSFER_AUDIT: requested log write failed or was incomplete\n", 125);
  }
  if (::pthread_mutex_unlock(&log_mutex) != 0) {
    Fail("XGB_NPU_TRANSFER_AUDIT: cannot unlock audit log\n", 125);
  }
  errno = saved_errno;
}
}  // namespace

extern "C" {
aclError aclrtMemcpy(void* dst, std::size_t dst_max, void const* src, std::size_t bytes,
                     aclrtMemcpyKind kind) {
  static auto next = Resolve<decltype(&aclrtMemcpy)>("aclrtMemcpy");
  auto result = next(dst, dst_max, src, bytes, kind);
  Record("aclrtMemcpy", dst, src, bytes, dst_max, static_cast<int>(kind), -1, nullptr, result);
  return result;
}

aclError aclrtMemcpyAsync(void* dst, std::size_t dst_max, void const* src, std::size_t bytes,
                          aclrtMemcpyKind kind, aclrtStream stream) {
  static auto next = Resolve<decltype(&aclrtMemcpyAsync)>("aclrtMemcpyAsync");
  auto result = next(dst, dst_max, src, bytes, kind, stream);
  Record("aclrtMemcpyAsync", dst, src, bytes, dst_max, static_cast<int>(kind), -1, stream, result);
  return result;
}

aclError aclrtMalloc(void** ptr, std::size_t bytes, aclrtMemMallocPolicy policy) {
  static auto next = Resolve<decltype(&aclrtMalloc)>("aclrtMalloc");
  auto result = next(ptr, bytes, policy);
  Record("aclrtMalloc", result == ACL_SUCCESS && ptr != nullptr ? *ptr : nullptr, nullptr,
         bytes, 0, -1, static_cast<int>(policy), nullptr, result);
  return result;
}

aclError aclrtMallocHost(void** ptr, std::size_t bytes) {
  static auto next = Resolve<decltype(&aclrtMallocHost)>("aclrtMallocHost");
  auto result = next(ptr, bytes);
  Record("aclrtMallocHost", result == ACL_SUCCESS && ptr != nullptr ? *ptr : nullptr, nullptr,
         bytes, 0, -1, -1, nullptr, result);
  return result;
}

aclError aclrtFree(void* ptr) {
  static auto next = Resolve<decltype(&aclrtFree)>("aclrtFree");
  auto result = next(ptr);
  Record("aclrtFree", ptr, nullptr, 0, 0, -1, -1, nullptr, result);
  return result;
}

aclError aclrtFreeHost(void* ptr) {
  static auto next = Resolve<decltype(&aclrtFreeHost)>("aclrtFreeHost");
  auto result = next(ptr);
  Record("aclrtFreeHost", ptr, nullptr, 0, 0, -1, -1, nullptr, result);
  return result;
}
}  // extern "C"