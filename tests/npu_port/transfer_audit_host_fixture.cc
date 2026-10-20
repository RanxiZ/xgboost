// Copyright 2026, XGBoost Contributors
// Host-only mock and probe. No real ACL runtime is linked or loaded.
#include <acl/acl.h>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#ifdef XGB_TRANSFER_MOCK_LIBRARY
#include <fcntl.h>
#include <unistd.h>
namespace {
void Called() {
  // Proves the real (mock) callee runs even when the audit cannot open its log.
  if (auto path = std::getenv("XGB_TRANSFER_MOCK_CALLED")) {
    int fd = ::open(path, O_WRONLY | O_APPEND | O_CREAT, 0600);
    if (fd >= 0) {
      char c = 'x';
      auto ignored = ::write(fd, &c, 1);
      (void)ignored;
      ::close(fd);
    }
  }
  errno = EDOM;
}
}  // namespace
extern "C" {
aclError aclrtMemcpy(void* dst, std::size_t dst_max, void const* src, std::size_t bytes,
                     aclrtMemcpyKind) {
  Called();
  if (bytes > dst_max) { return 123; }
  std::memcpy(dst, src, bytes);
  return ACL_SUCCESS;
}
aclError aclrtMemcpyAsync(void* dst, std::size_t dst_max, void const* src, std::size_t bytes,
                          aclrtMemcpyKind kind, aclrtStream) {
  // Deliberately avoid calling the exported sync symbol: test exactly one event.
  (void)kind;
  Called();
  if (bytes > dst_max) { return 123; }
  std::memcpy(dst, src, bytes);
  return ACL_SUCCESS;
}
aclError aclrtMalloc(void** ptr, std::size_t bytes, aclrtMemMallocPolicy) {
  Called();
  if (bytes == 999999) { return 124; }
  *ptr = std::malloc(bytes);
  return *ptr == nullptr ? 124 : ACL_SUCCESS;
}
aclError aclrtMallocHost(void** ptr, std::size_t bytes) {
  Called();
  *ptr = std::malloc(bytes);
  return *ptr == nullptr ? 124 : ACL_SUCCESS;
}
aclError aclrtFree(void* ptr) { Called(); std::free(ptr); return ACL_SUCCESS; }
aclError aclrtFreeHost(void* ptr) { Called(); std::free(ptr); return ACL_SUCCESS; }
}
#else
#include <atomic>
#include <thread>
#include <vector>
int main(int argc, char**) {
  if (argc > 1) {
    char dst[4] = {}, src[4] = {'a', 'b', 'c', 'd'};
    return aclrtMemcpy(dst, sizeof(dst), src, sizeof(src), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  void* device = nullptr;
  void* host = nullptr;
  if (aclrtMalloc(&device, 32, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMallocHost(&host, 32) != ACL_SUCCESS) { return 1; }
  char src[32] = "host-only audit validation";
  for (int kind = 0; kind != 4; ++kind) {
    if (aclrtMemcpy(device, 32, src, 32, static_cast<aclrtMemcpyKind>(kind)) != ACL_SUCCESS ||
        std::memcmp(device, src, 32) != 0 || errno != EDOM) { return 2; }
  }
  if (aclrtMemcpyAsync(host, 32, device, 32, ACL_MEMCPY_DEVICE_TO_HOST, nullptr) != ACL_SUCCESS ||
      std::memcmp(host, src, 32) != 0) { return 3; }
  if (aclrtMemcpy(host, 0, src, 32, ACL_MEMCPY_DEVICE_TO_HOST) != 123) { return 4; }
  // A failed allocation must not cause the interceptor to dereference ptr.
  if (aclrtMalloc(nullptr, 999999, ACL_MEM_MALLOC_NORMAL_ONLY) != 124) { return 5; }
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (int i = 0; i != 8; ++i) {
    threads.emplace_back([&] {
      char local[32];
      for (int j = 0; j != 100; ++j) {
        if (aclrtMemcpy(local, 32, src, 32, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            std::memcmp(local, src, 32) != 0) { failed = true; }
      }
    });
  }
  for (auto& thread : threads) { thread.join(); }
  if (failed) { return 6; }
  if (aclrtFree(device) != ACL_SUCCESS || aclrtFreeHost(host) != ACL_SUCCESS) { return 7; }
  return 0;
}
#endif