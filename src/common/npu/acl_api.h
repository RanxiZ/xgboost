// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_ACL_API_H_
#define XGBOOST_COMMON_NPU_ACL_API_H_
#include <acl/acl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include "runtime.h"

namespace xgboost::common::npu {
inline void CheckAcl(aclError status, char const* operation) {
  if (status != ACL_SUCCESS) {
    auto detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " status=" + std::to_string(status) +
                             (detail ? std::string(" detail=") + detail : ""));
  }
}
// Borrow the application's ACL initialization. Context and stream are private;
// this adapter deliberately exposes neither aclrtResetDevice nor aclFinalize.
inline std::shared_ptr<Api const> MakeAclApi() {
  auto a = std::make_shared<Api>();
  // CUDA's caching allocator retains geometric temporary allocations. Bound
  // our per-context idle cache to 4 GiB so the dataset-sized temporaries (the
  // 622 MiB feature-entry upload rounds to 1 GiB) stay poolable and their
  // release never needs a stream drain; live tensors remain separately owned.
  a->cache_limit_bytes = std::size_t{4} << 30;
  a->create_context = [](int ordinal) -> void* {
    aclrtContext context{};
    CheckAcl(aclrtCreateContext(&context, ordinal), "aclrtCreateContext");
    return context;
  };
  a->set_context = [](void* p) { CheckAcl(aclrtSetCurrentContext(p), "aclrtSetCurrentContext"); };
  a->destroy_context = [](void* p) { CheckAcl(aclrtDestroyContext(p), "aclrtDestroyContext"); };
  a->create_stream = []() -> void* {
    aclrtStream stream{};
    CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    return stream;
  };
  a->synchronize = [](void* p) { CheckAcl(aclrtSynchronizeStream(p), "aclrtSynchronizeStream"); };
  a->destroy_stream = [](void* p) { CheckAcl(aclrtDestroyStream(p), "aclrtDestroyStream"); };
  a->allocate = [](std::size_t n) -> void* {
    void* p{};
    CheckAcl(aclrtMalloc(&p, n, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    return p;
  };
  a->free = [](void* p) { CheckAcl(aclrtFree(p), "aclrtFree"); };
  a->copy = [](void* dst, void const* src, std::size_t n, bool upload) {
    CheckAcl(aclrtMemcpy(dst, n, src, n,
                        upload ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy");
  };
  a->zero = [](void* p, std::size_t n) { CheckAcl(aclrtMemset(p, n, 0, n), "aclrtMemset"); };
  // Asynchronous extensions: pinned host staging plus stream-ordered copies.
  a->allocate_pinned = [](std::size_t n) -> void* {
    void* p{};
    CheckAcl(aclrtMallocHost(&p, n), "aclrtMallocHost");
    return p;
  };
  a->free_pinned = [](void* p) { CheckAcl(aclrtFreeHost(p), "aclrtFreeHost"); };
  a->copy_async = [](void* dst, void const* src, std::size_t n, bool upload, void* stream) {
    CheckAcl(aclrtMemcpyAsync(dst, n, src, n,
                              upload ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST,
                              static_cast<aclrtStream>(stream)),
             "aclrtMemcpyAsync");
  };
  a->zero_async = [](void* p, std::size_t n, void* stream) {
    CheckAcl(aclrtMemsetAsync(p, n, 0, n, static_cast<aclrtStream>(stream)), "aclrtMemsetAsync");
  };
  a->create_event = []() -> void* {
    aclrtEvent event{};
    CheckAcl(aclrtCreateEvent(&event), "aclrtCreateEvent");
    return event;
  };
  a->record_event = [](void* event, void* stream) {
    CheckAcl(aclrtRecordEvent(static_cast<aclrtEvent>(event), static_cast<aclrtStream>(stream)),
             "aclrtRecordEvent");
  };
  a->synchronize_event = [](void* event) {
    CheckAcl(aclrtSynchronizeEvent(static_cast<aclrtEvent>(event)), "aclrtSynchronizeEvent");
  };
  a->destroy_event = [](void* event) {
    CheckAcl(aclrtDestroyEvent(static_cast<aclrtEvent>(event)), "aclrtDestroyEvent");
  };
  return a;
}
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_ACL_API_H_
