// Copyright 2026, XGBoost Contributors
// Test-only LD_PRELOAD guard: aborts before any runtime operation can run.
// It intentionally does not link CANN. Used only by check_runtime.py, never by
// device tests or the production adapter.
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>

namespace {
[[noreturn]] void Block(char const* name) {
  std::fprintf(stderr, "FORBIDDEN_ACL_CALL %s\n", name);
  std::_Exit(86);
}
}  // namespace
extern "C" {
aclError aclInit(char const*) { Block("aclInit"); }
aclError aclFinalize() { Block("aclFinalize"); }
aclError aclrtSetDevice(int32_t) { Block("aclrtSetDevice"); }
aclError aclrtResetDevice(int32_t) { Block("aclrtResetDevice"); }
aclError aclrtCreateContext(aclrtContext*, int32_t) { Block("aclrtCreateContext"); }
aclError aclrtSetCurrentContext(aclrtContext) { Block("aclrtSetCurrentContext"); }
aclError aclrtDestroyContext(aclrtContext) { Block("aclrtDestroyContext"); }
aclError aclrtCreateStream(aclrtStream*) { Block("aclrtCreateStream"); }
aclError aclrtSynchronizeStream(aclrtStream) { Block("aclrtSynchronizeStream"); }
aclError aclrtDestroyStream(aclrtStream) { Block("aclrtDestroyStream"); }
aclError aclrtMalloc(void**, std::size_t, aclrtMemMallocPolicy) { Block("aclrtMalloc"); }
aclError aclrtFree(void*) { Block("aclrtFree"); }
aclError aclrtMemcpy(void*, std::size_t, void const*, std::size_t, aclrtMemcpyKind) {
  Block("aclrtMemcpy");
}
aclError aclrtMemset(void*, std::size_t, int32_t, std::size_t) { Block("aclrtMemset"); }
}
