// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_PRECISE_ARITHMETIC_ABI_H_
#define XGBOOST_TESTS_NPU_PORT_PRECISE_ARITHMETIC_ABI_H_
#include <cstdint>
namespace precise_test {
struct ArithmeticInput { uint64_t a,b; uint32_t operation,padding{0}; };
struct ArithmeticOutput { uint64_t value; uint32_t error,padding{0}; };
static_assert(sizeof(ArithmeticInput)==24 && sizeof(ArithmeticOutput)==16,"NPU test ABI layout mismatch");
}
extern "C" void XgbNpuTestArithmetic(void* stream,void const* input,void* output,uint32_t count);
#endif  // XGBOOST_TESTS_NPU_PORT_PRECISE_ARITHMETIC_ABI_H_
