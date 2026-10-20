// Copyright 2026, XGBoost Contributors
// Standalone ACLNN feasibility probe. This is not an XGBoost device backend.
#include <acl/acl.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_index_select.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_scatter_add.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_sub.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/tree/param.h"  // CPU oracle: exact functions from this checkout.

namespace {
void Check(long status, char const* operation) {
  if (status != 0) {
    auto detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " status=" + std::to_string(status) +
                             (detail ? std::string(" detail=") + detail : ""));
  }
}
#define CHECK_ACL(expr) Check((expr), #expr)

struct Params {
  float reg_alpha{0}, reg_lambda{1}, max_delta_step{0}, min_child_weight{1};
};
using xgboost::tree::CalcGain;
using xgboost::tree::CalcWeight;
using xgboost::tree::IsValidSplit;

void Compare(std::string const& name, std::vector<float> const& actual,
             std::vector<float> const& expected, float atol = 1e-5f, float rtol = 1e-5f) {
  if (actual.size() != expected.size()) throw std::runtime_error(name + ": size mismatch");
  float max_error = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    float error = std::abs(actual[i] - expected[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        error > atol + rtol * std::abs(expected[i])) {
      throw std::runtime_error(name + ": mismatch at " + std::to_string(i) +
                               " actual=" + std::to_string(actual[i]) +
                               " expected=" + std::to_string(expected[i]));
    }
    max_error = std::max(error, max_error);
  }
  std::cout << "{\"test\":\"" << name << "\",\"status\":\"passed\",\"elements\":"
            << actual.size() << ",\"max_abs_error\":" << max_error << "}" << std::endl;
}

void HostReferenceTests() {
  Params p;
  Compare("host_weight_basic", {CalcWeight(p, 4.f, 3.f)}, {-1.f});
  Compare("host_gain_basic", {CalcGain(p, 4.f, 3.f)}, {4.f});
  p.reg_alpha = 2;
  Compare("host_l1", {CalcWeight(p, 1.f, 3.f), CalcWeight(p, -6.f, 3.f)}, {0, 1});
  p.max_delta_step = 0.25f;
  Compare("host_clipped_gain", {CalcWeight(p, -6.f, 3.f), CalcGain(p, -6.f, 3.f)},
          {0.25f, 1.75f});
  Compare("host_nonpositive_hessian", {CalcWeight(p, 3.f, 0.f), CalcGain(p, 3.f, -1.f)},
          {0, 0});
  // In 3.4.1 min_child_weight constrains split validity, not CalcWeight itself.
  p = Params{};
  Compare("host_small_positive_hessian", {CalcWeight(p, 3.f, 0.5f)}, {-2.f});
  if (IsValidSplit(p, 0.5f, 2.f) || IsValidSplit(p, 0.f, 2.f) ||
      !IsValidSplit(p, 1.f, 2.f)) {
    throw std::runtime_error("host split validity semantics mismatch");
  }
  std::cout << "{\"test\":\"host_split_validity\",\"status\":\"passed\"}" << std::endl;
}

struct Scalar {
  aclScalar* value;
  explicit Scalar(float v) : value{aclCreateScalar(&v, ACL_FLOAT)} {
    if (!value) throw std::runtime_error("aclCreateScalar failed");
  }
  ~Scalar() { aclDestroyScalar(value); }
  Scalar(Scalar const&) = delete;
  Scalar& operator=(Scalar const&) = delete;
};
struct Tensor {
  void* data{nullptr};
  aclTensor* value{nullptr};
  std::size_t bytes;
  int64_t size;
  Tensor(int64_t n, aclDataType dtype, std::size_t element_size, void const* host)
      : bytes{static_cast<std::size_t>(n) * element_size}, size{n} {
    if (n <= 0) throw std::runtime_error("probe tensors must be nonempty");
    CHECK_ACL(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    try {
      if (host) CHECK_ACL(aclrtMemcpy(data, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
      else CHECK_ACL(aclrtMemset(data, bytes, 0, bytes));
      int64_t stride = 1;
      value = aclCreateTensor(&size, 1, dtype, &stride, 0, ACL_FORMAT_ND, &size, 1, data);
      if (!value) throw std::runtime_error("aclCreateTensor failed");
    } catch (...) {
      aclrtFree(data);
      throw;
    }
  }
  ~Tensor() {
    if (value) aclDestroyTensor(value);
    if (data) aclrtFree(data);
  }
  Tensor(Tensor const&) = delete;
  Tensor& operator=(Tensor const&) = delete;
  template <typename T> std::vector<T> Read() const {
    if (bytes != static_cast<std::size_t>(size) * sizeof(T)) {
      throw std::runtime_error("tensor read type size mismatch");
    }
    std::vector<T> host(size);
    CHECK_ACL(aclrtMemcpy(host.data(), bytes, data, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    return host;
  }
};
using T = Tensor*;

struct Device {
  int id;
  aclrtStream stream{nullptr};
  explicit Device(int ordinal) : id{ordinal} {
    CHECK_ACL(aclInit(nullptr));
    try {
      CHECK_ACL(aclrtSetDevice(id));
      CHECK_ACL(aclrtCreateStream(&stream));
    } catch (...) {
      aclrtResetDevice(id);
      aclFinalize();
      throw;
    }
  }
  ~Device() {
    if (stream) { aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); }
    aclrtResetDevice(id);
    aclFinalize();
  }
};

// Each operator synchronizes before temporary workspace is freed. Deliberately
// diagnostic, not a performance implementation. All tensors live in this arena.
struct Ops {
  aclrtStream stream;
  std::vector<std::unique_ptr<Tensor>> tensors;
  explicit Ops(aclrtStream s) : stream{s} {}
  T New(int64_t n, aclDataType type = ACL_FLOAT, std::size_t bytes = sizeof(float),
        void const* host = nullptr) {
    tensors.push_back(std::make_unique<Tensor>(n, type, bytes, host));
    return tensors.back().get();
  }
  T Floats(std::vector<float> const& v) { return New(v.size(), ACL_FLOAT, sizeof(float), v.data()); }
  T Ints(std::vector<int32_t> const& v) { return New(v.size(), ACL_INT32, sizeof(int32_t), v.data()); }
  T Fill(int64_t n, float v) { return Floats(std::vector<float>(n, v)); }
  template <typename Prepare, typename Execute> void Run(char const* name, Prepare prep, Execute exec) {
    uint64_t size = 0;
    aclOpExecutor* executor = nullptr;
    Check(prep(&size, &executor), name);
    void* workspace = nullptr;
    try {
      if (size) CHECK_ACL(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_NORMAL_ONLY));
      Check(exec(workspace, size, executor, stream), name);
      CHECK_ACL(aclrtSynchronizeStream(stream));
    } catch (...) {
      aclrtSynchronizeStream(stream);
      if (workspace) aclrtFree(workspace);
      throw;
    }
    if (workspace) CHECK_ACL(aclrtFree(workspace));
  }
#define RUN(op, ...) Run(#op, [&](uint64_t* size, aclOpExecutor** executor) { \
    return op##GetWorkspaceSize(__VA_ARGS__, size, executor); }, op)
  T Add(T a, T b, float alpha = 1) {
    T out = New(a->size); Scalar scale(alpha);
    RUN(aclnnAdd, a->value, b->value, scale.value, out->value); return out;
  }
  T Sub(T a, T b) { return Add(a, b, -1); }
  T Mul(T a, T b) {
    T out = New(a->size); RUN(aclnnMul, a->value, b->value, out->value); return out;
  }
  T Div(T a, T b) {
    T out = New(a->size); RUN(aclnnDiv, a->value, b->value, out->value); return out;
  }
  T Abs(T a) { T out = New(a->size); RUN(aclnnAbs, a->value, out->value); return out; }
  T Sign(T a) { T out = New(a->size); RUN(aclnnSign, a->value, out->value); return out; }
  T Clip(T a, float lo, float hi) {
    T out = New(a->size); Scalar low(lo), high(hi);
    RUN(aclnnClamp, a->value, low.value, high.value, out->value); return out;
  }
  T Gt(T a, float v) {
    T out = New(a->size, ACL_BOOL, 1); Scalar scalar(v);
    RUN(aclnnGtScalar, a->value, scalar.value, out->value); return out;
  }
  T Ge(T a, float v) {
    T out = New(a->size, ACL_BOOL, 1); Scalar scalar(v);
    RUN(aclnnGeScalar, a->value, scalar.value, out->value); return out;
  }
  T Lt(T a, float v) {
    T out = New(a->size, ACL_BOOL, 1); Scalar scalar(v);
    RUN(aclnnLtScalar, a->value, scalar.value, out->value); return out;
  }
  T And(T a, T b) {
    T out = New(a->size, ACL_BOOL, 1);
    RUN(aclnnLogicalAnd, a->value, b->value, out->value); return out;
  }
  T Where(T condition, T a, T b, bool boolean = false) {
    T out = New(a->size, boolean ? ACL_BOOL : ACL_FLOAT, boolean ? 1 : sizeof(float));
    RUN(aclnnSWhere, condition->value, a->value, b->value, out->value); return out;
  }
  T Histogram(T index, T values, int64_t bins) {
    T zero = Fill(bins, 0), out = New(bins);
    RUN(aclnnScatterAdd, zero->value, 0, index->value, values->value, out->value); return out;
  }
  T Prefix(T a) {
    T out = New(a->size); RUN(aclnnCumsum, a->value, 0, ACL_FLOAT, out->value); return out;
  }
  T Argmax(T a) {
    T out = New(1, ACL_INT64, sizeof(int64_t));
    RUN(aclnnArgMax, a->value, 0, true, out->value); return out;
  }
  T Gather(T a, T index) {
    T out = New(index->size);
    RUN(aclnnIndexSelect, a->value, 0, index->value, out->value); return out;
  }
  std::pair<T, T> WeightGain(T g, T h, Params p) {
    auto n = g->size;
    T zero = Fill(n, 0), positive = Gt(h, 0);
    T threshold = Mul(Sign(g), Clip(Sub(Abs(g), Fill(n, p.reg_alpha)), 0,
                                       std::numeric_limits<float>::max()));
    T denominator = Where(positive, Add(h, Fill(n, p.reg_lambda)), Fill(n, 1));
    T weight = Mul(Div(threshold, denominator), Fill(n, -1));
    T gain;
    if (p.max_delta_step != 0) {
      weight = Clip(weight, -p.max_delta_step, p.max_delta_step);
      T linear = Mul(Mul(g, weight), Fill(n, 2));
      T quadratic = Mul(denominator, Mul(weight, weight));
      T l1 = Mul(Abs(weight), Fill(n, 2 * p.reg_alpha));
      gain = Mul(Add(Add(linear, quadratic), l1), Fill(n, -1));
    } else {
      gain = Div(Mul(threshold, threshold), denominator);
    }
    return {Where(positive, weight, zero), Where(positive, gain, zero)};
  }
  T Valid(T left_h, T right_h, Params p) {
    return And(And(Gt(left_h, 0), Ge(left_h, p.min_child_weight)),
               And(Gt(right_h, 0), Ge(right_h, p.min_child_weight)));
  }
#undef RUN
};

void CheckMask(std::string const& name, T actual, std::vector<uint8_t> const& expected) {
  auto bytes = actual->Read<uint8_t>();
  if (bytes != expected) throw std::runtime_error(name + ": mask mismatch");
  std::cout << "{\"test\":\"" << name << "\",\"status\":\"passed\"}" << std::endl;
}

void DeviceTests(aclrtStream stream) {
  {
    Ops o(stream);
    // Last bin represents missing entries; bin 3 is deliberately empty.
    std::vector<int32_t> bins{0, 1, 1, 2, 4, 0, 2, 4};
    std::vector<float> g{-2, 1, 3, 0, -1, 2, 4, 0}, h{1, 1, 2, 0, 1, 3, 2, 0};
    auto index = o.Ints(bins);
    for (auto const& pair : {std::make_pair("gradient", g), std::make_pair("hessian", h)}) {
      std::vector<float> expected(5, 0), prefix;
      for (std::size_t i = 0; i < bins.size(); ++i) expected[bins[i]] += pair.second[i];
      prefix = expected; std::partial_sum(prefix.begin(), prefix.end(), prefix.begin());
      T hist = o.Histogram(index, o.Floats(pair.second), 5);
      Compare(std::string("npu_histogram_") + pair.first, hist->Read<float>(), expected);
      Compare(std::string("npu_prefix_") + pair.first, o.Prefix(hist)->Read<float>(), prefix);
    }
  }
  {
    Ops o(stream);
    std::vector<int32_t> bins(8192);
    std::vector<float> values(bins.size()), expected(33, 0);
    for (std::size_t i = 0; i < bins.size(); ++i) {
      bins[i] = (i * 17 + i / 7) % expected.size();
      values[i] = (static_cast<int>(i % 31) - 15) / 8.f;
      expected[bins[i]] += values[i];
    }
    Compare("npu_histogram_collisions", o.Histogram(o.Ints(bins), o.Floats(values), 33)->Read<float>(),
            expected);
  }
  for (Params p : {Params{}, Params{2, 1, 0, 1}, Params{2, 1, 0.25f, 1},
                   Params{0, 0, 0, 0}}) {
    Ops o(stream);
    std::vector<float> g{-6, -2, -1, 0, 1, 2, 6, 3, 3, 3};
    std::vector<float> h{3, 3, 3, 3, 3, 3, 3, 0.5f, 0, -1}, w, gain;
    for (std::size_t i = 0; i < g.size(); ++i) {
      w.push_back(CalcWeight(p, g[i], h[i])); gain.push_back(CalcGain(p, g[i], h[i]));
    }
    auto result = o.WeightGain(o.Floats(g), o.Floats(h), p);
    std::string suffix = "_alpha" + std::to_string(p.reg_alpha) + "_lambda" +
                         std::to_string(p.reg_lambda) + "_delta" + std::to_string(p.max_delta_step);
    Compare("npu_weight" + suffix, result.first->Read<float>(), w);
    Compare("npu_gain" + suffix, result.second->Read<float>(), gain);
    std::vector<float> rh{2, 0, 0.5f, 1, 2, 3, -1, 4, 3, 3};
    std::vector<uint8_t> valid;
    for (std::size_t i = 0; i < h.size(); ++i) valid.push_back(IsValidSplit(p, h[i], rh[i]));
    CheckMask("npu_split_validity" + suffix, o.Valid(o.Floats(h), o.Floats(rh), p), valid);
  }
  {
    Ops o(stream);
    // Candidate child sums. Both missing directions must be enumerated by the future builder.
    std::vector<float> lg{-4, -2, 0, 2}, lh{1, 2, 0, 3}, rg{4, 2, 0, -2}, rh{3, 2, 4, 1};
    Params p;
    auto left = o.WeightGain(o.Floats(lg), o.Floats(lh), p);
    auto right = o.WeightGain(o.Floats(rg), o.Floats(rh), p);
    auto parent = o.WeightGain(o.Fill(4, 0), o.Fill(4, 4), p);
    T improvement = o.Sub(o.Add(left.second, right.second), parent.second);
    T masked = o.Where(o.Valid(o.Floats(lh), o.Floats(rh), p), improvement, o.Fill(4, -1e30f));
    std::vector<float> expected;
    for (std::size_t i = 0; i < lg.size(); ++i) {
      expected.push_back(IsValidSplit(p, lh[i], rh[i]) ?
                         CalcGain(p, lg[i], lh[i]) + CalcGain(p, rg[i], rh[i]) : -1e30f);
    }
    Compare("npu_split_improvement", masked->Read<float>(), expected);
    auto best = o.Argmax(masked)->Read<int64_t>().at(0);
    if (best != std::distance(expected.begin(), std::max_element(expected.begin(), expected.end()))) {
      throw std::runtime_error("npu argmax mismatch");
    }
    // Ascend tie ordering is checked explicitly; future tree split ordering needs a separate contract.
    if (o.Argmax(o.Floats({1, 3, 3, 0}))->Read<int64_t>().at(0) != 1) {
      throw std::runtime_error("npu argmax tie does not select first index");
    }
    std::cout << "{\"test\":\"npu_argmax_and_tie\",\"status\":\"passed\"}" << std::endl;
  }
  for (bool default_left : {false, true}) {
    Ops o(stream);
    T bins = o.Ints({0, 2, -1, 1, -1, 3});
    T defaults = o.Gt(o.Fill(6, default_left ? 1 : 0), 0);
    T route = o.Where(o.Ge(bins, 0), o.Lt(bins, 2), defaults, true);
    CheckMask(default_left ? "npu_route_missing_left" : "npu_route_missing_right", route,
              {1, 0, static_cast<uint8_t>(default_left), 1, static_cast<uint8_t>(default_left), 0});
    T pred = o.Add(o.Fill(6, 0.5f), o.Where(route, o.Fill(6, -0.25f), o.Fill(6, 0.75f)));
    std::vector<float> expected{0.25f, 1.25f, default_left ? 0.25f : 1.25f,
                                0.25f, default_left ? 0.25f : 1.25f, 1.25f};
    Compare(default_left ? "npu_prediction_missing_left" : "npu_prediction_missing_right",
            pred->Read<float>(), expected);
    Compare(default_left ? "npu_gather_left" : "npu_gather_right",
            o.Gather(pred, o.Ints({5, 0, 2}))->Read<float>(), {expected[5], expected[0], expected[2]});
  }
}

void Provenance() {
  for (char const* symbol : {"aclInit", "aclnnScatterAdd", "aclnnArgMax"}) {
    Dl_info info{};
    if (!dladdr(dlsym(RTLD_DEFAULT, symbol), &info) || !info.dli_fname) {
      throw std::runtime_error(std::string("cannot resolve ") + symbol);
    }
    std::cout << "library " << symbol << " " << info.dli_fname << std::endl;
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    bool host_only = argc == 2 && std::string(argv[1]) == "--host-reference-only";
    bool device_tests = argc == 4 && std::string(argv[1]) == "--device-test" &&
                        std::string(argv[3]) == "--confirm-device-idle";
    if (!host_only && !device_tests) {
      std::cerr << "Usage: native_probe --host-reference-only\n"
                << "       native_probe --device-test DEVICE --confirm-device-idle\n"
                << "The caller must coordinate exclusive device use before device tests.\n";
      return 2;
    }
    Provenance();
    HostReferenceTests();
    if (host_only) {
      std::cout << "HOST_REFERENCE_PASS (no aclInit or device work executed)" << std::endl;
      return 0;
    }
    std::size_t consumed = 0;
    int ordinal = std::stoi(argv[2], &consumed);
    if (consumed != std::string(argv[2]).size() || ordinal < 0) {
      throw std::runtime_error("invalid device ordinal");
    }
    Device device(ordinal);
    DeviceTests(device.stream);
    // Include dynamically loaded operator libraries as provenance evidence.
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
      if (line.find("/Ascend/") != std::string::npos) std::cout << "mapping " << line << '\n';
    }
    std::cout << "NPU_PROBE_PASS (standalone operators only; not XGBoost training)" << std::endl;
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "PROBE_FAIL: " << e.what() << std::endl;
    return 1;
  }
}