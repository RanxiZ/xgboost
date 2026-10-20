// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_ACL_OPS_H_
#define XGBOOST_COMMON_NPU_ACL_OPS_H_

#include "acl_api.h"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_bitwise_and_scalar.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_eq_scalar.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_index_select.h>
#include <aclnnop/aclnn_left_shift.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_right_shift.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_scatter_add.h>
#include <aclnnop/aclnn_sub.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace xgboost::common::npu {
inline std::size_t ElementBytes(aclDataType type) {
  switch (type) {
    case ACL_BOOL: return 1;
    case ACL_FLOAT: case ACL_INT32: return 4;
    case ACL_INT64: case ACL_DOUBLE: return 8;
    default: throw std::invalid_argument("Unsupported NPU tensor dtype");
  }
}
// One-dimensional, owning, contiguous device tensor. Views only reinterpret
// equal-width data; they never perform a copy or floating-point conversion.
class Tensor {
  Runtime const* owner_;
  std::shared_ptr<Buffer> storage_;
  aclTensor* descriptor_{nullptr};
  int64_t count_;
  aclDataType type_;
  static std::size_t Bytes(int64_t count, aclDataType type) {
    if (count <= 0 || static_cast<uint64_t>(count) >
                          std::numeric_limits<std::size_t>::max() / ElementBytes(type)) {
      throw std::invalid_argument("Invalid NPU tensor size");
    }
    return static_cast<std::size_t>(count) * ElementBytes(type);
  }
 private:
  Tensor(Runtime const& runtime, int64_t count, aclDataType type, Buffer&& storage)
      : owner_{&runtime}, storage_{std::make_shared<Buffer>(std::move(storage))},
        count_{count}, type_{type} {
    int64_t stride = 1;
    descriptor_ = aclCreateTensor(&count_, 1, type_, &stride, 0, ACL_FORMAT_ND,
                                  &count_, 1, storage_->Data());
    if (!descriptor_) throw std::runtime_error("aclCreateTensor failed");
  }
 public:
  // Bounded views retain the underlying allocation, including after the parent
  // tensor is destroyed. Runtime and dtype are inherited, never caller supplied.
  static Tensor Slice(Tensor const& tensor, int64_t offset, int64_t count) {
    if (!tensor.storage_ || offset < 0 || offset > tensor.Count() ||
        count <= 0 || count > tensor.Count() - offset) {
      throw std::out_of_range("Invalid NPU tensor slice");
    }
    return Tensor{*tensor.Owner(), count, tensor.Type(),
                  Buffer::Slice(tensor.storage_, static_cast<std::size_t>(offset) *
                                    ElementBytes(tensor.Type()), Bytes(count, tensor.Type()))};
  }
  Tensor(Runtime const& runtime, int64_t count, aclDataType type)
      : Tensor{runtime, count, type, runtime.Allocate(Bytes(count, type))} {}
  ~Tensor() {
    if (descriptor_) {
      auto status = aclDestroyTensor(descriptor_);
      if (status != ACL_SUCCESS) std::cerr << "NPU tensor descriptor cleanup status=" << status << '\n';
    }
  }
  Tensor(Tensor const&) = delete;
  Tensor& operator=(Tensor const&) = delete;
  Tensor(Tensor&& other) noexcept
      : owner_{other.owner_}, storage_{std::move(other.storage_)},
        descriptor_{std::exchange(other.descriptor_, nullptr)}, count_{other.count_}, type_{other.type_} {}
  Tensor& operator=(Tensor&&) = delete;
  aclTensor* Descriptor() const { return descriptor_; }
  aclDataType Type() const { return type_; }
  int64_t Count() const { return count_; }
  Runtime const* Owner() const { return owner_; }
  Buffer& Storage() { return *storage_; }
  Buffer const& Storage() const { return *storage_; }
  class View {
    aclTensor* descriptor_;
   public:
    View(Tensor const& tensor, aclDataType type) {
      if (ElementBytes(type) != ElementBytes(tensor.Type())) {
        throw std::invalid_argument("NPU tensor bit view requires equal element widths");
      }
      auto count = tensor.Count();
      int64_t stride = 1;
      descriptor_ = aclCreateTensor(&count, 1, type, &stride, 0, ACL_FORMAT_ND, &count, 1,
                                    tensor.Storage().Data());
      if (!descriptor_) throw std::runtime_error("aclCreateTensor view failed");
    }
    ~View() {
      if (aclDestroyTensor(descriptor_) != ACL_SUCCESS) std::cerr << "NPU tensor view cleanup failure\n";
    }
    View(View const&) = delete;
    View& operator=(View const&) = delete;
    aclTensor* Descriptor() const { return descriptor_; }
  };
};
struct IntScalar {
  aclScalar* value;
  explicit IntScalar(int64_t n) : value{aclCreateScalar(&n, ACL_INT64)} {
    if (!value) throw std::runtime_error("aclCreateScalar failed");
  }
  ~IntScalar() {
    if (aclDestroyScalar(value) != ACL_SUCCESS) std::cerr << "NPU scalar cleanup failure\n";
  }
  IntScalar(IntScalar const&) = delete;
  IntScalar& operator=(IntScalar const&) = delete;
};
// Synchronous correctness-first operator runner. No CPU numerical fallback.
// Every workspace is retained through stream synchronization. Async batching and
// workspace reuse are separate optimizations, not performance claims here.
class Ops {
  Runtime const& runtime_;
  std::size_t launches_{0}, control_download_bytes_{0};
  void Check(Tensor const& a) const {
    if (a.Owner() != &runtime_) throw std::invalid_argument("NPU tensor belongs to a different runtime");
  }
  void Pair(Tensor const& a, Tensor const& b, bool same_type = true) const {
    Check(a); Check(b);
    if ((a.Count() != b.Count() && a.Count() != 1 && b.Count() != 1) ||
        (same_type && a.Type() != b.Type())) {
      throw std::invalid_argument("NPU operator tensor shape/dtype mismatch");
    }
  }
 public:
  explicit Ops(Runtime const& runtime) : runtime_{runtime} {}
  Runtime const& GetRuntime() const { return runtime_; }
  std::size_t Launches() const { return launches_; }
  std::size_t ControlDownloadBytes() const { return control_download_bytes_; }
  template <typename Prepare, typename Execute, typename Cancel = decltype(&aclDestroyAclOpExecutor)>
  void Run(char const* name, Prepare prepare, Execute execute,
           Cancel cancel = &aclDestroyAclOpExecutor) {
    auto stream = static_cast<aclrtStream>(runtime_.Stream());
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    CheckAcl(prepare(&bytes, &executor), name);
    Buffer workspace;
    try {
      if (bytes > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("NPU workspace too large");
      workspace = runtime_.Allocate(static_cast<std::size_t>(bytes));
    } catch (...) {
      // Phase two normally consumes a non-repeatable executor. If we cannot
      // enter phase two, explicitly cancel the successfully prepared executor.
      if (executor) {
        auto status = cancel(executor);
        if (status != ACL_SUCCESS) std::cerr << "NPU executor cancellation status=" << status << '\n';
      }
      throw;
    }
    CheckAcl(execute(workspace.Data(), bytes, executor, stream), name);
    ++launches_;
    runtime_.Synchronize();
    workspace.Release();
  }
#define XGB_NPU_RUN(op, ...) Run(#op, [&](uint64_t* n, aclOpExecutor** e) { \
    return op##GetWorkspaceSize(__VA_ARGS__, n, e); }, op)
  Tensor Constant(int64_t n, int64_t value) {
    Tensor scalar{runtime_, 1, ACL_INT64}; scalar.Storage().Upload(&value, sizeof(value));
    if (n == 1) return scalar;
    Tensor zero{runtime_, n, ACL_INT64}; zero.Storage().Zero();
    return Add(zero, scalar);
  }
  Tensor Cast(Tensor const& a, aclDataType type) {
    Check(a); Tensor out{runtime_, a.Count(), type};
    XGB_NPU_RUN(aclnnCast, a.Descriptor(), type, out.Descriptor()); return out;
  }
  Tensor FloatBits(Tensor const& a) {
    Check(a);
    if (a.Type() != ACL_FLOAT) throw std::invalid_argument("Quantizer requires float32 input");
    Tensor::View view{a, ACL_INT32};
    Tensor out{runtime_, a.Count(), ACL_INT64};
    XGB_NPU_RUN(aclnnCast, view.Descriptor(), ACL_INT64, out.Descriptor()); return out;
  }
  Tensor Add(Tensor const& a, Tensor const& b, int64_t alpha = 1) {
    Pair(a, b); Tensor out{runtime_, std::max(a.Count(), b.Count()), a.Type()}; IntScalar s{alpha};
    XGB_NPU_RUN(aclnnAdd, a.Descriptor(), b.Descriptor(), s.value, out.Descriptor()); return out;
  }
  Tensor Mul(Tensor const& a, Tensor const& b) {
    Pair(a, b); Tensor out{runtime_, std::max(a.Count(), b.Count()), a.Type()};
    XGB_NPU_RUN(aclnnMul, a.Descriptor(), b.Descriptor(), out.Descriptor()); return out;
  }
  Tensor AddScalar(Tensor const& a, int64_t value) {
    Check(a); Tensor out{runtime_, a.Count(), a.Type()}; IntScalar s{value}, one{1};
    XGB_NPU_RUN(aclnnAdds, a.Descriptor(), s.value, one.value, out.Descriptor()); return out;
  }
  Tensor AndBits(Tensor const& a, int64_t value) {
    Check(a); Tensor out{runtime_, a.Count(), a.Type()}; IntScalar s{value};
    XGB_NPU_RUN(aclnnBitwiseAndScalar, a.Descriptor(), s.value, out.Descriptor()); return out;
  }
  Tensor Shift(Tensor const& a, Tensor const& bits, bool left) {
    Pair(a, bits); Tensor out{runtime_, std::max(a.Count(), bits.Count()), a.Type()};
    if (left) { XGB_NPU_RUN(aclnnLeftShift, a.Descriptor(), bits.Descriptor(), out.Descriptor()); }
    else { XGB_NPU_RUN(aclnnRightShift, a.Descriptor(), bits.Descriptor(), out.Descriptor()); }
    return out;
  }
  Tensor Clamp(Tensor const& a, int64_t low, int64_t high) {
    Check(a); Tensor out{runtime_, a.Count(), a.Type()}; IntScalar lo{low}, hi{high};
    XGB_NPU_RUN(aclnnClamp, a.Descriptor(), lo.value, hi.value, out.Descriptor()); return out;
  }
  Tensor Compare(Tensor const& a, int64_t value, int relation) {
    Check(a); Tensor out{runtime_, a.Count(), ACL_BOOL}; IntScalar s{value};
    if (relation < 0) { XGB_NPU_RUN(aclnnLtScalar, a.Descriptor(), s.value, out.Descriptor()); }
    else if (relation > 0) { XGB_NPU_RUN(aclnnGtScalar, a.Descriptor(), s.value, out.Descriptor()); }
    else { XGB_NPU_RUN(aclnnEqScalar, a.Descriptor(), s.value, out.Descriptor()); }
    return out;
  }
  Tensor Logic(Tensor const& a, Tensor const& b, bool conjunction) {
    Pair(a, b); if (a.Type() != ACL_BOOL) throw std::invalid_argument("NPU logic requires bool tensors");
    Tensor out{runtime_, std::max(a.Count(), b.Count()), ACL_BOOL};
    if (conjunction) { XGB_NPU_RUN(aclnnLogicalAnd, a.Descriptor(), b.Descriptor(), out.Descriptor()); }
    else { XGB_NPU_RUN(aclnnLogicalOr, a.Descriptor(), b.Descriptor(), out.Descriptor()); }
    return out;
  }
  Tensor Not(Tensor const& a) {
    Check(a); if (a.Type() != ACL_BOOL) throw std::invalid_argument("NPU Not requires bool tensor");
    Tensor out{runtime_, a.Count(), ACL_BOOL};
    XGB_NPU_RUN(aclnnLogicalNot, a.Descriptor(), out.Descriptor()); return out;
  }
  Tensor Where(Tensor const& condition, Tensor const& a, Tensor const& b) {
    Pair(a, b); Pair(condition, a, false); Pair(condition, b, false);
    if (condition.Type() != ACL_BOOL) throw std::invalid_argument("NPU Where requires bool condition");
    Tensor out{runtime_, std::max({condition.Count(), a.Count(), b.Count()}), a.Type()};
    XGB_NPU_RUN(aclnnSWhere, condition.Descriptor(), a.Descriptor(), b.Descriptor(), out.Descriptor()); return out;
  }
  Tensor Sum(Tensor const& a) {
    Check(a); Tensor out{runtime_, 1, a.Type()};
    int64_t dim = 0; auto dims = aclCreateIntArray(&dim, 1);
    if (!dims) throw std::runtime_error("aclCreateIntArray failed");
    try { XGB_NPU_RUN(aclnnReduceSum, a.Descriptor(), dims, true, a.Type(), out.Descriptor()); }
    catch (...) { aclDestroyIntArray(dims); throw; }
    CheckAcl(aclDestroyIntArray(dims), "aclDestroyIntArray"); return out;
  }
  // Caller proves every partial sum fits int64 (not merely the final sum).
  Tensor ExclusiveInt64Scan(Tensor const& a) {
    Check(a);
    if (a.Type() != ACL_INT64) throw std::invalid_argument("NPU integer scan requires int64");
    Tensor inclusive{runtime_, a.Count(), ACL_INT64};
    XGB_NPU_RUN(aclnnCumsum, a.Descriptor(), 0, ACL_INT64, inclusive.Descriptor());
    // inclusive[i] - a[i] is the sum of earlier bins, including across the
    // CUDA evaluator's 32-bin chunks. No float cast or downloaded prefix.
    return Add(inclusive, a, -1);
  }
  // Caller proves nonnegative indices < input.Count() before invoking gather.
  Tensor Gather(Tensor const& input, Tensor const& indices) {
    Check(input); Check(indices);
    if (indices.Type() != ACL_INT64) throw std::invalid_argument("NPU gather needs int64 indices");
    Tensor out{runtime_, indices.Count(), input.Type()};
    XGB_NPU_RUN(aclnnIndexSelect, input.Descriptor(), 0, indices.Descriptor(), out.Descriptor());
    return out;
  }
  // Caller proves index bounds and keeps input/output storage disjoint. Used
  // by trusted row-range adapters to write directly into reusable row buffers.
  void GatherInto(Tensor const& input, Tensor const& indices, Tensor& out) {
    Check(input); Check(indices); Check(out);
    if (indices.Type() != ACL_INT64 || out.Type() != input.Type() ||
        out.Count() != indices.Count()) {
      throw std::invalid_argument("NPU gather output dtype or shape mismatch");
    }
    XGB_NPU_RUN(aclnnIndexSelect, input.Descriptor(), 0, indices.Descriptor(), out.Descriptor());
  }
  // Caller proves indices are in [0,bins); this primitive never downloads them.
  Tensor ScatterSum(Tensor const& indices, Tensor const& values, int64_t bins) {
    Pair(indices, values);
    if (indices.Type() != ACL_INT64 || indices.Count() != values.Count()) {
      throw std::invalid_argument("NPU scatter sum requires matching int64 vectors");
    }
    Tensor zero{runtime_, bins, ACL_INT64}; zero.Storage().Zero();
    Tensor out{runtime_, bins, ACL_INT64};
    XGB_NPU_RUN(aclnnScatterAdd, zero.Descriptor(), 0, indices.Descriptor(),
                values.Descriptor(), out.Descriptor());
    return out;
  }
  void DownloadControl(Tensor const& a, int64_t* output, std::size_t count) {
    Check(a);
    if (a.Type() != ACL_INT64 || count != static_cast<std::size_t>(a.Count())) {
      throw std::invalid_argument("NPU control download requires a complete int64 summary");
    }
    a.Storage().Download(output, count * sizeof(int64_t));
    control_download_bytes_ += count * sizeof(int64_t);
  }
  int64_t SumControl(Tensor const& a) {
    if (a.Type() != ACL_INT64) throw std::invalid_argument("NPU control sum requires int64");
    auto total = Sum(a); int64_t result;
    total.Storage().Download(&result, sizeof(result)); control_download_bytes_ += sizeof(result);
    return result;
  }
#undef XGB_NPU_RUN
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_ACL_OPS_H_
