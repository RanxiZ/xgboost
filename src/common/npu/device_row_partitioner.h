// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_DEVICE_ROW_PARTITIONER_H_
#define XGBOOST_COMMON_NPU_DEVICE_ROW_PARTITIONER_H_
#include "integer_histogram.h"
#include "row_compact_abi.h"
#include "row_partition_contract.h"
#include <cstddef>
#include <memory>
#include <utility>
namespace xgboost::common::npu {
// Numerical partitioning over one immutable dataset/runtime. The reference
// adapter owns independent row sets. The consuming training adapter alternates
// two full-row buffers and retains disjoint node ranges in either buffer.
// Source contract: gpu_hist/row_partitioner.cuh SortPositionBatch predicates,
// scans flags, scatters row ids, and downloads child counts. We preserve child
// membership, not its reverse-right physical order, and do not assign leaf ids.
class DeviceRowPartitioner {
  using Builder = IntegerHistogramBuilder;
  using Source = std::shared_ptr<Builder::Data const>;
  struct Arena {
    Tensor first, second;
    explicit Arena(Tensor&& root)
        : first{std::move(root)}, second{*first.Owner(), first.Count(), ACL_INT64} {}
    Tensor const& At(uint32_t slot) const { return slot == 0 ? first : second; }
  };
 public:
  // Opaque owning device row tensor. Even const Tensor exposes writable CANN
  // handles, so no Tensor, descriptor, storage, or host row export is public.
  // Empty rows have no tensor. Source ownership survives builder destruction;
  // Runtime and Ops must outlive the partitioner, row sets and histograms.
  class Rows {
    friend class DeviceRowPartitioner;
    friend class RowPartitionTestAccess;  // Defined only in standalone tests.
    Source source_;
    std::unique_ptr<Tensor> ids_;
    int64_t count_;
    std::shared_ptr<Arena> arena_;
    int64_t offset_{0};
    uint32_t slot_{0};
    Rows(Source source, std::unique_ptr<Tensor> ids, int64_t count,
         std::shared_ptr<Arena> arena = nullptr, int64_t offset = 0, uint32_t slot = 0)
        : source_{std::move(source)}, ids_{std::move(ids)}, count_{count},
          arena_{std::move(arena)}, offset_{offset}, slot_{slot} {}
   public:
    Rows(Rows&&) = default;
    Rows(Rows const&) = delete;
    Rows& operator=(Rows const&) = delete;
    int64_t Count() const { return count_; }
  };
  struct Result { Rows left, right; };
 private:
  Ops& ops_;
  Source source_;
  // Reused device scratch for the fused SIMT compact: one byte of left flag per
  // dataset row, the per-warp count array and a generation control word. Pass 1
  // publishes only the per-warp counts (each warp owns its own word, so the
  // hand-off to pass 2 is a cross-launch GM read, never an intra-launch one); the
  // block bases are folded in pass 2 by a warp-cooperative reduction. Allocated
  // once per partitioner so a node costs two launches plus one small count
  // download and no temporary tensors beyond the child views.
  Buffer compact_scratch_;
  uint32_t compact_generation_{0};
  struct CompactScratch {
    char* flags;
    int32_t* warp_counts;
    int32_t* control;
    int32_t* block_counts;       // reserved slot, not written by the current kernels
    std::size_t summary_offset;  // byte offset of the warp counts inside the scratch
    std::size_t summary_words;   // warp counts + control words in the summary
  };
  static std::size_t AlignUp32(std::size_t bytes) { return (bytes + 31u) & ~std::size_t{31}; }
  CompactScratch PrepareCompactScratch(int64_t dataset_rows) {
    constexpr auto kBlocks = rowcompact::kRowCompactBlocks;
    constexpr auto kWarps = rowcompact::kRowCompactWarps;
    auto flags_bytes = AlignUp32(static_cast<std::size_t>(dataset_rows));
    auto warp_bytes = static_cast<std::size_t>(kBlocks) * kWarps * sizeof(int32_t);
    auto control_bytes = static_cast<std::size_t>(rowcompact::kRowCompactControlWords) *
                         sizeof(int32_t);
    auto block_bytes = static_cast<std::size_t>(kBlocks) * sizeof(int32_t);
    auto total = flags_bytes + warp_bytes + control_bytes + block_bytes;
    if (!compact_scratch_.Data() || compact_scratch_.Size() < total) {
      compact_scratch_ = ops_.GetRuntime().Allocate(total);
      // The control word is read before the first violating launch writes it and
      // the CUB-style pool recycles dirty blocks: clear it so a stale stamp can
      // never alias a live generation. One-time cost per (re)allocation.
      compact_scratch_.Zero();
      compact_generation_ = 0;
    }
    auto* base = static_cast<char*>(compact_scratch_.Data());
    auto* warp_counts = reinterpret_cast<int32_t*>(base + flags_bytes);
    auto* control = warp_counts + static_cast<std::ptrdiff_t>(kBlocks) * kWarps;
    return {base, warp_counts, control, control + rowcompact::kRowCompactControlWords,
            flags_bytes,
            static_cast<std::size_t>(kBlocks) * kWarps + rowcompact::kRowCompactControlWords};
  }
  // Child row set of the fused compact. base_offset is the arena offset kept in
  // the row-set metadata; the slice is taken from the arena or from the owned
  // temporary used by the non-consuming reference path.
  Rows FusedChild(Tensor const* owned, std::shared_ptr<Arena> const& arena, uint32_t slot,
                  int64_t base_offset, int64_t slice_offset, int64_t count) {
    if (count == 0) return Rows{source_, nullptr, 0, arena, base_offset, slot};
    auto slice = arena ? Tensor::Slice(arena->At(slot), slice_offset, count)
                       : Tensor::Slice(*owned, slice_offset, count);
    return Rows{source_, std::make_unique<Tensor>(std::move(slice)), count, arena, base_offset,
                slot};
  }
  void CheckSource(Source const& source) const {
    if (!source || source.get() != source_.get()) {
      throw std::invalid_argument("NPU row partition requires identical immutable dataset provenance");
    }
  }
  void CheckRows(Rows const& rows) const {
    CheckSource(rows.source_);
    if (rows.count_ < 0 || rows.count_ > source_->gradient.Count() ||
        static_cast<bool>(rows.ids_) != (rows.count_ != 0) ||
        (rows.arena_ && (rows.slot_ > 1 || rows.offset_ < 0 ||
                        rows.offset_ > source_->gradient.Count() - rows.count_))) {
      throw std::invalid_argument("Invalid or moved-from NPU row set");
    }
  }
  void RejectIf(Tensor const& condition, char const* message) {
    if (ops_.SumControl(ops_.Cast(condition, ACL_INT64))) throw std::invalid_argument(message);
  }
  Tensor Sequence(int64_t count) { return ops_.ExclusiveInt64Scan(ops_.Constant(count, 1)); }
  Rows Empty() const { return Rows{source_, nullptr, 0}; }
  Rows Compact(Tensor const& ids, Tensor const& flag, int64_t count,
               std::shared_ptr<Arena> arena = nullptr, int64_t offset = 0, uint32_t slot = 0) {
    if (count == 0) return Rows{source_, nullptr, 0, std::move(arena), offset, slot};
    auto mask = ops_.Cast(flag, ACL_INT64);
    auto prefix = ops_.ExclusiveInt64Scan(mask);
    // Selected indices are unique [0,count). Unselected rows target a dedicated
    // zero-valued sink at m. Thus every ScatterAdd partial sum is either one
    // validated row id or zero: no overflow, float cast, or host index work.
    auto target = ops_.Where(flag, prefix, ops_.Constant(1, ids.Count()));
    auto values = ops_.Where(flag, ids, ops_.Constant(1, 0));
    auto packed = ops_.ScatterSum(target, values, ids.Count() + 1);
    if (!arena) {
      return Rows{source_, std::make_unique<Tensor>(ops_.Gather(packed, Sequence(count))), count};
    }
    auto out = std::make_unique<Tensor>(Tensor::Slice(arena->At(slot), offset, count));
    ops_.GatherInto(packed, Sequence(count), *out);
    return Rows{source_, std::move(out), count, std::move(arena), offset, slot};
  }
 public:
  explicit DeviceRowPartitioner(Ops& ops, Builder::Histogram const& context)
      : ops_{ops}, source_{context.source_} {
    if (!source_ || source_->gradient.Owner() != &ops_.GetRuntime()) {
      throw std::invalid_argument("NPU row partition histogram/runtime mismatch");
    }
  }
  // Root includes every dataset row (also zero-gradient/sampled-out rows).
  Rows Root() {
    auto n = source_->gradient.Count();
    return Rows{source_, std::make_unique<Tensor>(Sequence(n)), n};
  }
  // Production roots keep one full-row sequence and one destination buffer.
  // Descendants retain bounded views and alternate slots within their parent's
  // interval. A split consumes the parent so no live ancestor can be overwritten.
  Rows RootBuffered() {
    auto n = source_->gradient.Count();
    auto arena = std::make_shared<Arena>(Sequence(n));
    auto ids = std::make_unique<Tensor>(Tensor::Slice(arena->first, 0, n));
    return Rows{source_, std::move(ids), n, std::move(arena), 0, 0};
  }
  // Import only rows exactly matching an already validated histogram membership.
  // The row tensor is copied ON DEVICE so callers retain no writable row alias.
  // nullptr is the sole empty representation. No full row/bin downloads occur.
  Rows Capture(Builder::Histogram const& node, Tensor const* ids) {
    CheckSource(node.source_);
    auto count = ids ? ids->Count() : 0;
    if (count != node.rows_ || count > source_->gradient.Count()) {
      throw std::invalid_argument("NPU captured row count differs from histogram");
    }
    if (!ids) return Empty();
    if (ids->Owner() != &ops_.GetRuntime() || ids->Type() != ACL_INT64 || !ids->Descriptor()) {
      throw std::invalid_argument("NPU captured row tensor runtime/dtype mismatch");
    }
    auto n = source_->gradient.Count();
    RejectIf(ops_.Logic(ops_.Compare(*ids, 0, -1), ops_.Compare(*ids, n - 1, 1), false),
             "NPU captured row id out of range");
    auto membership = ops_.ScatterSum(*ids, ops_.Constant(count, 1), n);
    RejectIf(ops_.Compare(membership, 1, 1), "Duplicate NPU captured row ids");
    auto equal = ops_.Compare(ops_.Add(membership, node.membership_, -1), 0, 0);
    RejectIf(ops_.Not(equal), "NPU captured rows differ from histogram membership");
    return Rows{source_, std::make_unique<Tensor>(ops_.AddScalar(*ids, 0)), count};
  }
 private:
  Result PartitionImpl(Rows const& rows, NumericalRowSplit split, bool buffered) {
    CheckRows(rows);
    ValidateRowSplit(source_->bin_counts, split);
    auto arena = buffered ? rows.arena_ : nullptr;
    auto slot = buffered ? 1U - rows.slot_ : 0U;
    if (!rows.count_) {
      return {Rows{source_, nullptr, 0, arena, rows.offset_, slot},
              Rows{source_, nullptr, 0, arena, rows.offset_, slot}};
    }
    // Fused SIMT path: pass 1 gathers the split column once, validates the bin
    // domain and stores the left flag plus per-warp/per-block counts; pass 2
    // writes both children from that single flag (right position = node position
    // minus left prefix). No ACLNN operator, host round trip or scratch tensor
    // sits between the two launches; the only download is the count summary.
    auto scratch = PrepareCompactScratch(source_->gradient.Count());
    auto blocks = static_cast<uint32_t>(
        (rows.count_ + rowcompact::kRowCompactRowsPerBlockTarget - 1) /
        rowcompact::kRowCompactRowsPerBlockTarget);
    if (blocks > rowcompact::kRowCompactBlocks) blocks = rowcompact::kRowCompactBlocks;
    if (++compact_generation_ == 0) ++compact_generation_;  // slot 0 stays unused
    auto stream = ops_.GetRuntime().Stream();
    auto* ids = rows.ids_->Storage().Data();
    XgbNpuRowCompactCount(
        stream, ids, source_->columns[split.feature].Storage().Data(), rows.count_,
        source_->bin_counts[split.feature], split.candidate, split.missing_left ? 1U : 0U, blocks,
        scratch.flags, scratch.warp_counts, scratch.block_counts, scratch.control,
        compact_generation_);
    std::unique_ptr<Tensor> owned;
    void* left_base = nullptr;
    if (arena) {
      left_base = static_cast<char*>(arena->At(slot).Storage().Data()) +
                  rows.offset_ * static_cast<int64_t>(sizeof(int64_t));
    } else {
      owned = std::make_unique<Tensor>(ops_.GetRuntime(), rows.count_, ACL_INT64);
      left_base = owned->Storage().Data();
    }
    XgbNpuRowCompactWrite(stream, ids, scratch.flags, scratch.warp_counts,
                                      scratch.block_counts, rows.count_, blocks, left_base);
    // One small download carries the per-warp totals (the same words pass 2 folds
    // into its block bases) and the control word right after them, so the host
    // count and the device right-child offset agree bit for bit.
    int32_t summary[rowcompact::kRowCompactBlocks * rowcompact::kRowCompactWarps +
                    rowcompact::kRowCompactControlWords];
    compact_scratch_.Download(summary, sizeof(summary), scratch.summary_offset);
    int64_t count = 0;
    for (uint32_t i = 0; i < blocks * rowcompact::kRowCompactWarps; ++i) count += summary[i];
    if (summary[rowcompact::kRowCompactBlocks * rowcompact::kRowCompactWarps +
                rowcompact::kRowCompactControlViolation] ==
        static_cast<int32_t>(compact_generation_)) {
      throw std::invalid_argument("NPU row partition bin outside immutable numerical source domain");
    }
    if (count < 0 || count > rows.count_) throw std::runtime_error("Invalid NPU child count");
    // Arena children are sliced at their absolute arena offset; the owned
    // reference buffer (non-consuming path) is indexed from zero.
    auto slice_offset = arena ? rows.offset_ : 0;
    return {FusedChild(owned.get(), arena, slot, rows.offset_, slice_offset, count),
            FusedChild(owned.get(), arena, slot, rows.offset_ + count, slice_offset + count,
                       rows.count_ - count)};
  }
 public:
  // Reference path: repeated partitioning of the same immutable row set is safe.
  Result Partition(Rows const& rows, NumericalRowSplit split) {
    return PartitionImpl(rows, split, false);
  }
  // Training path: disjoint descendant ranges can progress in any order,
  // including loss-guided growth and unsplit terminal siblings. Only children
  // survive success; metadata failures leave the original parent usable.
  Result PartitionConsume(Rows&& rows, NumericalRowSplit split) {
    CheckRows(rows);
    if (!rows.arena_) throw std::invalid_argument("NPU consuming partition requires buffered rows");
    auto out = PartitionImpl(rows, split, true);
    rows.ids_.reset(); rows.arena_.reset(); rows.source_.reset(); rows.count_ = -1;
    return out;
  }
  // Trusted adapter; never publishes row tensors or mutates the dataset.
  Builder::Histogram BuildHistogram(Builder& builder, Rows const& rows) {
    CheckRows(rows); CheckSource(builder.data_);
    if (&builder.ops_.GetRuntime() != &ops_.GetRuntime()) {
      throw std::invalid_argument("NPU child histogram runtime mismatch");
    }
    return builder.Build(rows.ids_.get());
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_DEVICE_ROW_PARTITIONER_H_