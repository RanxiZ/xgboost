// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_RUNTIME_H_
#define XGBOOST_COMMON_NPU_RUNTIME_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <climits>
#include <limits>
#include <memory>
#include <map>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>

namespace xgboost::common::npu {
// Host-side counters for the NPU orchestration layer. Disabled by default and
// activated with XGB_NPU_STATS=1; a summary is printed at process exit. The
// counters exist to prove which host costs the orchestration eliminates and do
// not influence the device task order or numerical results.
struct NpuStats {
  std::uint64_t stream_syncs{0}, stream_sync_ns{0};
  std::uint64_t event_syncs{0}, event_sync_ns{0};
  std::uint64_t mallocs{0}, frees{0}, cache_hits{0}, recycles{0}, trims{0};
  std::uint64_t h2d_calls{0}, h2d_bytes{0}, d2h_calls{0}, d2h_bytes{0};
  std::uint64_t memsets{0}, memset_bytes{0}, launches{0};
  std::uint64_t tensor_creates{0}, tensor_destroys{0}, view_creates{0};
  std::uint64_t tensor_create_ns{0}, tensor_destroy_ns{0};
  std::uint64_t prepare_ns{0}, execute_ns{0};
  std::uint64_t context_switches{0};
  // Coarse host-phase wall times for the training pipeline.
  std::uint64_t bind_ns{0}, quantize_ns{0}, grow_ns{0};
  std::uint64_t partition_ns{0}, build_ns{0}, subtract_ns{0}, evaluate_ns{0};
  std::uint64_t partition_wait_ns{0}, evaluate_wait_ns{0};
  std::uint64_t bind_sketch_ns{0}, bind_layout_ns{0}, bind_upload_ns{0}, bind_validate_ns{0};
  std::uint64_t predict_ns{0}, gradient_ns{0}, export_ns{0}, upload_rows_ns{0};
  std::uint64_t dl_single{0}, dl_batch{0}, flush_calls{0}, upload_wraps{0};
  std::uint64_t dl_ops{0};
  std::uint64_t n_partitions{0}, n_builds{0}, n_eval{0};
  std::uint64_t sync_recycle{0}, sync_upload{0}, sync_download{0}, sync_batch{0},
      sync_zero{0}, sync_explicit{0};
};
inline std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline bool StatsEnabled() {
  static bool enabled = [] {
    char const* value = std::getenv("XGB_NPU_STATS");
    return value && value[0] && value[0] != '0';
  }();
  return enabled;
}
inline NpuStats& Stats() {
  static NpuStats stats;
  return stats;
}
// Total host time spent inside stream/event waits so far.
inline std::uint64_t WaitTotalNs() { return Stats().event_sync_ns + Stats().stream_sync_ns; }
inline void DumpStats() {
  auto const& s = Stats();
  std::fprintf(stderr,
               "XGB_NPU_STATS stream_syncs=%llu stream_sync_ms=%.3f event_syncs=%llu "
               "event_sync_ms=%.3f launches=%llu mallocs=%llu frees=%llu cache_hits=%llu "
               "recycles=%llu trims=%llu h2d=%llu h2d_MB=%.3f d2h=%llu d2h_KB=%.3f "
               "memsets=%llu tensor_new=%llu tensor_del=%llu view_new=%llu ctx_switch=%llu "
               "tensor_new_ms=%.3f tensor_del_ms=%.3f prepare_ms=%.3f execute_ms=%.3f "
               "bind_ms=%.3f quantize_ms=%.3f grow_ms=%.3f partition_ms=%.3f build_ms=%.3f "
               "subtract_ms=%.3f evaluate_ms=%.3f partition_wait_ms=%.3f evaluate_wait_ms=%.3f "
               "bind_sketch_ms=%.3f bind_layout_ms=%.3f bind_upload_ms=%.3f bind_validate_ms=%.3f "
               "predict_ms=%.3f gradient_ms=%.3f export_ms=%.3f upload_rows_ms=%.3f "
               "dl_single=%llu dl_batch=%llu flush_calls=%llu upload_wraps=%llu dl_ops=%llu "
               "n_partitions=%llu n_builds=%llu n_eval=%llu "
               "sync_recycle=%llu sync_upload=%llu sync_download=%llu sync_batch=%llu "
               "sync_zero=%llu sync_explicit=%llu\n",
               static_cast<unsigned long long>(s.stream_syncs),
               static_cast<double>(s.stream_sync_ns) / 1e6,
               static_cast<unsigned long long>(s.event_syncs),
               static_cast<double>(s.event_sync_ns) / 1e6,
               static_cast<unsigned long long>(s.launches),
               static_cast<unsigned long long>(s.mallocs),
               static_cast<unsigned long long>(s.frees),
               static_cast<unsigned long long>(s.cache_hits),
               static_cast<unsigned long long>(s.recycles),
               static_cast<unsigned long long>(s.trims),
               static_cast<unsigned long long>(s.h2d_calls),
               static_cast<double>(s.h2d_bytes) / (1024.0 * 1024.0),
               static_cast<unsigned long long>(s.d2h_calls),
               static_cast<double>(s.d2h_bytes) / 1024.0,
               static_cast<unsigned long long>(s.memsets),
               static_cast<unsigned long long>(s.tensor_creates),
               static_cast<unsigned long long>(s.tensor_destroys),
               static_cast<unsigned long long>(s.view_creates),
               static_cast<unsigned long long>(s.context_switches),
               static_cast<double>(s.tensor_create_ns) / 1e6,
               static_cast<double>(s.tensor_destroy_ns) / 1e6,
               static_cast<double>(s.prepare_ns) / 1e6,
               static_cast<double>(s.execute_ns) / 1e6,
               static_cast<double>(s.bind_ns) / 1e6,
               static_cast<double>(s.quantize_ns) / 1e6,
               static_cast<double>(s.grow_ns) / 1e6,
               static_cast<double>(s.partition_ns) / 1e6,
               static_cast<double>(s.build_ns) / 1e6,
               static_cast<double>(s.subtract_ns) / 1e6,
               static_cast<double>(s.evaluate_ns) / 1e6,
               static_cast<double>(s.partition_wait_ns) / 1e6,
               static_cast<double>(s.evaluate_wait_ns) / 1e6,
               static_cast<double>(s.bind_sketch_ns) / 1e6,
               static_cast<double>(s.bind_layout_ns) / 1e6,
               static_cast<double>(s.bind_upload_ns) / 1e6,
               static_cast<double>(s.bind_validate_ns) / 1e6,
               static_cast<double>(s.predict_ns) / 1e6,
               static_cast<double>(s.gradient_ns) / 1e6,
               static_cast<double>(s.export_ns) / 1e6,
               static_cast<double>(s.upload_rows_ns) / 1e6,
               static_cast<unsigned long long>(s.dl_single),
               static_cast<unsigned long long>(s.dl_batch),
               static_cast<unsigned long long>(s.flush_calls),
               static_cast<unsigned long long>(s.upload_wraps),
               static_cast<unsigned long long>(s.dl_ops),
               static_cast<unsigned long long>(s.n_partitions),
               static_cast<unsigned long long>(s.n_builds),
               static_cast<unsigned long long>(s.n_eval),
               static_cast<unsigned long long>(s.sync_recycle),
               static_cast<unsigned long long>(s.sync_upload),
               static_cast<unsigned long long>(s.sync_download),
               static_cast<unsigned long long>(s.sync_batch),
               static_cast<unsigned long long>(s.sync_zero),
               static_cast<unsigned long long>(s.sync_explicit));
  std::fflush(stderr);
}
inline void EnsureStatsHook() {
  static bool once = [] {
    if (StatsEnabled()) std::atexit(&DumpStats);
    return true;
  }();
  static_cast<void>(once);
}
// The last context selected on this host thread. The one host thread owns both
// the training runtime and the prediction runtime, so skipping a repeated
// aclrtSetCurrentContext is safe as long as destruction clears the cache.
inline thread_local void* tls_current_context = nullptr;

// Single-host-thread runtime. The application owns process-wide ACL initialization
// and must finalize ACL only after all runtimes/buffers have been released.
// No device reset, process-global finalize or CPU fallback occurs in this layer.
// Calls activate this runtime's context and leave it current on the calling
// thread. Callers sharing that thread with another runtime must reactivate their
// own context before issuing operations. All use, including destruction, must
// remain on one host thread. Raw stream/data handles borrow their owner's life.
// The injected interface also permits lifecycle tests without loading CANN.
struct Api {
  // Injection defaults to uncached allocations; the production adapter enables
  // a bounded, context-local analogue of CUDA's CUB caching allocator.
  std::size_t cache_limit_bytes{0};
  std::function<void*(int)> create_context;
  std::function<void(void*)> set_context;
  std::function<void(void*)> destroy_context;
  std::function<void*()> create_stream;
  std::function<void(void*)> synchronize;
  std::function<void(void*)> destroy_stream;
  std::function<void*(std::size_t)> allocate;
  std::function<void(void*)> free;
  std::function<void(void*, void const*, std::size_t, bool)> copy;  // true: H2D
  std::function<void(void*, std::size_t)> zero;
  // Optional asynchronous extensions. When every entry below is present the
  // runtime stages host copies through pinned memory and keeps all transfers
  // stream-ordered; otherwise it falls back to the synchronous primitives above.
  std::function<void*(std::size_t)> allocate_pinned;
  std::function<void(void*)> free_pinned;
  std::function<void(void*, void const*, std::size_t, bool, void*)> copy_async;
  std::function<void(void*, std::size_t, void*)> zero_async;
  std::function<void*()> create_event;
  std::function<void(void*, void*)> record_event;  // (event, stream)
  std::function<void(void*)> synchronize_event;
  std::function<void(void*)> destroy_event;
  bool Async() const {
    return allocate_pinned && free_pinned && copy_async && zero_async && create_event &&
           record_event && synchronize_event && destroy_event;
  }
  std::function<void(std::string const&)> cleanup_error = [](std::string const& s) {
    std::cerr << "NPU cleanup failure: " << s << '\n';
  };
  void Validate() const {
    if (!create_context || !set_context || !destroy_context || !create_stream ||
        !synchronize || !destroy_stream || !allocate || !free || !copy || !zero ||
        !cleanup_error) {
      throw std::invalid_argument("Incomplete NPU runtime API");
    }
  }
};

struct State {
  // Pinned staging bound for asynchronous host copies. Larger transfers (dataset
  // binding, row uploads) keep the synchronous path; they are not per-node work.
  static constexpr std::size_t kStagingLimit = std::size_t{8} << 20;
  std::shared_ptr<Api const> api;
  void* context{nullptr};
  void* stream{nullptr};
  std::size_t live_bytes{0}, cached_bytes{0};
  std::map<std::size_t, std::vector<void*>> cache;
  // Asynchronous copy state (single host thread, one stream).
  void* upload_staging{nullptr};
  std::size_t upload_staging_bytes{0}, upload_staging_used{0};
  bool upload_recorded{false};
  void* download_staging{nullptr};
  std::size_t download_staging_bytes{0};
  void* upload_event{nullptr};
  void* download_event{nullptr};
  // Same geometric classes as device_vector.cuh's CUB allocator: minimum 512
  // bytes, powers of two. Logical buffer bounds never include bucket padding.
  std::size_t Capacity(std::size_t bytes) const {
    if (!api->cache_limit_bytes || bytes > api->cache_limit_bytes) return bytes;
    std::size_t size = 512;
    while (size < bytes) {
      if (size > std::numeric_limits<std::size_t>::max() / 2) return bytes;
      size *= 2;
    }
    return size <= api->cache_limit_bytes ? size : bytes;
  }
  void TrimCache() {
    // Cached blocks may still be referenced by queued device work, so the
    // whole-stream drain stays on this (rare) allocation-pressure path.
    Synchronize();
    if (StatsEnabled()) ++Stats().trims;
    for (auto& entry : cache) {
      auto& blocks = entry.second;
      while (!blocks.empty()) {
        api->free(blocks.back());  // Retain ownership on failure for a retry.
        if (StatsEnabled()) ++Stats().frees;
        blocks.pop_back();
        cached_bytes -= entry.first;
      }
    }
  }
  void* Acquire(std::size_t capacity) {
    Activate();
    auto found = cache.find(capacity);
    if (found != cache.end() && !found->second.empty()) {
      auto p = found->second.back(); found->second.pop_back();
      cached_bytes -= capacity;
      if (StatsEnabled()) ++Stats().cache_hits;
      return p;
    }
    auto allocate = [&] {
      auto p = api->allocate(capacity);
      if (!p) throw std::runtime_error("NPU allocation returned null");
      if (StatsEnabled()) ++Stats().mallocs;
      return p;
    };
    try { return allocate(); } catch (...) {
      // Like CUB, release cached storage under pressure before a single retry.
      if (!cached_bytes) throw;
      TrimCache();
      return allocate();
    }
  }
  void Recycle(void* p, std::size_t capacity) {
    if (StatsEnabled()) ++Stats().recycles;
    // Single-stream semantics: every consumer of this block was enqueued before
    // its release, and any later reuse is enqueued after them, so returning the
    // block to the pool needs no host wait. Blocks that cannot be pooled are
    // freed, and freeing memory still referenced by queued work is unsafe, so
    // only that (non-cached) path drains the stream first.
    if (capacity <= api->cache_limit_bytes &&
        capacity <= api->cache_limit_bytes - cached_bytes) {
      try {
        cache[capacity].push_back(p);
        cached_bytes += capacity;
        return;
      } catch (std::bad_alloc const&) {
        // A host bookkeeping failure must not lose the device allocation.
      }
    }
    if (StatsEnabled()) ++Stats().sync_recycle;
    Synchronize();
    api->free(p);
  }
  explicit State(std::shared_ptr<Api const> a, int ordinal) : api{std::move(a)} {
    if (!api || ordinal < 0) throw std::invalid_argument("Invalid NPU runtime arguments");
    api->Validate();
    EnsureStatsHook();
    context = api->create_context(ordinal);
    if (!context) throw std::runtime_error("NPU context is null");
    try {
      api->set_context(context);
      tls_current_context = context;
      stream = api->create_stream();
      if (!stream) throw std::runtime_error("NPU stream is null");
      if (api->Async()) {
        upload_event = api->create_event();
        download_event = api->create_event();
      }
    } catch (...) {
      Cleanup([&] { api->destroy_context(context); });
      throw;
    }
  }
  State(State const&) = delete;
  State& operator=(State const&) = delete;
  template <typename F> void Cleanup(F f) noexcept {
    try { f(); } catch (std::exception const& e) {
      try { api->cleanup_error(e.what()); } catch (...) {}
    } catch (...) {
      try { api->cleanup_error("unknown exception"); } catch (...) {}
    }
  }
  void Activate() const {
    if (tls_current_context != context) {
      api->set_context(context);
      tls_current_context = context;
      if (StatsEnabled()) ++Stats().context_switches;
    }
  }
  void Synchronize() const {
    Activate();
    if (!StatsEnabled()) {
      api->synchronize(stream);
      return;
    }
    auto begin = std::chrono::steady_clock::now();
    api->synchronize(stream);
    Stats().stream_syncs += 1;
    Stats().stream_sync_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
  }
  // Host-to-device copy. When asynchronous primitives are available the host
  // bytes are staged in pinned memory and the copy is queued behind the current
  // stream tail, so no host wait is required here.
  void CopyUpload(void* dst, void const* src, std::size_t bytes) {
    Activate();
    if (StatsEnabled()) { ++Stats().h2d_calls; Stats().h2d_bytes += bytes; }
    if (!api->Async()) {
      if (StatsEnabled()) ++Stats().sync_upload;
      Synchronize();
      api->copy(dst, src, bytes, true);
      return;
    }
    if (!upload_staging) {
      upload_staging = api->allocate_pinned(kStagingLimit);
      upload_staging_bytes = kStagingLimit;
      upload_staging_used = 0;
      upload_recorded = false;
    }
    static constexpr std::size_t kAlign = 64;
    // Large transfers (dataset binding, row upload) stream through the same
    // pinned ring in chunks: the CPU reads pageable host memory into pinned
    // staging at memory speed and the chunk copies stay stream-ordered, which
    // is far faster than one synchronous pageable copy per call.
    std::size_t offset = 0;
    while (offset < bytes) {
      std::size_t chunk = std::min(bytes - offset, upload_staging_bytes);
      std::size_t aligned = (chunk + (kAlign - 1)) & ~(kAlign - 1);
      if (upload_staging_used + aligned > upload_staging_bytes) {
        // The event is re-recorded by every upload on the same stream, so waiting
        // for the latest record also retires every earlier staged copy.
        if (upload_recorded) {
          if (StatsEnabled()) ++Stats().upload_wraps;
          WaitEvent(upload_event);
          upload_recorded = false;
        }
        upload_staging_used = 0;
      }
      auto* slot = static_cast<char*>(upload_staging) + upload_staging_used;
      std::memcpy(slot, static_cast<char const*>(src) + offset, chunk);
      api->copy_async(static_cast<char*>(dst) + offset, slot, chunk, true, stream);
      api->record_event(upload_event, stream);
      upload_recorded = true;
      upload_staging_used += aligned;
      offset += chunk;
    }
  }
  // Device-to-host copy: the caller needs the bytes, so this path waits for the
  // queued copy (which is ordered behind all producers on the stream).
  void CopyDownload(void* dst, void const* src, std::size_t bytes) {
    Activate();
    if (StatsEnabled()) { ++Stats().d2h_calls; Stats().d2h_bytes += bytes; }
    if (!api->Async() || bytes > kStagingLimit) {
      if (StatsEnabled()) ++Stats().sync_download;
      Synchronize();
      api->copy(dst, src, bytes, false);
      return;
    }
    if (download_staging_bytes < bytes) {
      if (download_staging) {
        api->free_pinned(download_staging);
        download_staging = nullptr;
        download_staging_bytes = 0;
      }
      std::size_t want = std::size_t{64} << 10;
      while (want < bytes) want <<= 1;
      download_staging = api->allocate_pinned(want);
      download_staging_bytes = want;
    }
    if (StatsEnabled()) ++Stats().dl_single;
    api->copy_async(download_staging, src, bytes, false, stream);
    api->record_event(download_event, stream);
    WaitEvent(download_event);
    std::memcpy(dst, download_staging, bytes);
  }
  // Batched device-to-host copies: queue every copy, record the event once and
  // perform a single host wait. Used to read back deferred validation scalars so
  // a whole expansion batch shares one stream drain instead of one per check.
  struct CopyRequest {
    void* dst;
    void const* src;
    std::size_t bytes;
  };
  void CopyDownloadBatch(std::vector<CopyRequest> const& requests) {
    Activate();
    if (requests.empty()) return;
    std::size_t total = 0;
    for (auto const& request : requests) {
      if (!request.dst || !request.src || !request.bytes) {
        throw std::invalid_argument("Invalid NPU batched download");
      }
      if (request.bytes > kStagingLimit - std::min(total, kStagingLimit)) {
        total = kStagingLimit + 1;
        break;
      }
      total += request.bytes;
    }
    if (!api->Async() || total > kStagingLimit) {
      if (StatsEnabled()) ++Stats().sync_batch;
      for (auto const& request : requests) CopyDownload(request.dst, request.src, request.bytes);
      return;
    }
    if (StatsEnabled()) {
      for (auto const& request : requests) {
        ++Stats().d2h_calls;
        Stats().d2h_bytes += request.bytes;
      }
    }
    if (download_staging_bytes < total) {
      if (download_staging) {
        api->free_pinned(download_staging);
        download_staging = nullptr;
        download_staging_bytes = 0;
      }
      std::size_t want = std::size_t{64} << 10;
      while (want < total) want <<= 1;
      download_staging = api->allocate_pinned(want);
      download_staging_bytes = want;
    }
    if (StatsEnabled()) ++Stats().dl_batch;
    auto* slot = static_cast<char*>(download_staging);
    for (auto const& request : requests) {
      api->copy_async(slot, request.src, request.bytes, false, stream);
      slot += request.bytes;
    }
    api->record_event(download_event, stream);
    WaitEvent(download_event);
    auto const* staged = static_cast<char const*>(download_staging);
    for (auto const& request : requests) {
      std::memcpy(request.dst, staged, request.bytes);
      staged += request.bytes;
    }
  }
  void ZeroDevice(void* p, std::size_t bytes) {
    Activate();
    if (StatsEnabled()) { ++Stats().memsets; Stats().memset_bytes += bytes; }
    if (!api->Async()) {
      if (StatsEnabled()) ++Stats().sync_zero;
      Synchronize();
      api->zero(p, bytes);
      return;
    }
    api->zero_async(p, bytes, stream);
  }
  void WaitEvent(void* event) const {
    if (!StatsEnabled()) {
      api->synchronize_event(event);
      return;
    }
    auto begin = std::chrono::steady_clock::now();
    api->synchronize_event(event);
    Stats().event_syncs += 1;
    Stats().event_sync_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
  }
  ~State() {
    // A failed context switch makes subsequent context-dependent cleanup unsafe.
    try { Activate(); } catch (...) {
      Cleanup([] { throw std::runtime_error("Cannot activate NPU context for cleanup"); });
      return;
    }
    Cleanup([&] { api->synchronize(stream); });
    // Cached blocks were synchronized before admission and cannot be live.
    for (auto& entry : cache) {
      for (auto p : entry.second) Cleanup([&] { api->free(p); });
    }
    if (upload_event) Cleanup([&] { api->destroy_event(upload_event); });
    if (download_event) Cleanup([&] { api->destroy_event(download_event); });
    if (upload_staging) Cleanup([&] { api->free_pinned(upload_staging); });
    if (download_staging) Cleanup([&] { api->free_pinned(download_staging); });
    Cleanup([&] { api->destroy_stream(stream); });
    Cleanup([&] { api->destroy_context(context); });
    if (tls_current_context == context) tls_current_context = nullptr;
  }
};

class Buffer {
  // Allocation ownership is independent of the mutable Buffer object. Moving,
  // releasing or reassigning an owner cannot invalidate a surviving slice.
  struct Allocation {
    std::shared_ptr<State> state;
    void* data;
    std::size_t bytes, capacity;
    Allocation(std::shared_ptr<State> s, std::size_t n)
        : state{std::move(s)}, data{nullptr}, bytes{n}, capacity{state->Capacity(n)} {
      data = state->Acquire(capacity);
      state->live_bytes += bytes;
    }
    void Release() {
      if (!data) return;
      state->Recycle(data, capacity);
      state->live_bytes -= bytes;
      data = nullptr;
    }
    ~Allocation() { if (data) state->Cleanup([&] { Release(); }); }
  };
  std::shared_ptr<State> state_;
  std::shared_ptr<Allocation> allocation_;
  void* data_{nullptr};
  std::size_t bytes_{0};
  void Bounds(void const* host, std::size_t bytes, std::size_t offset) const {
    if (offset > bytes_ || bytes > bytes_ - offset) {
      throw std::out_of_range("NPU buffer copy exceeds allocation");
    }
    if (bytes && !host) throw std::invalid_argument("Null host pointer for NPU copy");
  }
 public:
  Buffer() = default;
  Buffer(std::shared_ptr<State> state, std::size_t bytes) : state_{std::move(state)} {
    if (!state_) throw std::invalid_argument("NPU buffer has no runtime");
    if (bytes > std::numeric_limits<std::size_t>::max() - state_->live_bytes) {
      throw std::overflow_error("NPU allocation accounting overflow");
    }
    if (bytes) {
      allocation_ = std::make_shared<Allocation>(state_, bytes);
      data_ = allocation_->data;
      bytes_ = bytes;
    }
  }
  static Buffer Slice(std::shared_ptr<Buffer const> backing, std::size_t offset,
                      std::size_t bytes) {
    if (!backing || !bytes || offset > backing->Size() || bytes > backing->Size() - offset) {
      throw std::out_of_range("Invalid NPU buffer slice");
    }
    Buffer out;
    out.state_ = backing->state_;
    out.data_ = static_cast<char*>(backing->Data()) + offset;
    out.bytes_ = bytes;
    out.allocation_ = backing->allocation_;
    return out;
  }
  ~Buffer() = default;
  Buffer(Buffer const&) = delete;
  Buffer& operator=(Buffer const&) = delete;
  Buffer(Buffer&& other) noexcept
      : state_{std::move(other.state_)}, allocation_{std::move(other.allocation_)},
        data_{std::exchange(other.data_, nullptr)}, bytes_{std::exchange(other.bytes_, 0)} {}
  // Explicit release before reassignment prevents hiding failed cleanup.
  Buffer& operator=(Buffer&& other) {
    if (this != &other) {
      Release();
      state_ = std::move(other.state_);
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
      allocation_ = std::move(other.allocation_);
    }
    return *this;
  }
  void* Data() const { return data_; }
  std::size_t Size() const { return bytes_; }
  void Release() {
    if (!data_) return;
    if (allocation_.use_count() == 1) allocation_->Release();
    allocation_.reset();
    data_ = nullptr;
    bytes_ = 0;
  }
  void Upload(void const* host, std::size_t bytes, std::size_t offset = 0) {
    Bounds(host, bytes, offset);
    if (!bytes) return;
    state_->CopyUpload(static_cast<char*>(data_) + offset, host, bytes);
  }
  void Download(void* host, std::size_t bytes, std::size_t offset = 0) const {
    Bounds(host, bytes, offset);
    if (!bytes) return;
    state_->CopyDownload(host, static_cast<char const*>(data_) + offset, bytes);
  }
  void Zero() {
    if (!bytes_) return;
    state_->ZeroDevice(data_, bytes_);
  }
};

class Runtime {
  std::shared_ptr<State> state_;
 public:
  explicit Runtime(std::shared_ptr<Api const> api, int ordinal)
      : state_{std::make_shared<State>(std::move(api), ordinal)} {}
  Buffer Allocate(std::size_t bytes) const { return Buffer{state_, bytes}; }
  template <typename T> Buffer AllocateElements(std::size_t count) const {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("NPU element byte count overflow");
    }
    return Allocate(count * sizeof(T));
  }
  void Synchronize() const {
    if (StatsEnabled()) ++Stats().sync_explicit;
    state_->Synchronize();
  }
  void DownloadBatch(std::vector<State::CopyRequest> const& requests) const {
    state_->CopyDownloadBatch(requests);
  }
  void* Stream() const { state_->Activate(); return state_->stream; }
  std::size_t LiveBytes() const { return state_->live_bytes; }
  std::size_t CachedBytes() const { return state_->cached_bytes; }
  void TrimCache() const { state_->TrimCache(); }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_RUNTIME_H_
