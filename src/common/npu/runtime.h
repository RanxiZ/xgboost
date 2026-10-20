// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_RUNTIME_H_
#define XGBOOST_COMMON_NPU_RUNTIME_H_

#include <cstddef>
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
  std::shared_ptr<Api const> api;
  void* context{nullptr};
  void* stream{nullptr};
  std::size_t live_bytes{0}, cached_bytes{0};
  std::map<std::size_t, std::vector<void*>> cache;
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
    Synchronize();
    for (auto& entry : cache) {
      auto& blocks = entry.second;
      while (!blocks.empty()) {
        api->free(blocks.back());  // Retain ownership on failure for a retry.
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
      return p;
    }
    auto allocate = [&] {
      auto p = api->allocate(capacity);
      if (!p) throw std::runtime_error("NPU allocation returned null");
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
    Synchronize();  // Reuse only after all uses on this context's stream finish.
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
    api->free(p);
  }
  explicit State(std::shared_ptr<Api const> a, int ordinal) : api{std::move(a)} {
    if (!api || ordinal < 0) throw std::invalid_argument("Invalid NPU runtime arguments");
    api->Validate();
    context = api->create_context(ordinal);
    if (!context) throw std::runtime_error("NPU context is null");
    try {
      api->set_context(context);
      stream = api->create_stream();
      if (!stream) throw std::runtime_error("NPU stream is null");
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
  void Activate() const { api->set_context(context); }
  void Synchronize() const { Activate(); api->synchronize(stream); }
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
    Cleanup([&] { api->destroy_stream(stream); });
    Cleanup([&] { api->destroy_context(context); });
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
    state_->Synchronize();
    state_->api->copy(static_cast<char*>(data_) + offset, host, bytes, true);
  }
  void Download(void* host, std::size_t bytes, std::size_t offset = 0) const {
    Bounds(host, bytes, offset);
    if (!bytes) return;
    state_->Synchronize();
    state_->api->copy(host, static_cast<char*>(data_) + offset, bytes, false);
  }
  void Zero() {
    if (!bytes_) return;
    state_->Synchronize();
    state_->api->zero(data_, bytes_);
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
  void Synchronize() const { state_->Synchronize(); }
  void* Stream() const { state_->Activate(); return state_->stream; }
  std::size_t LiveBytes() const { return state_->live_bytes; }
  std::size_t CachedBytes() const { return state_->cached_bytes; }
  void TrimCache() const { state_->TrimCache(); }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_RUNTIME_H_