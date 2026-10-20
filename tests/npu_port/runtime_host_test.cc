// Copyright 2026, XGBoost Contributors
// Host-only injection tests: no CANN headers, libraries, initialization or devices.
#include "../../src/common/npu/runtime.h"
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>

namespace npu = xgboost::common::npu;
namespace {
void Require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
template <typename F> void Throws(F f) {
  bool threw = false;
  try { f(); } catch (std::exception const&) { threw = true; }
  Require(threw, "expected exception was not raised");
}
struct Fake {
  std::shared_ptr<npu::Api> api = std::make_shared<npu::Api>();
  std::vector<std::string> events;
  std::vector<std::size_t> allocation_sizes;
  std::string fail;
  int contexts{0}, streams{0}, allocations{0}, errors{0};
  bool fail_allocate_once{false};
  void Event(std::string const& name) {
    events.push_back(name);
    if (fail == name) throw std::runtime_error("injected " + name);
  }
  Fake() {
    api->create_context = [&](int) -> void* { Event("context"); ++contexts; return this; };
    api->set_context = [&](void*) { Event("activate"); };
    api->destroy_context = [&](void*) { Event("destroy_context"); --contexts; };
    api->create_stream = [&]() -> void* { Event("stream"); ++streams; return this; };
    api->synchronize = [&](void*) { Event("sync"); };
    api->destroy_stream = [&](void*) { Event("destroy_stream"); --streams; };
    api->allocate = [&](std::size_t n) -> void* {
      Event("allocate"); allocation_sizes.push_back(n);
      if (fail_allocate_once) { fail_allocate_once = false; throw std::runtime_error("injected allocate"); }
      auto p = new char[n]; ++allocations; return p;
    };
    api->free = [&](void* p) { Event("free"); delete[] static_cast<char*>(p); --allocations; };
    api->copy = [&](void* dst, void const* src, std::size_t n, bool) {
      Event("copy"); std::memcpy(dst, src, n);
    };
    api->zero = [&](void* p, std::size_t n) { Event("zero"); std::memset(p, 0, n); };
    api->cleanup_error = [&](std::string const&) { ++errors; };
  }
  void Clean() const {
    Require(contexts == 0 && streams == 0 && allocations == 0 && errors == 0,
            "fake resource leak or cleanup error");
  }
};
template <typename F> void Test(char const* name, F f) {
  f(); std::cout << "PASS " << name << '\n';
}
}  // namespace
int main() {
  try {
    Test("bounded_slices_share_one_allocation", [] {
      Fake f;
      { npu::Runtime r{f.api, 0};
        auto base = std::make_shared<npu::Buffer>(r.AllocateElements<int64_t>(4));
        int64_t values[]{11, 22, 33, 44}, got[2]{};
        base->Upload(values, sizeof(values));
        auto a = npu::Buffer::Slice(base, sizeof(int64_t), 2 * sizeof(int64_t));
        Require(f.allocations == 1 && r.LiveBytes() == sizeof(values), "slice allocated storage");
        a.Download(got, sizeof(got)); Require(got[0] == 22 && got[1] == 33, "slice offset");
        a.Zero(); base->Download(values, sizeof(values));
        Require(values[0] == 11 && values[1] == 0 && values[2] == 0 && values[3] == 44,
                "slice zero escaped bounds");
        Throws([&] { npu::Buffer::Slice(base, sizeof(values), 1); });
        Throws([&] { npu::Buffer::Slice(base, 1, sizeof(values)); });
        Throws([&] { npu::Buffer::Slice(base, 0, 0); });
        Throws([&] { npu::Buffer::Slice(nullptr, 0, 8); });
        Throws([&] { a.Download(got, sizeof(got) + 1); });
        auto moved = std::move(a); base.reset();
        Require(f.allocations == 1 && !a.Data(), "slice failed to retain allocation");
        moved.Release(); Require(r.LiveBytes() == 0, "last slice did not release allocation"); }
      f.Clean();
    });
    Test("nested_slice_outlives_runtime", [] {
      Fake f;
      { npu::Buffer survivor;
        { npu::Runtime r{f.api, 0};
          auto base = std::make_shared<npu::Buffer>(r.Allocate(32));
          auto slice = std::make_shared<npu::Buffer>(npu::Buffer::Slice(base, 8, 16));
          survivor = npu::Buffer::Slice(slice, 8, 8); }
        Require(f.allocations == 1 && f.contexts == 1, "nested slice lifetime");
        survivor.Zero(); survivor.Release(); }
      f.Clean();
    });
    Test("empty_and_zero_byte", [] {
      Fake f;
      { npu::Buffer empty; empty.Upload(nullptr, 0); empty.Download(nullptr, 0); empty.Zero();
        npu::Runtime r{f.api, 0}; auto b = r.Allocate(0);
        Require(!b.Data() && r.LiveBytes() == 0, "zero allocation"); }
      f.Clean();
    });
    Test("roundtrip_partial_and_zero", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.AllocateElements<int64_t>(4);
        int64_t source[]{1, -2, INT64_C(1) << 50, -9}, target[4]{};
        b.Upload(source, sizeof(source)); b.Download(target, sizeof(target));
        Require(std::equal(source, source + 4, target), "roundtrip");
        b.Zero(); b.Upload(source + 2, sizeof(int64_t), 2 * sizeof(int64_t));
        b.Download(target, sizeof(target));
        Require(target[0] == 0 && target[1] == 0 && target[2] == source[2] && target[3] == 0,
                "partial copy or zero");
        Require(r.LiveBytes() == sizeof(source), "byte accounting"); }
      f.Clean();
    });
    Test("bounds_null_and_overflow_guards", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); char h[9]{};
        Throws([&] { b.Upload(h, 9); }); Throws([&] { b.Download(h, 1, 8); });
        Throws([&] { b.Upload(h, 0, 9); }); Throws([&] { b.Upload(nullptr, 1); });
        Throws([&] { b.Download(nullptr, 1); });
        Throws([&] { b.Upload(h, 1, std::numeric_limits<std::size_t>::max()); });
        Throws([&] { r.AllocateElements<int64_t>(std::numeric_limits<std::size_t>::max()); }); }
      f.Clean();
    });
    Test("buffer_outlives_runtime", [] {
      Fake f;
      { npu::Buffer survivor;
        { npu::Runtime r{f.api, 0}; survivor = r.Allocate(16); }
        Require(f.contexts == 1 && f.streams == 1, "premature context release");
        survivor.Zero(); }
      f.Clean();
      auto free = std::find(f.events.begin(), f.events.end(), "free");
      auto stream = std::find(f.events.begin(), f.events.end(), "destroy_stream");
      auto context = std::find(f.events.begin(), f.events.end(), "destroy_context");
      Require(free < stream && stream < context, "destruction order");
    });
    Test("move_and_explicit_release", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto a = r.Allocate(8); auto b = std::move(a);
        Require(!a.Data() && !a.Size(), "move source");
        auto c = r.Allocate(16); c = std::move(b);
        Require(r.LiveBytes() == 8 && !b.Data(), "move assignment");
        auto* self = &c; c = std::move(*self);
        Require(c.Size() == 8, "self move");
        c.Release(); c.Release(); Require(r.LiveBytes() == 0, "double release"); }
      f.Clean();
    });
    Test("stream_creation_failure_unwinds_context", [] {
      Fake f; f.fail = "stream"; Throws([&] { npu::Runtime r{f.api, 0}; }); f.Clean();
    });
    Test("activation_failure_unwinds_context", [] {
      Fake f; f.fail = "activate"; Throws([&] { npu::Runtime r{f.api, 0}; }); f.Clean();
    });
    Test("allocation_failure_keeps_accounting", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; f.fail = "allocate";
        Throws([&] { r.Allocate(8); }); f.fail.clear();
        Require(r.LiveBytes() == 0, "failed allocation accounted"); }
      f.Clean();
    });
    Test("failed_sync_prevents_free_and_allows_retry", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); f.fail = "sync";
        Throws([&] { b.Release(); });
        Require(f.allocations == 1 && b.Size() == 8, "freed after failed sync");
        f.fail.clear(); b.Release(); }
      f.Clean();
    });
    Test("failed_free_allows_retry", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); f.fail = "free";
        Throws([&] { b.Release(); }); Require(r.LiveBytes() == 8, "failed free accounting");
        f.fail.clear(); b.Release(); }
      f.Clean();
    });
    Test("destructor_reports_cleanup_failure", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; f.fail = "sync"; }
      Require(f.errors == 1 && f.streams == 0 && f.contexts == 0, "cleanup error lost");
    });
    Test("invalid_runtime_arguments", [] {
      Throws([] { npu::Runtime r{nullptr, 0}; });
      Fake f; Throws([&] { npu::Runtime r{f.api, -1}; });
      auto incomplete = std::make_shared<npu::Api>();
      Throws([&] { npu::Runtime r{incomplete, 0}; }); f.Clean();
    });
    Test("null_context_is_rejected", [] {
      Fake f; f.api->create_context = [](int) -> void* { return nullptr; };
      Throws([&] { npu::Runtime r{f.api, 0}; }); f.Clean();
    });
    Test("null_stream_unwinds_context", [] {
      Fake f; f.api->create_stream = []() -> void* { return nullptr; };
      Throws([&] { npu::Runtime r{f.api, 0}; }); f.Clean();
    });
    Test("null_allocation_is_rejected", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; f.api->allocate = [](std::size_t) -> void* { return nullptr; };
        Throws([&] { r.Allocate(8); }); Require(r.LiveBytes() == 0, "null allocation counted"); }
      f.Clean();
    });
    Test("context_creation_failure_has_no_resources", [] {
      Fake f; f.fail = "context"; Throws([&] { npu::Runtime r{f.api, 0}; }); f.Clean();
    });
    Test("copy_and_zero_failures_propagate", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); int64_t value = 9;
        b.Upload(&value, sizeof(value)); f.fail = "copy";
        Throws([&] { b.Upload(&value, sizeof(value)); });
        Throws([&] { b.Download(&value, sizeof(value)); });
        f.fail = "zero"; Throws([&] { b.Zero(); }); f.fail.clear();
        b.Download(&value, sizeof(value)); Require(value == 9, "copy/zero failure changed data"); }
      f.Clean();
    });
    Test("failed_activation_prevents_memory_operations", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); int64_t host = 0;
        auto before = f.events.size(); f.fail = "activate";
        Throws([&] { b.Upload(&host, sizeof(host)); });
        Throws([&] { b.Download(&host, sizeof(host)); });
        Throws([&] { b.Zero(); }); Throws([&] { b.Release(); });
        Throws([&] { r.Allocate(8); }); Throws([&] { r.Stream(); });
        Require(std::all_of(f.events.begin() + before, f.events.end(),
                           [](auto const& name) { return name == "activate"; }),
                "operation issued after failed activation"); f.fail.clear(); }
      f.Clean();
    });
    Test("failed_sync_prevents_copy_and_zero", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); int64_t host = 0;
        auto before = f.events.size(); f.fail = "sync";
        Throws([&] { b.Upload(&host, sizeof(host)); });
        Throws([&] { b.Download(&host, sizeof(host)); }); Throws([&] { b.Zero(); });
        Require(std::none_of(f.events.begin() + before, f.events.end(), [](auto const& name) {
                  return name == "copy" || name == "zero" || name == "free";
                }), "memory touched after failed synchronization"); f.fail.clear(); }
      f.Clean();
    });
    Test("failed_move_assignment_preserves_both_buffers", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto a = r.Allocate(8), b = r.Allocate(16);
        auto pa = a.Data(), pb = b.Data(); f.fail = "sync";
        Throws([&] { a = std::move(b); });
        Require(a.Data() == pa && b.Data() == pb && r.LiveBytes() == 24,
                "failed move lost ownership"); f.fail.clear(); }
      f.Clean();
    });
    Test("sync_precedes_copy_zero_and_free", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); int64_t value = 9;
        b.Upload(&value, sizeof(value)); b.Zero(); b.Download(&value, sizeof(value)); b.Release(); }
      f.Clean();
      for (std::size_t i = 0; i < f.events.size(); ++i) {
        auto const& name = f.events[i];
        if (name == "copy" || name == "zero" || name == "free") {
          Require(i >= 2 && f.events[i - 1] == "sync" && f.events[i - 2] == "activate",
                  "missing activation or synchronization before memory operation");
        }
      }
    });
    Test("accounting_overflow_rejected_before_allocate", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(8); auto events = f.events.size();
        Throws([&] { r.Allocate(std::numeric_limits<std::size_t>::max()); });
        Require(f.events.size() == events && r.LiveBytes() == 8, "overflow reached allocator"); }
      f.Clean();
    });
    Test("independent_runtimes_and_cross_runtime_move", [] {
      Fake a, b;
      { npu::Runtime ra{a.api, 0}, rb{b.api, 1};
        auto x = ra.Allocate(8), y = rb.Allocate(16); x = std::move(y);
        Require(ra.LiveBytes() == 0 && rb.LiveBytes() == 16, "cross-runtime accounting");
        x.Zero(); }
      a.Clean(); b.Clean();
    });
    Test("throwing_cleanup_reporter_does_not_terminate", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; f.fail = "sync";
        f.api->cleanup_error = [](std::string const&) { throw std::runtime_error("reporter"); }; }
      f.Clean();
    });
    Test("nonstandard_cleanup_exception_is_reported", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; f.api->synchronize = [](void*) { throw 7; }; }
      Require(f.errors == 1 && f.contexts == 0 && f.streams == 0, "unknown cleanup failure lost");
    });
    Test("zero_byte_operations_do_not_call_api", [] {
      Fake f;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(0); auto count = f.events.size();
        b.Upload(nullptr, 0); b.Download(nullptr, 0); b.Zero(); b.Release();
        Require(f.events.size() == count, "zero byte request reached API"); }
      f.Clean();
    });
    Test("cache_reuses_exact_class_without_aliasing_live", [] {
      Fake f; f.api->cache_limit_bytes = 2048;
      { npu::Runtime r{f.api, 0};
        auto a = r.Allocate(600); auto first = a.Data(); a.Release();
        auto b = r.Allocate(700); Require(b.Data() == first, "cache did not reuse class");
        Require(std::count(f.events.begin(), f.events.end(), "allocate") == 1,
                "cache hit called allocator");
        auto c = r.Allocate(600);
        Require(c.Data() != b.Data() && f.allocations == 2, "live allocation aliased");
        Require(b.Size() == 700 && c.Size() == 600 && r.LiveBytes() == 1300,
                "logical cache sizes changed");
      }
      f.Clean();
    });
    Test("cache_bucket_preserves_logical_bounds", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(513); char data[513]{};
        Require(b.Size() == 513 && r.LiveBytes() == 513, "logical size rounded");
        b.Upload(data, sizeof(data)); Throws([&] { b.Upload(data, sizeof(data) + 1); });
        Throws([&] { b.Download(data, 1, 513); });
        auto owner = std::make_shared<npu::Buffer>(std::move(b));
        Throws([&] { npu::Buffer::Slice(owner, 513, 1); });
        owner->Release(); Require(r.LiveBytes() == 0 && r.CachedBytes() == 1024,
                                "power-of-two cache bucket");
        auto small = r.Allocate(1);
        Require(f.allocation_sizes == std::vector<std::size_t>({1024, 512}),
                "wrong bucket or cross-class cache reuse");
        Require(small.Size() == 1 && r.LiveBytes() == 1 && r.CachedBytes() == 1024,
                "minimum bucket affected logical bytes");
        auto large = r.Allocate(1025); large.Release();
        Require(f.allocation_sizes.back() == 1025 && r.CachedBytes() == 1024 &&
                std::count(f.events.begin(), f.events.end(), "free") == 1,
                "allocation above cache limit was cached or rounded");
      }
      f.Clean();
    });
    Test("slice_owner_blocks_cache_reuse_until_gone", [] {
      Fake f; f.api->cache_limit_bytes = 2048;
      { npu::Runtime r{f.api, 0};
        auto base = std::make_shared<npu::Buffer>(r.Allocate(600));
        auto slice = std::make_shared<npu::Buffer>(npu::Buffer::Slice(base, 0, 600));
        auto original = base->Data(); base.reset();
        auto fresh = r.Allocate(600);
        Require(fresh.Data() != original && f.allocations == 2, "slice released owner early");
        Require(r.CachedBytes() == 0 && r.LiveBytes() == 1200,
                "live sliced allocation reached cache");
        slice.reset();
        Require(r.CachedBytes() == 1024 && r.LiveBytes() == 600,
                "last slice did not cache its owner");
        auto reused = r.Allocate(600);
        Require(f.allocations == 2 && reused.Data() == original, "slice owner was not reused");
      }
      f.Clean();
    });
    Test("bounded_cache_evicts_excess_blocks", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Runtime r{f.api, 0};
        auto a = r.Allocate(600), b = r.Allocate(600); a.Release(); b.Release();
        Require(r.CachedBytes() == 1024 && f.allocations == 1,
                "cache exceeded byte limit");
        Require(std::count(f.events.begin(), f.events.end(), "free") == 1,
                "overflow block was not freed");
      }
      f.Clean();
    });
    Test("failed_sync_keeps_buffer_out_of_cache", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(600); auto pointer = b.Data();
        f.fail = "sync"; Throws([&] { b.Release(); }); f.fail.clear();
        Require(b.Data() == pointer && b.Size() == 600 && r.LiveBytes() == 600 &&
                r.CachedBytes() == 0 && f.allocations == 1 &&
                std::count(f.events.begin(), f.events.end(), "free") == 0,
                "failed synchronization changed ownership");
        b.Release(); Require(r.LiveBytes() == 0 && r.CachedBytes() == 1024,
                             "retry did not cache buffer");
        auto count = f.events.size(); b.Release();
        Require(f.events.size() == count && r.CachedBytes() == 1024,
                "double release changed cache");
      }
      f.Clean();
    });
    Test("allocation_failure_trims_cache_and_retries", [] {
      Fake f; f.api->cache_limit_bytes = 4096;
      { npu::Runtime r{f.api, 0}; auto cached = r.Allocate(600); cached.Release();
        f.events.clear(); f.fail_allocate_once = true;
        auto b = r.Allocate(1200);
        Require(b.Size() == 1200 && r.LiveBytes() == 1200 && r.CachedBytes() == 0,
                "allocation retry did not trim cache");
        Require(std::count(f.events.begin(), f.events.end(), "allocate") == 2 &&
                std::count(f.events.begin(), f.events.end(), "free") == 1,
                "allocation failure did not free cache and retry once");
        auto first = std::find(f.events.begin(), f.events.end(), "allocate");
        auto free = std::find(f.events.begin(), f.events.end(), "free");
        auto retry = std::find(first + 1, f.events.end(), "allocate");
        Require(first < free && free < retry, "allocation retry preceded cache drain");
        auto cached_again = r.Allocate(600); cached_again.Release();
        f.fail = "allocate"; f.events.clear();
        Throws([&] { r.Allocate(2500); }); f.fail.clear();
        Require(std::count(f.events.begin(), f.events.end(), "allocate") == 2 &&
                r.LiveBytes() == 1200 && r.CachedBytes() == 0,
                "persistent allocation failure did not stop after retry");
      }
      f.Clean();
    });
    Test("cache_reuse_retains_bytes_until_explicit_zero", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Runtime r{f.api, 0}; auto a = r.Allocate(600); unsigned char value[600];
        std::memset(value, 0x5a, sizeof(value)); a.Upload(value, sizeof(value)); a.Release();
        auto b = r.Allocate(600); unsigned char got[600]{}; b.Download(got, sizeof(got));
        Require(std::equal(std::begin(value), std::end(value), std::begin(got)),
                "cached storage was implicitly zeroed");
        b.Zero(); b.Download(got, sizeof(got));
        Require(std::all_of(std::begin(got), std::end(got), [](unsigned char x) { return x == 0; }),
                "explicit zero failed on reused storage");
      }
      f.Clean();
    });
    Test("trim_cache_drains_before_runtime_destroy", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(600); b.Release();
        auto live = r.Allocate(1); auto pointer = live.Data();
        f.events.clear(); f.fail = "sync";
        Throws([&] { r.TrimCache(); }); f.fail.clear();
        Require(r.CachedBytes() == 1024 && f.allocations == 2 &&
                std::count(f.events.begin(), f.events.end(), "free") == 0,
                "failed trim synchronization freed cached storage");
        f.events.clear(); r.TrimCache();
        Require(r.CachedBytes() == 0 && std::count(f.events.begin(), f.events.end(), "free") == 1,
                "TrimCache did not drain");
        Require(r.LiveBytes() == 1 && live.Data() == pointer && f.allocations == 1,
                "TrimCache changed a live allocation");
        auto free = std::find(f.events.begin(), f.events.end(), "free");
        auto sync = std::find(f.events.begin(), f.events.end(), "sync");
        Require(sync < free, "TrimCache freed before synchronization");
      }
      f.Clean();
    });
    Test("cached_blocks_free_before_stream_and_context", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(600); b.Release(); }
      f.Clean();
      auto free = std::find(f.events.begin(), f.events.end(), "free");
      auto stream = std::find(f.events.begin(), f.events.end(), "destroy_stream");
      auto context = std::find(f.events.begin(), f.events.end(), "destroy_context");
      Require(free < stream && stream < context, "cached block destroyed too late");
    });
    Test("cache_buffer_outlives_runtime", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Buffer survivor;
        { npu::Runtime r{f.api, 0};
          auto cached = r.Allocate(1); cached.Release(); survivor = r.Allocate(600);
        }
        Require(f.allocations >= 1 && f.contexts == 1 && f.streams == 1,
                "runtime destruction invalidated live buffer");
        survivor.Zero(); survivor.Release();
      }
      f.Clean();
      auto stream = std::find(f.events.begin(), f.events.end(), "destroy_stream");
      Require(std::count(f.events.begin(), stream, "free") == 2,
              "surviving buffer storage was not freed before stream destruction");
    });
    Test("slices_survive_explicit_owner_release_and_cache_reuse", [] {
      Fake f; f.api->cache_limit_bytes = 4096;
      { npu::Runtime r{f.api, 0};
        auto owner = std::make_shared<npu::Buffer>(r.Allocate(600));
        unsigned char value = 91, got = 0; owner->Upload(&value, 1, 8);
        auto slice = npu::Buffer::Slice(owner, 8, 1); auto pointer = owner->Data();
        owner->Release(); auto unrelated = r.Allocate(600);
        Require(unrelated.Data() != pointer && r.LiveBytes() == 1200, "owner release recycled a live view");
        slice.Download(&got, 1); Require(got == value, "released owner invalidated view");
        slice.Release(); auto reused = r.Allocate(600);
        Require(reused.Data() == pointer, "final view did not recycle allocation");
      }
      f.Clean();
    });
    Test("slices_survive_owner_move_and_reassignment", [] {
      Fake f; f.api->cache_limit_bytes = 4096;
      { npu::Runtime r{f.api, 0};
        auto owner = std::make_shared<npu::Buffer>(r.Allocate(600));
        int64_t value = 123, got = 0; owner->Upload(&value, sizeof(value));
        auto slice = npu::Buffer::Slice(owner, 0, sizeof(value));
        { auto moved = std::move(*owner); }
        *owner = r.Allocate(600); owner->Zero();
        slice.Download(&got, sizeof(got)); Require(got == value, "moved owner invalidated slice");
        owner->Release(); slice.Release(); Require(r.LiveBytes() == 0, "shared allocation accounting");
      }
      f.Clean();
    });
    Test("failed_cached_free_retains_trim_ownership", [] {
      Fake f; f.api->cache_limit_bytes = 1024;
      { npu::Runtime r{f.api, 0}; auto b = r.Allocate(600); b.Release();
        f.fail = "free"; Throws([&] { r.TrimCache(); });
        Require(r.CachedBytes() == 1024 && r.LiveBytes() == 0, "failed trim lost ownership");
        f.fail.clear(); r.TrimCache(); Require(r.CachedBytes() == 0, "trim retry failed");
      }
      f.Clean();
    });
    std::cout << "RUNTIME_HOST_PASS checks=41 device_execution=not_run\n";
    return 0;
  } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
