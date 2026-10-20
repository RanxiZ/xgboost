/**
 * Copyright 2014-2026, XGBoost Contributors
 *
 * \brief Context object used for controlling runtime parameters.
 */
#include "xgboost/context.h"

#include <algorithm>  // for find_if
#include <cctype>     // for isdigit
#include <charconv>   // for from_chars
#include <iterator>   // for distance
#include <limits>     // for numeric_limits
#include <optional>   // for optional
#include <regex>      // for regex_replace, regex_match
#include <sstream>    // for stringstream

#include "common/cuda_rt_utils.h"  // for AllVisibleGPUs
#include "common/random.h"
#include "common/threading_utils.h"
#include "xgboost/json.h"  // for Json, Object, String, ToJson, FromJson
#include "xgboost/string_view.h"

#if !defined(XGBOOST_USE_CUDA)

#include "common/common.h"  // for AssertGPUSupport

#endif  // !defined(XGBOOST_USE_CUDA)

#if defined(XGBOOST_USE_SYCL)
#include "../plugin/sycl/context_helper.h"
#endif  // defined (XGBOOST_USE_SYCL)

namespace xgboost {

DMLC_REGISTER_PARAMETER(Context);

std::int64_t constexpr Context::kDefaultSeed;

Context::Context() : cfs_cpu_count_{common::GetCfsCPUCount()} {}

namespace {
inline constexpr char const* kDevice = "device";

#if !defined(XGBOOST_USE_CUDA)
DeviceOrd CUDAOrdinal(DeviceOrd device, bool) {
  device = DeviceOrd::CPU();
  return device;
}
#else
// Check CUDA on the current device, wrap the ordinal if necessary.
[[nodiscard]] DeviceOrd CUDAOrdinal(DeviceOrd device, bool fail_on_invalid) {
  // When booster is loaded from a memory image (Python pickle or R raw model), number of
  // available GPUs could be different.  Wrap around it.
  std::int32_t n_visible = curt::AllVisibleGPUs();
  if (n_visible == 0) {
    if (device.IsCUDA()) {
      LOG(WARNING) << "No visible GPU is found, setting device to CPU.";
    }
    device = DeviceOrd::CPU();
  } else if (fail_on_invalid) {
    CHECK(device.IsCPU() || device.ordinal < n_visible)
        << "Only " << n_visible << " GPUs are visible, ordinal " << device.ordinal
        << " is invalid.";
  } else if (device.IsCUDA() && device.ordinal >= n_visible) {
    device.ordinal = device.ordinal % n_visible;
    LOG(WARNING) << "Only " << n_visible << " GPUs are visible, setting device ordinal to "
                 << device.ordinal;
  }

  if (device.IsCUDA()) {
    curt::SetDevice(device.ordinal);
  }
  return device;
}
#endif  //  !defined(XGBOOST_USE_CUDA)

[[nodiscard]] std::optional<std::int32_t> ParseInt(StringView ordinal) {
  // Some basic checks to ensure valid `gpu_id` and device ordinal instead of directly parsing and
  // letting go of unknown characters.
  if (ordinal.empty()) {
    return std::nullopt;
  }

  std::size_t offset{0};
  if (ordinal[0] == '-') {
    offset = 1;
  }
  if (ordinal.size() <= offset) {
    return std::nullopt;
  }

  bool valid = std::all_of(ordinal.cbegin() + offset, ordinal.cend(),
                           [](auto c) { return std::isdigit(static_cast<unsigned char>(c)); });
  if (!valid) {
    return std::nullopt;
  }

  std::int32_t parsed_id{DeviceOrd::CPUOrdinal()};
  auto res = std::from_chars(ordinal.c_str(), ordinal.c_str() + ordinal.size(), parsed_id);
  if (res.ec != std::errc()) {
    return std::nullopt;
  }

  return parsed_id;
}

[[nodiscard]] DeviceOrd MakeDeviceOrd(std::string const& input, bool fail_on_invalid_gpu_id) {
  // Fast path for the most common case. `device=cpu` is re-parsed on hot paths
  // (e.g. DMatrixProxy::SetArray during inplace prediction), and the general
  // path below costs two regex constructions plus string copies per call.
  if (input == DeviceSym::CPU()) {
    return DeviceOrd::CPU();
  }
  StringView msg{R"(Invalid argument for `device`. Expected to be one of the following:
- cpu
- cuda
- cuda:<device ordinal>  # e.g. cuda:0
- npu
- npu:<device ordinal>   # e.g. npu:0
- gpu
- gpu:<device ordinal>   # e.g. gpu:0
)"};
  auto fatal = [&] {
    LOG(FATAL) << msg << "Got: `" << input << "`.";
  };

#if defined(__MINGW32__) || defined(__EMSCRIPTEN__)
  // mingw hangs on regex using rtools 430. Use the same regex-free grammar for
  // Emscripten, including alias normalization, instead of platform prefix checks.
  std::string input_string{input};
  auto valid_ordinal = [](StringView value) {
    if (value == "-1") {
      return true;
    }
    return !value.empty() &&
           std::all_of(value.cbegin(), value.cend(), [](auto c) {
             return std::isdigit(static_cast<unsigned char>(c));
           });
  };
  auto valid = [&] {
    if (input_string == "cpu" || input_string == "cuda" || input_string == "gpu" ||
        input_string == "npu" || input_string == "sycl" || input_string == "sycl:cpu" ||
        input_string == "sycl:gpu") {
      return true;
    }
    auto first = input_string.find(':');
    if (first == std::string::npos || first == 0 || first + 1 == input_string.size()) {
      return false;
    }
    auto kind = input_string.substr(0, first);
    auto ordinal = input_string.substr(first + 1);
    if (kind == "cuda" || kind == "gpu" || kind == "npu") {
      return valid_ordinal(ordinal) && ordinal != "-1";
    }
    if (kind == "sycl") {
      if (valid_ordinal(ordinal)) {
        return true;
      }
      auto second = ordinal.find(':');
      return second != std::string::npos &&
             (ordinal.substr(0, second) == "cpu" || ordinal.substr(0, second) == "gpu") &&
             valid_ordinal(ordinal.substr(second + 1));
    }
    return false;
  }();
  CHECK(valid) << msg;
#else
  thread_local static std::regex pattern{
      "gpu(:[0-9]+)?|cuda(:[0-9]+)?|npu(:[0-9]+)?|cpu|sycl(:cpu|:gpu)?(:-1|:[0-9]+)?"};
  if (!std::regex_match(input, pattern)) {
    fatal();
  }
#endif  // defined(__MINGW32__) || defined(__EMSCRIPTEN__)

  // handle alias
#if defined(__MINGW32__) || defined(__EMSCRIPTEN__)
  bool is_sycl = input == "sycl" || input == "sycl:cpu" || input == "sycl:gpu" ||
                 input.compare(0, 5, "sycl:") == 0;
#else
  thread_local static std::regex sycl_pattern{"sycl(:cpu|:gpu)?(:-1|:[0-9]+)?"};
  bool is_sycl = std::regex_match(input, sycl_pattern);
#endif  // defined(__MINGW32__) || defined(__EMSCRIPTEN__)

  std::string s_device = input;
  if (!is_sycl) {
#if defined(__MINGW32__) || defined(__EMSCRIPTEN__)
    if (s_device == "gpu") {
      s_device = DeviceSym::CUDA();
    } else if (s_device.compare(0, 4, "gpu:") == 0) {
      s_device.replace(0, 3, DeviceSym::CUDA());
    }
#else
    thread_local static std::regex gpu_pattern{"gpu"};
    s_device = std::regex_replace(s_device, gpu_pattern, DeviceSym::CUDA());
#endif  // defined(__MINGW32__) || defined(__EMSCRIPTEN__)
  }

  auto split_it = std::find(s_device.cbegin(), s_device.cend(), ':');

  // For these cases we need to move iterator to the end, not to look for a ordinal.
  if ((s_device == "sycl:cpu") || (s_device == "sycl:gpu")) {
    split_it = s_device.cend();
  }

  // For s_device like "sycl:gpu:1"
  if (split_it != s_device.cend()) {
    auto second_split_it = std::find(split_it + 1, s_device.cend(), ':');
    if (second_split_it != s_device.cend()) {
      split_it = second_split_it;
    }
  }

  DeviceOrd device;
  device.ordinal = DeviceOrd::InvalidOrdinal();  // mark it invalid for check.
  if (split_it == s_device.cend()) {
    // no ordinal.
    if (s_device == DeviceSym::CPU()) {
      device = DeviceOrd::CPU();
    } else if (s_device == DeviceSym::CUDA()) {
      device = DeviceOrd::CUDA(0);  // use 0 as default;
    } else if (s_device == DeviceSym::NPU()) {
      device = DeviceOrd::NPU(0);
    } else if (s_device == DeviceSym::SyclDefault()) {
      device = DeviceOrd::SyclDefault();
    } else if (s_device == DeviceSym::SyclCPU()) {
      device = DeviceOrd::SyclCPU();
    } else if (s_device == DeviceSym::SyclGPU()) {
      device = DeviceOrd::SyclGPU();
    } else {
      fatal();
    }
  } else {
    // CUDA, NPU, or SYCL with an explicit ordinal.
    // +1 for colon
    std::size_t offset = std::distance(s_device.cbegin(), split_it) + 1;
    // substr
    StringView s_ordinal = {s_device.data() + offset, s_device.size() - offset};
    StringView s_type = {s_device.data(), offset - 1};
    if (s_ordinal.empty()) {
      fatal();
    }
    auto opt_id = ParseInt(s_ordinal);
    if (!opt_id.has_value()) {
      fatal();
    }
    CHECK_LE(opt_id.value(), std::numeric_limits<bst_d_ordinal_t>::max())
        << "Ordinal value too large.";
    if (s_type == DeviceSym::NPU()) {
      device = DeviceOrd::NPU(opt_id.value());
    } else if (s_type == DeviceSym::SyclDefault()) {
      device = DeviceOrd::SyclDefault(opt_id.value());
    } else if (s_type == DeviceSym::SyclCPU()) {
      device = DeviceOrd::SyclCPU(opt_id.value());
    } else if (s_type == DeviceSym::SyclGPU()) {
      device = DeviceOrd::SyclGPU(opt_id.value());
    } else if (s_type == DeviceSym::CUDA()) {
      device = DeviceOrd::CUDA(opt_id.value());
    } else {
      fatal();
    }
  }

  if (device.ordinal < DeviceOrd::CPUOrdinal() ||
      ((device.IsCUDA() || device.IsNPU()) && device.ordinal < 0)) {
    fatal();
  }
  if (device.IsNPU()) {
#if !defined(XGBOOST_USE_NPU)
    LOG(FATAL) << "XGBoost is not compiled with NPU support (XGBOOST_USE_NPU). "
                  "Requested device: " << device.Name() << ". CPU fallback is disabled.";
#else
    // Native learner, grow_npu_hist and npu_predictor own their device state.
    // Generic device dispatch remains fail-closed for unadapted call sites.
    CHECK_EQ(device.ordinal, 0) << "The initial native NPU backend supports device npu:0 only.";
#endif  // !defined(XGBOOST_USE_NPU)
  }
  if (device.IsCUDA()) {
    device = CUDAOrdinal(device, fail_on_invalid_gpu_id);
    if (!device.IsCUDA()) {
      // We allow loading a GPU-based pickle on a CPU-only machine.
      LOG(WARNING) << "Device is changed from GPU to CPU as we couldn't find any available GPU on "
                      "the system.";
    }
  }
  return device;
}
}  // namespace

std::ostream& operator<<(std::ostream& os, DeviceOrd ord) {
  os << ord.Name();
  return os;
}

void Context::Init(Args const& kwargs) {
  auto unknown = this->UpdateAllowUnknown(kwargs);
  if (!unknown.empty()) {
    std::stringstream ss;
    std::size_t i = 0;
    ss << "[Internal Error] Unknown parameters passed to the Context {";
    for (auto const& [k, _] : unknown) {
      ss << '"' << k << '"';
      if (++i != unknown.size()) {
        ss << ", ";
      }
    }
    ss << "}\n";
    LOG(FATAL) << ss.str();
  }
}

void Context::SetDeviceOrdinal(Args const& kwargs) {
  auto gpu_id_it = std::find_if(kwargs.cbegin(), kwargs.cend(),
                                [](auto const& p) { return p.first == "gpu_id"; });
  auto has_gpu_id = gpu_id_it != kwargs.cend();
  if (has_gpu_id) {
    LOG(FATAL) << "`gpu_id` has been removed since 3.1. Use `device` instead.";
  }

  auto device_it = std::find_if(kwargs.cbegin(), kwargs.cend(),
                                [](auto const& p) { return p.first == kDevice; });
  auto has_device = device_it != kwargs.cend();
  auto new_d = MakeDeviceOrd(this->device, this->fail_on_invalid_gpu_id);

  if (!has_device) {
    CHECK_EQ(new_d.ordinal, this->device_.ordinal);  // unchanged
  }
  this->SetDevice(new_d);

  if (this->IsCPU()) {
    CHECK_EQ(this->device_.ordinal, DeviceOrd::CPUOrdinal());
  } else if (this->IsCUDA()) {
    CHECK_GT(this->device_.ordinal, DeviceOrd::CPUOrdinal());
  } else if (this->IsNPU()) {
    CHECK_GT(this->device_.ordinal, DeviceOrd::CPUOrdinal());
  }
}

std::int32_t Context::Threads() const {
  auto n_threads = common::OmpGetNumThreads(nthread);
  if (cfs_cpu_count_ > 0) {
    n_threads = std::min(n_threads, cfs_cpu_count_);
  }
  return n_threads;
}

DeviceOrd Context::DeviceFP64() const {
#if defined(XGBOOST_USE_SYCL)
  return sycl::DeviceFP64(device_);
#else
  return device_;
#endif  // defined(XGBOOST_USE_SYCL)
}

[[nodiscard]] Json Context::ToJson() const {
  auto obj = Json{::xgboost::ToJson(*this)};
  common::SaveRng(&obj, this->rng_);
  return obj;
}

void Context::FromJson(Json const& in) {
  ::xgboost::FromJson(in, this);
  common::LoadRng(in, &this->rng_);
}

#if !defined(XGBOOST_USE_CUDA)
CUDAContext const* Context::CUDACtx() const {
  common::AssertGPUSupport();
  return nullptr;
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
