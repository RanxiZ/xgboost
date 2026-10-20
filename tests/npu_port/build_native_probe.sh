#!/usr/bin/env bash
# Build and host-check only. This script never initializes an NPU device.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cann=${CANN_ROOT:-/usr/local/Ascend/cann-9.1.0}
out=${1:-/workspace/xgb-npu-runs/stage2-native}
mkdir -p "$out"
cann=$(realpath "$cann")
# Avoid inheriting mixed toolkit libraries. The driver is intentionally shared.
export LD_LIBRARY_PATH="$cann/lib64:$cann/lib64/plugin/opskernel:$cann/lib64/plugin/nnengine:$cann/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$root/lib"
unset LD_PRELOAD
export ASCEND_HOME_PATH="$cann" ASCEND_TOOLKIT_HOME="$cann" ASCEND_TOOLKIT_LATEST_HOME="$cann"
export ASCEND_OPP_PATH="$cann/opp" ASCEND_AICPU_PATH="$cann"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
bash -n "$root/tests/npu_port/build_native_probe.sh"
{
  printf 'source_root=%s\ncann_root=%s\n' "$root" "$cann"
  git -C "$root" branch --show-current
  git -C "$root" rev-parse HEAD
  "${CXX:-g++}" --version
  sha256sum "$root/tests/npu_port/native_probe.cc" "$root/tests/npu_port/build_native_probe.sh"
  printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
} > "$out/build-manifest.txt"
"${CXX:-g++}" -std=c++17 -O2 -fopenmp -Wall -Wextra -Werror \
  -I"$cann/include" -I"$root/include" -I"$root/dmlc-core/include" \
  "$root/tests/npu_port/native_probe.cc" -L"$cann/lib64" -L"$root/lib" \
  -Wl,-rpath,"$cann/lib64" -Wl,-rpath,"$root/lib" \
  -lopapi -lnnopbase -lascendcl -lxgboost -ldl -o "$out/native_probe" > "$out/compile.log" 2>&1
ldd "$out/native_probe" > "$out/linked-libraries.txt"
if grep -q 'not found' "$out/linked-libraries.txt"; then
  echo 'Unresolved dynamic library; see linked-libraries.txt' >&2
  exit 1
fi
# Verify every resolved Ascend toolkit library, allowing only the shared driver.
while read -r library; do
  case "$library" in
    "$cann"/*|/usr/local/Ascend/driver/*) ;;
    *) echo "Mixed Ascend library: $library" >&2; exit 1 ;;
  esac
done < <(awk '$3 ~ /^\/usr\/local\/Ascend\// {print $3}' "$out/linked-libraries.txt")
"$out/native_probe" --host-reference-only > "$out/host-reference.log" 2>&1
# Missing device acknowledgement must fail before aclInit, not run device work.
set +e
"$out/native_probe" --device-test 0 > "$out/device-guard.log" 2>&1
guard_status=$?
set -e
if [[ $guard_status -ne 2 ]]; then
  echo "Device opt-in guard failed: status=$guard_status" >&2
  exit 1
fi
printf 'device_opt_in_guard=passed\ndevice_tests=not_run\n' > "$out/validation.txt"
sha256sum "$out/native_probe" >> "$out/build-manifest.txt"
echo 'STAGE2A_BUILD_AND_HOST_REFERENCE_PASS (device tests not run)'