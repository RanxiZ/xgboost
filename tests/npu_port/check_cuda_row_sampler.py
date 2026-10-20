#!/usr/bin/env python3
"""Build the production row-sampler ABI test; NPU execution requires both opt-ins.

The caller must hold its shared external device flock for the entire --run-npu
invocation. This checker deliberately does not acquire a second/nested lock.
The default only compiles and runs processes protected by the no-device guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPECTED_CASES = 573
CCCL_INCLUDE = Path('/workspace/npu-evidence/cccl-3.3.3/libcudacxx/include')
CANN_ROOT = Path('/usr/local/Ascend/cann-9.1.0')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='fresh, nonexistent directory for this invocation')
    parser.add_argument('--expected-source-root', type=Path, required=True)
    parser.add_argument('--run-npu', action='store_true')
    parser.add_argument('--confirm-device-idle', action='store_true')
    args = parser.parse_args()
    # Reject incomplete opt-in before creating artifacts or loading any library.
    if args.run_npu != args.confirm_device_idle:
        parser.error('device execution requires both --run-npu and --confirm-device-idle')
    root = Path(__file__).resolve().parents[2]
    if root != args.expected_source_root.resolve():
        parser.error('unexpected source root')
    out = args.output_dir.absolute()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'output directory must be fresh/nonexistent: {out}')
    out = out.resolve()
    report: dict = {
        'status': 'failed', 'scope': 'production_XgbNpuNativeSampleRows_ABI_bitwise',
        'source_root': str(root), 'oracle': 'actual CCCL 3.3.3 cuda::std::philox4x64 headers',
        'cccl_include': str(CCCL_INCLUDE), 'npu_execution': 'not_run',
        'cuda_execution': 'not_run', 'profiling': 'not_run', 'commands': [],
        'device_lock': 'external caller must hold shared flock; checker does not acquire it',
        'limitations': ['device ordinal 0 only', 'standalone ABI regression, not training acceptance',
                        'host guard success does not validate the compiled device implementation'],
    }
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               LD_LIBRARY_PATH='')

    def run(command: list[str], name: str, e: dict | None = None, expected: int = 0) -> str:
        report['commands'].append({'argv': command, 'log': name, 'expected_exit': expected})
        with (out / name).open('w') as log:
            proc = subprocess.run(command, cwd=root, env=e or env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=300)
        require(proc.returncode == expected,
                f'exit {proc.returncode}, expected {expected}; see {out / name}')
        return (out / name).read_text()

    try:
        cann = CANN_ROOT.resolve(strict=True)
        cccl = CCCL_INCLUDE.resolve(strict=True)
        library = (root / 'lib-npu/libxgboost_npu_kernels.so').resolve(strict=True)
        source = root / 'tests/npu_port'
        native = {**env, 'LD_LIBRARY_PATH': ':'.join(str(p) for p in [
            library.parent, cann / 'lib64', cann / 'lib64/plugin/opskernel',
            cann / 'lib64/plugin/nnengine', cann / 'opp/built-in/op_impl/ai_core/tbe/op_tiling',
            Path('/usr/local/Ascend/driver/lib64'), Path('/usr/local/Ascend/driver/lib64/common'),
            Path('/usr/local/Ascend/driver/lib64/driver')])}
        native.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann),
                      ASCEND_TOOLKIT_LATEST_HOME=str(cann), ASCEND_OPP_PATH=str(cann / 'opp'),
                      ASCEND_AICPU_PATH=str(cann))
        report['device_environment'] = {
            k: v for k, v in native.items() if k.startswith('ASCEND_') or k == 'LD_LIBRARY_PATH'}
        flags = [os.environ.get('CXX', 'g++'), '-std=c++17', '-O2', '-Wall', '-Wextra',
                 '-Werror', '-pedantic', '-fno-fast-math', '-ffp-contract=off',
                 '-isystem', str(cccl), '-isystem', str(cann / 'include')]
        binary = out / 'cuda_row_sampler_device_test'
        run([*flags, str(source / 'cuda_row_sampler_device_test.cc'), str(library),
             f'-Wl,-rpath,{library.parent}', f'-L{cann / "lib64"}',
             f'-Wl,-rpath,{cann / "lib64"}', '-lascendcl', '-ldl', '-o', str(binary)],
            'device-compile.log', native)
        linked = run(['ldd', str(binary)], 'device-libraries.txt', native)
        require('not found' not in linked and 'libxgboost.so' not in linked and
                str(library) in linked, 'invalid production kernel linkage')
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith('/usr/local/Ascend/'):
                    path = Path(word).resolve()
                    require(path.is_relative_to(cann) or
                            path.is_relative_to('/usr/local/Ascend/driver'),
                            f'mixed CANN provenance: {path}')
        guard = out / 'no_device_guard.so'
        run([*flags, '-fPIC', '-shared', str(source / 'runtime_no_device_guard.cc'),
             '-o', str(guard)], 'guard-compile.log', native)
        guarded = {**native, 'LD_PRELOAD': str(guard)}
        text = run([str(binary), '--host-api-only'], 'host-api-guard.log', guarded)
        require('CUDA_ROW_SAMPLER_HOST_GUARD_PASS device_execution=not_run' in text and
                'FORBIDDEN_ACL_CALL' not in text, 'host guard path touched ACL')
        text = run([str(binary), '--run-npu', '--confirm-device-idle'],
                   'guard-positive.log', guarded, 86)
        require('FORBIDDEN_ACL_CALL aclInit' in text, 'no-device guard positive control failed')
        for i, argv in enumerate([[], ['--run-npu'], ['--confirm-device-idle'],
                                  ['--run-npu', '--wrong-ack'],
                                  ['--run-npu', '--confirm-device-idle', 'extra'],
                                  ['--device-test', '0', '--confirm-device-idle']]):
            text = run([str(binary), *argv], f'argument-guard-{i}.log', guarded, 2)
            require('FORBIDDEN_ACL_CALL' not in text, 'invalid arguments touched ACL')
        report['device_call_guards'] = 'passed'
        if args.run_npu:
            # Only the explicit opt-in branch even invokes npu-smi. The main
            # caller owns the device flock across these checks and the workload.
            def idle(name: str) -> None:
                text = run(['npu-smi', 'info'], name, native)
                require('No running processes found in NPU 0' in text,
                        'NPU occupied or idle state unavailable; device execution deferred')

            idle('device-before.txt')
            time.sleep(1)
            idle('device-recheck.txt')
            text = run([str(binary), '--run-npu', '--confirm-device-idle'],
                       'device-tests.log', native)
            require(f'CUDA_ROW_SAMPLER_NPU_PASS checks={EXPECTED_CASES} ' in text,
                    'missing device completion sentinel')
            require('cleanup failure' not in text and 'cleanup status=' not in text,
                    'device resource cleanup failed')
            records = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
            require(len(records) == EXPECTED_CASES and
                    len({r['test'] for r in records}) == EXPECTED_CASES and
                    all(r['status'] == 'passed' and r['repeats'] == 2 for r in records),
                    'missing, duplicate or failed device cases')
            require(any(r['nrows'] > 2048 for r in records), 'grid-stride coverage missing')
            report['device_tests'] = records
            idle('device-after.txt')
            report['npu_execution'] = 'passed'
        paths = [Path(__file__), source / 'cuda_row_sampler_device_test.cc',
                 source / 'runtime_no_device_guard.cc',
                 root / 'src/common/npu/native_training_abi.h',
                 root / 'src/common/npu/native_training.asc',
                 root / 'src/common/npu/cuda_row_sampler.h',
                 root / 'src/common/npu/acl_api.h', root / 'src/common/npu/runtime.h',
                 cccl / 'cuda/std/random', cccl / 'cuda/std/__cccl/version.h',
                 cccl / 'cuda/std/__random/philox_engine.h',
                 cccl / 'cuda/std/__random/generate_canonical.h',
                 cccl / 'cuda/std/__random/uniform_real_distribution.h', binary, guard, library]
        report['artifact_sha256'] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['status'] = 'passed'
        print('CUDA_ROW_SAMPLER_VALIDATION_PASS guards=passed '
              f'npu_execution={report["npu_execution"]}')
        return 0
    except Exception as error:
        report['error'] = str(error)
        print(f'CUDA_ROW_SAMPLER_VALIDATION_FAIL: {error}', file=sys.stderr)
        return 1
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())