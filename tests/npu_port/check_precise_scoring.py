#!/usr/bin/env python3
"""Validate standalone Ascend C strict split scores, not candidate selection/training."""
from __future__ import annotations
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from check_integer_histogram import require, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-source-root', type=Path, required=True)
    parser.add_argument('--device-test', type=int)
    parser.add_argument('--confirm-device-idle', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    require((args.device_test is None) == (not args.confirm_device_idle), 'device needs both opt-in arguments')
    require(args.device_test in (None, 0), 'single ordinal 0 only')
    require(not args.profile or args.device_test is not None, 'profile needs device opt-in')
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), 'unexpected source checkout')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    cann = Path('/usr/local/Ascend/cann-9.1.0').resolve(strict=True); build = out / 'kernel-build'
    env = os.environ.copy(); env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', LD_LIBRARY_PATH='')
    native = {**env, 'LD_LIBRARY_PATH': ':'.join(str(p) for p in [build, cann / 'lib64',
        cann / 'lib64/plugin/opskernel', cann / 'lib64/plugin/nnengine',
        cann / 'opp/built-in/op_impl/ai_core/tbe/op_tiling', Path('/usr/local/Ascend/driver/lib64'),
        Path('/usr/local/Ascend/driver/lib64/common'), Path('/usr/local/Ascend/driver/lib64/driver')])}
    native.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann), ASCEND_TOOLKIT_LATEST_HOME=str(cann),
                  ASCEND_OPP_PATH=str(cann / 'opp'), ASCEND_AICPU_PATH=str(cann),
                  PATH=f'{cann}/tools/bisheng_compiler/bin:' + env['PATH'])
    report = {'status': 'failed', 'scope': 'standalone_strict_scalar_weights_and_child_gains',
              'npu_execution': 'not_run', 'cuda_execution': 'not_run', 'candidate_selection': 'not_implemented',
              'training_backend': 'not_implemented', 'commands': [],
              'source_root': str(root), 'cann_root': str(cann),
              'limitations': ['single device scalar numerical features; no constraints',
                'binary64 uses integer RN-even emulation on vector cores, no native FP64 claim',
                'custom kernels use mixed AIC/AIV dispatch; not pure AIV-only tasks',
                'rounded nonzero binary64 subnormals and nonfinite intermediates rejected',
                'float32 subnormals preserved; negative-zero parameters rejected',
                'scale exponents supplied as scalar metadata, source binding pending',
                'no missing-direction/candidate reduction, parent subtraction or performance acceptance',
                'official shared scalar formula with contraction disabled; actual CUDA FMA/FTZ unverified']}

    def run(argv: list[str], name: str, e: dict | None = None, expected: int = 0) -> str:
        report['commands'].append({'argv': argv, 'log': name, 'expected_exit': expected})
        with (out / name).open('w') as log:
            result = subprocess.run(argv, cwd=root, env=e or env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        require(result.returncode == expected, f'exit {result.returncode}, expected {expected}; see {out / name}')
        return (out / name).read_text()

    def provenance(path: str) -> None:
        p = Path(path).resolve()
        require(p.is_relative_to(cann) or p.is_relative_to('/usr/local/Ascend/driver'), f'mixed CANN library: {p}')

    def idle(name: str) -> None:
        require('No running processes found in NPU 0' in run(['npu-smi', 'info'], name, native), 'NPU occupied; test deferred')

    def device_records(text: str) -> list[dict]:
        result = records(text)
        require('PRECISE_SCORING_NPU_PASS checks=35' in text and len(result) == 35, 'incomplete device tests')
        require('cleanup failure' not in text and 'cleanup status=' not in text, 'cleanup diagnostic')
        require({r['test'].removeprefix('npu_') for r in result if not r['test'].startswith('npu_integration_')} >=
                {r['test'].removeprefix('host_') for r in report['host_tests']}, 'device fixture coverage mismatch')
        for line in text.splitlines():
            if line.startswith('mapping '): provenance(line.split()[-1])
        return result

    try:
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip() == 'xgb-npu-port', 'wrong branch')
        src = root / 'tests/npu_port'; cxx = os.environ.get('CXX', 'g++')
        flags = [cxx, '-std=c++17', '-fopenmp', '-Wall', '-Wextra', '-Werror', '-pedantic',
                 '-fno-fast-math', '-ffp-contract=off', f'-I{root / "include"}', f'-I{root / "dmlc-core/include"}']
        host = out / 'precise_scoring_host_test'
        run([*flags, '-O2', str(src / 'precise_scoring_host_test.cc'), '-o', str(host)], 'host-compile.log')
        linked = run(['ldd', str(host)], 'host-libraries.txt')
        require(not any(x in linked.lower() for x in ('ascend', 'libcuda', 'libopapi', 'not found')), 'host linked device runtime')
        text = run([str(host)], 'host-tests.log'); report['host_tests'] = records(text)
        counts = re.search(r'PRECISE_SCORING_HOST_PASS groups=(\d+) arithmetic=(\d+) scores=(\d+)', text)
        require(counts is not None and len(report['host_tests']) == 13, 'host completion missing')
        report['fixture_counts'] = dict(zip(('groups', 'arithmetic', 'scores'), map(int, counts.groups())))
        san = out / 'precise_scoring_host_sanitized'
        run([*flags, '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
             str(src / 'precise_scoring_host_test.cc'), '-o', str(san)], 'sanitizer-compile.log')
        text = run([str(san)], 'sanitizer-tests.log', {**env, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1'})
        require(records(text) == report['host_tests'], 'sanitizer results differ'); report['host_sanitizers'] = 'passed'
        run(['cmake', '-S', str(src / 'precision_kernel'), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_ASC_ARCHITECTURES=dav-3510'], 'kernel-configure.log', native)
        run(['cmake', '--build', str(build), '-j2'], 'kernel-build.log', native)
        library = build / 'libxgb_npu_precise.so'
        require('.aicore_binary' in run(['readelf', '-S', str(library)], 'kernel-sections.txt'), 'no embedded device binary')
        binary = out / 'precise_scoring_test'
        run([*flags, '-O2', '-isystem', str(cann / 'include'), str(src / 'precise_scoring_test.cc'),
             f'-L{build}', f'-Wl,-rpath,{build}', '-lxgb_npu_precise', f'-L{cann / "lib64"}',
             f'-Wl,-rpath,{cann / "lib64"}', '-lopapi', '-lnnopbase', '-lascendcl', '-ldl', '-o', str(binary)], 'device-compile.log', native)
        linked = run(['ldd', str(binary)], 'device-libraries.txt', native)
        require('not found' not in linked and 'libxgboost' not in linked, 'invalid standalone linkage')
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith('/usr/local/Ascend/'): provenance(word)
        guard = out / 'no_device_guard.so'
        run([*flags, '-O2', '-fPIC', '-shared', '-isystem', str(cann / 'include'),
             str(src / 'runtime_no_device_guard.cc'), '-o', str(guard)], 'guard-compile.log')
        guarded = {**native, 'LD_PRELOAD': str(guard)}
        text = run([str(binary), '--host-api-only'], 'host-api-guard.log', guarded)
        require('PRECISE_SCORING_API_HOST_PASS' in text and 'FORBIDDEN_ACL_CALL' not in text, 'host API touched ACL')
        text = run([str(binary), '--device-test', '0', '--confirm-device-idle'], 'guard-positive.log', guarded, 86)
        require('FORBIDDEN_ACL_CALL aclInit' in text, 'guard positive control failed')
        for i, argv in enumerate([[], ['--device-test', '0'], ['--device-test', '0', '--wrong-ack'], ['--device-test', '1', '--confirm-device-idle']]):
            text = run([str(binary), *argv], f'argument-guard-{i}.log', guarded, 2)
            require('FORBIDDEN_ACL_CALL' not in text, 'invalid arguments touched ACL')
        report['device_call_guards'] = 'passed'
        if args.device_test is not None:
            with (out.parent / 'npu0-test.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); idle('device-before.txt'); time.sleep(1); idle('device-recheck.txt')
                command = [str(binary), '--device-test', '0', '--confirm-device-idle']
                report['device_tests'] = device_records(run(command, 'device-tests.log', native))
                idle('device-after.txt'); report['npu_execution'] = 'passed'
                if args.profile:
                    profile = out / 'profile'; require(not profile.exists(), 'profile exists; use a fresh output directory')
                    idle('profile-before.txt')
                    text = run([str(cann / 'tools/profiler/bin/msprof'), f'--output={profile}', '--runtime-api=on', '--task-time=on',
                                '--ai-core=on', '--aicpu=on', '--sys-lp=off', *command], 'profile-run.log', native)
                    require(device_records(text) == report['device_tests'], 'profile replay differs'); idle('profile-after.txt')
                    summaries = list(profile.rglob('op_summary_*.csv')); require(len(summaries) == 1, 'operator summary missing')
                    with summaries[0].open() as stream: rows = list(csv.DictReader(stream))
                    require(rows and all(r['Device_id'] == '0' for r in rows), 'missing NPU task evidence')
                    kernel_names = {'PreciseArithmeticKernel', 'PreciseScoreKernel', 'ScoreColumnsKernel'}
                    custom = [r for r in rows if r['OP Type'] in kernel_names]
                    require({r['OP Type'] for r in custom} == kernel_names, 'custom precision kernel task evidence missing')
                    histogram = [r for r in rows if r['OP Type'] == 'IntegerHistogramBuildKernel']
                    require(histogram, 'missing fused histogram kernel evidence')
                    # AscendC VF launches carry an AIC dispatch component on this
                    # compiler target. MIX_AIC is not AI_CPU: admit it only for
                    # these exact custom kernels, with positive vector execution
                    # and no cube multiply-accumulate activity in the PMU record.
                    for row in rows:
                        if row['Task Type'] in ('AI_VECTOR_CORE', 'MIX_AIV'):
                            continue
                        require(row['Task Type'] == 'MIX_AIC' and
                                row['OP Type'] in kernel_names | {'IntegerHistogramBuildKernel'},
                                f"unexpected compute task: {row['OP Type']} {row['Task Type']}")
                        require(float(row['aiv_vec_time(us)']) > 0 and float(row['aic_mac_time(us)']) == 0,
                                'mixed precision kernel lacks vector-only arithmetic evidence')
                    report['execution_evidence'] = {'operator_summary': str(summaries[0]), 'task_count': len(rows),
                        'custom_precision_task_count': len(custom), 'fused_histogram_task_count': len(histogram),
                        'custom_precision_task_types': dict(collections.Counter(r['Task Type'] for r in custom)),
                        'custom_mixed_dispatch': 'AIC dispatch plus AIV integer arithmetic; no pure-AIV claim',
                        'task_types': dict(collections.Counter(r['Task Type'] for r in rows)),
                        'performance_acceptance': 'not_run'}
        paths = list((root / 'src/common/npu').glob('*.h')) + list((root / 'src/common/npu').glob('*.asc'))
        paths += [src / name for name in ('precise_arithmetic_abi.h', 'precise_arithmetic.asc', 'precise_test_kernels.asc', 'precise_scoring_cases.h',
                  'precise_scoring_host_test.cc', 'precise_scoring_test.cc', 'split_statistics_cases.h', 'histogram_cases.h',
                  'quantizer_cases.h', 'check_integer_histogram.py', 'runtime_no_device_guard.cc', 'precision_kernel/CMakeLists.txt')]
        paths += [Path(__file__), host, binary, library]
        report['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['status'] = 'passed'
        print(f"PRECISE_SCORING_VALIDATION_PASS host_groups=13 npu_execution={report['npu_execution']} selection=not_implemented")
        return 0
    except Exception as error:
        report['error'] = str(error); print(f'PRECISE_SCORING_VALIDATION_FAIL: {error}', file=sys.stderr); return 1
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
