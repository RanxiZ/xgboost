#!/usr/bin/env python3
"""Check NPU integer candidate statistics, not split scoring or tree training."""
from __future__ import annotations
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from check_integer_histogram import records, require


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
    cann = Path('/usr/local/Ascend/cann-9.1.0').resolve(strict=True)
    build = out / 'kernel-build'
    report = {'status': 'failed', 'scope': 'integer_numerical_candidate_statistics_only',
              'source_root': str(root), 'cann_root': str(cann), 'commands': [],
              'npu_execution': 'not_run', 'cuda_execution': 'not_run',
              'split_scoring': 'not_implemented', 'training_backend': 'not_implemented',
              'limitations': ['no gains, weights, floating-point dequantization or candidate selection',
                              'numerical features, scalar target, single device',
                              'requires validated immutable IntegerHistogramBuilder result',
                              'synchronous correctness-first implementation; no speed claim']}
    env = os.environ.copy(); env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', LD_LIBRARY_PATH='')
    native_env = {**env, 'LD_LIBRARY_PATH': ':'.join(str(p) for p in [build,
        cann / 'lib64', cann / 'lib64/plugin/opskernel', cann / 'lib64/plugin/nnengine',
        cann / 'opp/built-in/op_impl/ai_core/tbe/op_tiling', Path('/usr/local/Ascend/driver/lib64'),
        Path('/usr/local/Ascend/driver/lib64/common'), Path('/usr/local/Ascend/driver/lib64/driver')])}
    native_env.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann),
                      ASCEND_TOOLKIT_LATEST_HOME=str(cann), ASCEND_OPP_PATH=str(cann / 'opp'), ASCEND_AICPU_PATH=str(cann),
                      PATH=f'{cann}/tools/bisheng_compiler/bin:' + env['PATH'])
    report['device_environment'] = {k: v for k, v in native_env.items() if k.startswith('ASCEND_') or k == 'LD_LIBRARY_PATH'}

    def run(command: list[str], name: str, e: dict | None = None, expected: int = 0) -> str:
        report['commands'].append({'argv': command, 'log': name, 'expected_exit': expected})
        with (out / name).open('w') as log:
            result = subprocess.run(command, cwd=root, env=e or env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        require(result.returncode == expected, f'exit {result.returncode}, expected {expected}; see {out / name}')
        return (out / name).read_text()

    def provenance(path: str) -> None:
        p = Path(path).resolve()
        require(p.is_relative_to(cann) or p.is_relative_to('/usr/local/Ascend/driver'), f'mixed CANN library: {p}')

    def idle(name: str) -> None:
        text = run(['npu-smi', 'info'], name, native_env)
        require('No running processes found in NPU 0' in text, 'NPU occupied; device test deferred')

    def device_records(text: str) -> list[dict]:
        require('SPLIT_STATISTICS_NPU_PASS checks=38' in text, 'missing NPU completion sentinel')
        require('cleanup failure' not in text and 'cleanup status=' not in text, 'resource cleanup error')
        result = records(text); require(len(result) == 38, 'wrong device test count')
        expected = {r['test'].removeprefix('host_') for r in report['host_tests'][:15]}
        for prefix in ('npu_direct_', 'npu_subtracted_'):
            require({r['test'].removeprefix(prefix) for r in result if r['test'].startswith(prefix)} == expected,
                    'host/device case mismatch')
        for line in text.splitlines():
            if line.startswith('mapping '): provenance(line.split()[-1])
            if line.startswith('library '):
                symbol, path = line.split()[1:3]
                if symbol == 'XgbNpuSplitStatistics':
                    require(Path(path).resolve() == library.resolve(), 'wrong statistics kernel library')
                else:
                    provenance(path)
        return result

    try:
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip() ==
                'xgb-npu-port', 'unexpected branch')
        source = root / 'tests/npu_port'; cxx = os.environ.get('CXX', 'g++')
        flags = [cxx, '-std=c++17', '-fopenmp', '-Wall', '-Wextra', '-Werror', '-pedantic',
                 '-fno-fast-math', '-ffp-contract=off', f'-I{root / "include"}', f'-I{root / "dmlc-core/include"}']
        host = out / 'split_statistics_host_test'
        run([*flags, '-O2', str(source / 'split_statistics_host_test.cc'), '-o', str(host)], 'host-compile.log')
        linked = run(['ldd', str(host)], 'host-libraries.txt')
        require(not any(x in linked.lower() for x in ('ascend', 'libopapi', 'libcuda', 'not found')), 'host linked device runtime')
        text = run([str(host)], 'host-tests.log')
        require('SPLIT_STATISTICS_HOST_PASS checks=16' in text, 'host sentinel missing')
        report['host_tests'] = records(text); require(len(report['host_tests']) == 16, 'host test count')
        fixtures = [json.loads(line.removeprefix('CUDA_SPLIT_FIXTURE ')) for line in text.splitlines()
                    if line.startswith('CUDA_SPLIT_FIXTURE ')]
        contract = json.loads((source / 'cuda_contract.json').read_text())['fixed_fixture']
        require(fixtures == [{'prefix': contract['expected_exclusive_prefix'], 'missing': contract['expected_missing']}],
                'candidate fixture differs from pinned CUDA source contract')
        report['cuda_source_fixture'] = 'passed; CUDA runtime not executed'
        sanitized = out / 'split_statistics_host_sanitized'
        run([*flags, '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
             str(source / 'split_statistics_host_test.cc'), '-o', str(sanitized)], 'sanitizer-compile.log')
        text = run([str(sanitized)], 'sanitizer-tests.log', {**env, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                   'UBSAN_OPTIONS': 'halt_on_error=1'})
        require(records(text) == report['host_tests'], 'sanitized tests differ'); report['host_sanitizers'] = 'passed'
        run(['cmake', '-S', str(source / 'precision_kernel'), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_ASC_ARCHITECTURES=dav-3510'], 'kernel-configure.log', native_env)
        run(['cmake', '--build', str(build), '-j2'], 'kernel-build.log', native_env)
        library = build / 'libxgb_npu_precise.so'
        require('.aicore_binary' in run(['readelf', '-S', str(library)], 'kernel-sections.txt'), 'no embedded device binary')
        binary = out / 'split_statistics_test'
        run([*flags, '-O2', '-isystem', str(cann / 'include'), str(source / 'split_statistics_test.cc'),
             f'-L{build}', f'-Wl,-rpath,{build}', '-lxgb_npu_precise',
             f'-L{cann / "lib64"}', f'-Wl,-rpath,{cann / "lib64"}', '-lopapi', '-lnnopbase', '-lascendcl', '-ldl',
             '-o', str(binary)], 'device-compile.log', native_env)
        linked = run(['ldd', str(binary)], 'device-libraries.txt', native_env)
        require('not found' not in linked and 'libxgboost' not in linked, 'invalid device test linkage')
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith('/usr/local/Ascend/'): provenance(word)
        require(str(library.resolve()) in linked, 'standalone statistics kernel library not linked')
        guard = out / 'no_device_guard.so'
        run([*flags, '-O2', '-fPIC', '-shared', '-isystem', str(cann / 'include'),
             str(source / 'runtime_no_device_guard.cc'), '-o', str(guard)], 'guard-compile.log')
        guarded = {**native_env, 'LD_PRELOAD': str(guard)}
        text = run([str(binary), '--host-api-only'], 'host-api-guard.log', guarded)
        require('SPLIT_STATISTICS_API_HOST_PASS' in text and 'FORBIDDEN_ACL_CALL' not in text, 'host API touched ACL')
        text = run([str(binary), '--device-test', '0', '--confirm-device-idle'], 'guard-positive.log', guarded, 86)
        require('FORBIDDEN_ACL_CALL aclInit' in text, 'guard positive control failed')
        for i, argv in enumerate([[], ['--device-test', '0'], ['--device-test', '0', '--wrong-ack'],
                                  ['--device-test', '1', '--confirm-device-idle']]):
            text = run([str(binary), *argv], f'argument-guard-{i}.log', guarded, 2)
            require('FORBIDDEN_ACL_CALL' not in text, 'invalid arguments touched ACL')
        report['device_call_guards'] = 'passed'
        if args.device_test is not None:
            with (out.parent / 'npu0-test.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                idle('device-before.txt'); time.sleep(1); idle('device-recheck.txt')
                command = [str(binary), '--device-test', '0', '--confirm-device-idle']
                report['device_tests'] = device_records(run(command, 'device-tests.log', native_env))
                idle('device-after.txt'); report['npu_execution'] = 'passed'
                if args.profile:
                    profile = out / 'profile'; require(not profile.exists(), 'profiling output exists; use fresh output')
                    idle('profile-before.txt')
                    text = run([str(cann / 'tools/profiler/bin/msprof'), f'--output={profile}', '--runtime-api=on',
                                '--task-time=on', '--ai-core=on', '--aicpu=on', '--sys-lp=off', *command],
                               'profile-run.log', native_env)
                    require(device_records(text) == report['device_tests'], 'profile replay differs'); idle('profile-after.txt')
                    summaries = list(profile.rglob('op_summary_*.csv')); require(len(summaries) == 1, 'operator summary missing')
                    with summaries[0].open() as stream: rows = list(csv.DictReader(stream))
                    require(rows and all(r['Device_id'] == '0' for r in rows), 'missing NPU task evidence')
                    custom = [r for r in rows if r['OP Type'] == 'SplitStatisticsKernel']
                    require(custom, 'missing custom statistics kernel evidence')
                    histogram = [r for r in rows if r['OP Type'] == 'IntegerHistogramBuildKernel']
                    require(histogram, 'missing fused histogram kernel evidence')
                    for row in rows:
                        if row['Task Type'] in ('MIX_AIV', 'AI_VECTOR_CORE'):
                            continue
                        require(row['Task Type'] == 'MIX_AIC' and row['OP Type'] in
                                {'SplitStatisticsKernel', 'IntegerHistogramBuildKernel'},
                                f"unexpected compute task: {row['OP Type']} {row['Task Type']}")
                        require(float(row['aiv_vec_time(us)']) > 0 and float(row['aic_mac_time(us)']) == 0,
                                'mixed statistics/histogram kernel lacks vector-only arithmetic evidence')
                    report['execution_evidence'] = {'operator_summary': str(summaries[0]), 'task_count': len(rows),
                        'custom_statistics_task_count': len(custom), 'fused_histogram_task_count': len(histogram),
                        'custom_statistics_task_types': dict(collections.Counter(r['Task Type'] for r in custom)),
                        'op_counts': dict(collections.Counter(r['OP Type'] for r in rows)),
                        'task_types': dict(collections.Counter(r['Task Type'] for r in rows)), 'performance_acceptance': 'not_run'}
        paths = list((root / 'src/common/npu').glob('*.h')) + list((root / 'src/common/npu').glob('*.asc'))
        paths += [source / name for name in ('quantizer_cases.h', 'histogram_cases.h', 'split_statistics_cases.h',
                  'split_statistics_host_test.cc', 'split_statistics_test.cc', 'cuda_contract.json',
                  'check_integer_histogram.py', 'runtime_no_device_guard.cc', 'precise_test_kernels.asc',
                  'precision_kernel/CMakeLists.txt')]
        paths += [Path(__file__), host, binary, library]
        report['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['status'] = 'passed'
        print(f"SPLIT_STATISTICS_VALIDATION_PASS host_checks=16 npu_execution={report['npu_execution']} split_scoring=not_implemented")
        return 0
    except Exception as error:
        report['error'] = str(error); print(f'SPLIT_STATISTICS_VALIDATION_FAIL: {error}', file=sys.stderr); return 1
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())