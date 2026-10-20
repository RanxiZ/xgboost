#!/usr/bin/env python3
"""Validate scalar split-finalization stages; candidate reduction remains pending."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-source-root', required=True, type=Path)
    parser.add_argument('--device-test', type=int)
    parser.add_argument('--confirm-device-idle', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    require((args.device_test is None) == (not args.confirm_device_idle), 'device needs both opt-in arguments')
    require(args.device_test in (None, 0), 'single device ordinal zero only')
    require(not args.profile or args.device_test is not None, 'profile needs device opt-in')
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), 'wrong source checkout')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = root / 'tests/npu_port'; cann = Path('/usr/local/Ascend/cann-9.1.0').resolve(strict=True)
    build = out / 'kernel-build'
    env = os.environ.copy(); env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', LD_LIBRARY_PATH='')
    native = {**env, 'LD_LIBRARY_PATH': ':'.join(str(p) for p in [build, cann / 'lib64',
        cann / 'lib64/plugin/opskernel', cann / 'lib64/plugin/nnengine',
        cann / 'opp/built-in/op_impl/ai_core/tbe/op_tiling', Path('/usr/local/Ascend/driver/lib64'),
        Path('/usr/local/Ascend/driver/lib64/common'), Path('/usr/local/Ascend/driver/lib64/driver')])}
    native.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann), ASCEND_TOOLKIT_LATEST_HOME=str(cann),
        ASCEND_OPP_PATH=str(cann / 'opp'), ASCEND_AICPU_PATH=str(cann),
        PATH=f'{cann}/tools/bisheng_compiler/bin:' + env['PATH'])
    report = dict(status='failed', scope='scalar_node_gain_missing_direction_and_post_selection_subtraction',
        source_root=str(root), npu_execution='not_run', cuda_execution='not_run',
        candidate_reduction='not_implemented', training_backend='not_implemented', commands=[],
        limitations=['no candidate/feature reduction or CUB tie-equivalence claim',
            'raw ABI scalar stages only; updater integration pending',
            'post-selection subtraction timing is caller contract, not a pre-selection transform',
            'binary64 integer emulation, finite supported domain; no native FP64 claim',
            'MIX_AIC custom VF dispatch with vector arithmetic; not pure AIV-only',
            'actual CUDA FMA/FTZ behavior not measured; host formula disables contraction'])

    def run(command, name, e=None, expected=0):
        report['commands'].append(dict(argv=command, log=name, expected_exit=expected))
        with (out / name).open('w') as stream:
            result = subprocess.run(command, cwd=root, env=e or env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=600)
        require(result.returncode == expected, f'exit {result.returncode}, expected {expected}: {out/name}')
        return (out / name).read_text()

    def provenance(path):
        p = Path(path).resolve()
        require(p.is_relative_to(cann) or p.is_relative_to('/usr/local/Ascend/driver'), f'mixed CANN: {p}')

    def idle(name):
        require('No running processes found in NPU 0' in run(['npu-smi', 'info'], name, native), 'NPU occupied; deferred')

    def device_records(text):
        result = records(text)
        require('SPLIT_FINALIZATION_NPU_PASS checks=6' in text and len(result) == 6, 'incomplete NPU fixtures')
        require('cleanup failure' not in text and 'cleanup status=' not in text, 'cleanup error')
        for line in text.splitlines():
            if line.startswith('mapping '): provenance(line.split()[-1])
        return result

    try:
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip() == 'xgb-npu-port', 'wrong branch')
        flags = [os.environ.get('CXX', 'g++'), '-std=c++17', '-fopenmp', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-fno-fast-math', '-ffp-contract=off', f'-I{root/"include"}', f'-I{root/"dmlc-core/include"}']
        host = out / 'split_finalization_host_test'
        run([*flags, '-O2', str(source/'split_finalization_host_test.cc'), '-o', str(host)], 'host-compile.log')
        linked = run(['ldd', str(host)], 'host-libraries.txt')
        require(not any(x in linked.lower() for x in ('ascend', 'libcuda', 'libopapi', 'not found')), 'host linked runtime')
        text = run([str(host)], 'host-tests.log'); report['host_tests'] = records(text)
        match = re.search(r'SPLIT_FINALIZATION_HOST_PASS checks=4 nodes=(\d+) directions=(\d+) gains=(\d+)', text)
        require(match is not None and len(report['host_tests']) == 4, 'incomplete host fixtures')
        report['fixture_counts'] = dict(zip(('nodes', 'directions', 'gains'), map(int, match.groups())))
        sanitized = out / 'split_finalization_host_sanitized'
        run([*flags, '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
            str(source/'split_finalization_host_test.cc'), '-o', str(sanitized)], 'sanitizer-compile.log')
        text = run([str(sanitized)], 'sanitizer-tests.log', {**env, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
            'UBSAN_OPTIONS': 'halt_on_error=1'})
        require(records(text) == report['host_tests'], 'sanitizer fixtures differ'); report['host_sanitizers'] = 'passed'
        run(['cmake', '-S', str(source/'precision_kernel'), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_ASC_ARCHITECTURES=dav-3510'], 'kernel-configure.log', native)
        run(['cmake', '--build', str(build), '-j2'], 'kernel-build.log', native)
        library = build/'libxgb_npu_precise.so'
        require('.aicore_binary' in run(['readelf', '-S', str(library)], 'kernel-sections.txt'), 'missing device binary')
        binary = out/'split_finalization_test'
        run([*flags, '-O2', '-isystem', str(cann/'include'), str(source/'split_finalization_test.cc'),
            f'-L{build}', f'-Wl,-rpath,{build}', '-lxgb_npu_precise', f'-L{cann/"lib64"}',
            f'-Wl,-rpath,{cann/"lib64"}', '-lopapi', '-lnnopbase', '-lascendcl', '-ldl', '-o', str(binary)],
            'device-compile.log', native)
        linked = run(['ldd', str(binary)], 'device-libraries.txt', native)
        require('not found' not in linked and 'libxgboost' not in linked, 'bad standalone linkage')
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith('/usr/local/Ascend/'): provenance(word)
        guard = out/'no_device_guard.so'
        run([*flags, '-O2', '-fPIC', '-shared', '-isystem', str(cann/'include'),
            str(source/'runtime_no_device_guard.cc'), '-o', str(guard)], 'guard-compile.log')
        guarded = {**native, 'LD_PRELOAD': str(guard)}
        text = run([str(binary), '--host-api-only'], 'host-api-guard.log', guarded)
        require('SPLIT_FINALIZATION_API_HOST_PASS' in text and 'FORBIDDEN_ACL_CALL' not in text, 'host touched device')
        text = run([str(binary), '--device-test', '0', '--confirm-device-idle'], 'guard-positive.log', guarded, 86)
        require('FORBIDDEN_ACL_CALL aclInit' in text, 'guard positive control')
        for i, argv in enumerate([[], ['--device-test', '0'], ['--device-test', '0', '--wrong-ack'],
                                 ['--device-test', '1', '--confirm-device-idle']]):
            text = run([str(binary), *argv], f'argument-guard-{i}.log', guarded, 2)
            require('FORBIDDEN_ACL_CALL' not in text, 'argument guard touched device')
        report['device_call_guards'] = 'passed'
        if args.device_test is not None:
            with (out.parent/'npu0-test.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                idle('device-before.txt'); time.sleep(1); idle('device-recheck.txt')
                command = [str(binary), '--device-test', '0', '--confirm-device-idle']
                report['device_tests'] = device_records(run(command, 'device-tests.log', native))
                idle('device-after.txt'); report['npu_execution'] = 'passed'
                if args.profile:
                    profile = out/'profile'; idle('profile-before.txt')
                    text = run([str(cann/'tools/profiler/bin/msprof'), f'--output={profile}', '--runtime-api=on',
                        '--task-time=on', '--ai-core=on', '--aicpu=on', '--sys-lp=off', *command], 'profile-run.log', native)
                    require(device_records(text) == report['device_tests'], 'profile replay differs'); idle('profile-after.txt')
                    summaries = list(profile.rglob('op_summary_*.csv')); require(len(summaries) == 1, 'missing op summary')
                    with summaries[0].open() as stream: rows = list(csv.DictReader(stream))
                    require(rows and all(r['Device_id'] == '0' for r in rows), 'missing device tasks')
                    new_names = {'NodeQuantitiesKernel', 'ChooseDirectionKernel', 'SubtractParentKernel'}
                    custom_names = new_names | {'PreciseArithmeticKernel', 'PreciseScoreKernel', 'ScoreColumnsKernel',
                                                'IntegerHistogramBuildKernel'}
                    require({r['OP Type'] for r in rows} >= new_names, 'missing finalization kernels')
                    histogram = [r for r in rows if r['OP Type'] == 'IntegerHistogramBuildKernel']
                    require(histogram, 'missing fused histogram kernel evidence')
                    for row in rows:
                        if row['Task Type'] in ('AI_VECTOR_CORE', 'MIX_AIV'): continue
                        require(row['Task Type'] == 'MIX_AIC' and row['OP Type'] in custom_names, 'unexpected compute task')
                        require(float(row['aiv_vec_time(us)']) > 0 and float(row['aic_mac_time(us)']) == 0, 'mixed task vector evidence')
                    report['execution_evidence'] = dict(operator_summary=str(summaries[0]), task_count=len(rows),
                        fused_histogram_task_count=len(histogram),
                        task_types=dict(collections.Counter(r['Task Type'] for r in rows)),
                        finalization_kernel_tasks=dict(collections.Counter(r['OP Type'] for r in rows if r['OP Type'] in new_names)),
                        performance_acceptance='not_run')
        paths = list((root/'src/common/npu').glob('*.h')) + list((root/'src/common/npu').glob('*.asc'))
        paths += [source/name for name in ('split_finalization_cases.h', 'split_finalization_host_test.cc',
            'split_finalization_test.cc', 'precise_scoring_cases.h', 'precise_arithmetic_abi.h', 'precise_arithmetic.asc',
            'precise_test_kernels.asc', 'precision_kernel/CMakeLists.txt', 'split_statistics_cases.h', 'histogram_cases.h',
            'quantizer_cases.h', 'check_integer_histogram.py', 'runtime_no_device_guard.cc')]
        paths += [Path(__file__), host, binary, library]
        report['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['status'] = 'passed'
        print(f"SPLIT_FINALIZATION_VALIDATION_PASS host_groups=4 npu_execution={report['npu_execution']} candidate_reduction=not_implemented")
        return 0
    except Exception as error:
        report['error'] = str(error); print(f'SPLIT_FINALIZATION_VALIDATION_FAIL: {error}', file=sys.stderr); return 1
    finally:
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
