#!/usr/bin/env python3
"""Validate standalone NPU int64 histograms; this is not tree training.

Feature-major local bins and unique global row ids are validated before device
indexing. Parent-minus-child requires identical source and a device subset proof.
CUDA source-derived fixtures are checked, but CUDA runtime is not executed.
"""
from __future__ import annotations
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def records(text: str) -> list[dict]:
    result = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    require(len({r['test'] for r in result}) == len(result), 'duplicate test names')
    require(all(r['status'] == 'passed' for r in result), 'failed test record')
    return result


def validate_profile_rows(rows: list[dict]) -> None:
    require(bool(rows), 'missing device-task evidence')
    for row in rows:
        require(row.get('Device_id') == '0', 'histogram task is not on NPU 0')
        # The production GM-atomic SIMT kernel is emitted as a mixed AIC/AIV
        # task on DAV-3510. Only that exact kernel may use this task type;
        # existing ACLNN stages must retain their vector-core evidence.
        fused = row.get('OP Type') == 'IntegerHistogramBuildKernel'
        require(row.get('Task Type') in ('MIX_AIV', 'AI_VECTOR_CORE') or
                (fused and row.get('Task Type') == 'MIX_AIC'),
                'unexpected compute task for histogram stage')
        try:
            duration = float(row.get('Task Duration(us)', ''))
        except (TypeError, ValueError):
            raise RuntimeError('invalid histogram task duration') from None
        require(math.isfinite(duration) and duration > 0, 'nonpositive histogram task duration')
    require(any(row.get('OP Type') == 'IntegerHistogramBuildKernel' for row in rows),
            'missing fused integer histogram device evidence')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-source-root', type=Path, required=True)
    parser.add_argument('--device-test', type=int)
    parser.add_argument('--confirm-device-idle', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-output-dir', type=Path,
                        help='fresh profiler data directory, separate from executable test artifacts')
    args = parser.parse_args()
    require((args.device_test is None) == (not args.confirm_device_idle), 'device needs both opt-in arguments')
    require(args.device_test in (None, 0), 'single ordinal 0 only')
    require(not args.profile or args.device_test is not None, 'profiling needs device opt-in')
    require(args.profile_output_dir is None or args.profile, 'profile output requires --profile')
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), 'unexpected source checkout')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    cann = Path('/usr/local/Ascend/cann-9.1.0').resolve(strict=True)
    report = {'status': 'failed', 'scope': 'unique_row_int64_histogram_build_and_proven_subset_subtraction',
              'source_root': str(root), 'cann_root': str(cann), 'commands': [],
              'npu_execution': 'not_run', 'cuda_execution': 'not_run', 'training_backend': 'not_implemented',
              'limitations': ['single-device scalar target in-memory', 'feature-major local int64 bins, not ELLPACK',
                              'zero-bin feature uses a zero placeholder', 'duplicate row ids rejected',
                              'sign totals <= INT64_MAX; per-value magnitude < 2^62',
                              'synchronous correctness-first implementation; caching/performance pending']}
    env = os.environ.copy(); env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', LD_LIBRARY_PATH='')

    def run(command: list[str], name: str, e: dict | None = None, expected: int = 0) -> str:
        report['commands'].append({'argv': command, 'log': name, 'expected_exit': expected})
        with (out / name).open('w') as log:
            result = subprocess.run(command, cwd=root, env=e or env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300)
        require(result.returncode == expected, f'exit {result.returncode}, expected {expected}; see {out / name}')
        return (out / name).read_text()

    def provenance(path: str) -> None:
        p = Path(path).resolve()
        require(p.is_relative_to(cann) or p.is_relative_to('/usr/local/Ascend/driver'), f'mixed CANN: {p}')

    def idle(name: str) -> None:
        text = run(['npu-smi', 'info'], name, native_env)
        require('No running processes found in NPU 0' in text, 'NPU occupied; device test deferred')

    def device_records(text: str) -> list[dict]:
        require('INTEGER_HISTOGRAM_NPU_PASS checks=50' in text, 'missing NPU completion sentinel')
        require('cleanup failure' not in text and 'cleanup status=' not in text, 'resource cleanup diagnostic')
        result = records(text)
        require(len(result) == 50, 'unexpected device test count')
        require(any(r['test'] == 'npu_shared_immutable_bins_and_packed_lifetime' for r in result),
                'missing shared bins and packed storage lifetime regression')
        fused = {f'npu_direct_fused_{features}x{rows}' for features, rows in
                 ((1,129),(16,129),(17,129),(31,257),(32,513),(33,1025),(158,2049),(17,4))}
        require({r['test'] for r in result if r['test'].startswith('npu_direct_fused_')} == fused,
                'missing fused histogram ownership/tail/canary regression')
        expected = {r['test'][5:] for r in report['host_tests'][:10]}
        require({r['test'][11:] for r in result if r['test'].startswith('npu_direct_')
                 and not r['test'].startswith('npu_direct_fused_')} == expected,
                'host/device direct fixture mismatch')
        require({r['test'][13:] for r in result if r['test'].startswith('npu_subtract_')} == expected,
                'host/device subtraction fixture mismatch')
        for r in result:
            if 'control_download_bytes' in r:
                require(r['control_download_bytes'] == (16 if r['rows'] else 0), 'unexpected histogram transfer')
        require(any(r['test'] == 'npu_dynamic_quantizer_histogram_integration' for r in result), 'missing quantizer integration')
        for line in text.splitlines():
            if line.startswith('mapping '): provenance(line.split()[-1])
        return result

    try:
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip() ==
                'xgb-npu-port', 'unexpected branch')
        cxx = os.environ.get('CXX', 'g++'); source = root / 'tests/npu_port'
        flags = [cxx, '-std=c++17', '-fopenmp', '-Wall', '-Wextra', '-Werror', '-pedantic',
                 '-fno-fast-math', '-ffp-contract=off', f'-I{root / "include"}', f'-I{root / "dmlc-core/include"}']
        host = out / 'integer_histogram_host_test'
        run([*flags, '-O2', str(source / 'integer_histogram_host_test.cc'), '-o', str(host)], 'host-compile.log')
        linked = run(['ldd', str(host)], 'host-libraries.txt')
        require(not any(x in linked.lower() for x in ('ascend', 'libopapi', 'libcuda', 'not found')), 'host test linked device runtime')
        text = run([str(host)], 'host-tests.log')
        require('INTEGER_HISTOGRAM_HOST_PASS checks=14' in text, 'host sentinel missing')
        report['host_tests'] = records(text); require(len(report['host_tests']) == 14, 'unexpected host test count')
        fixture_lines = [line for line in text.splitlines() if line.startswith('CUDA_FIXTURE ')]
        require(len(fixture_lines) == 1, 'missing CUDA source-derived fixture')
        actual = json.loads(fixture_lines[0][13:])
        contract = json.loads((source / 'cuda_contract.json').read_text())['fixed_fixture']
        require(actual['root'] == contract['expected_root'] and actual['histogram'] == contract['expected_histogram'],
                'histogram fixture differs from pinned CUDA-source contract')
        report['cuda_source_fixture'] = 'passed; runtime not executed'
        san = out / 'integer_histogram_host_sanitized'
        run([*flags, '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
             str(source / 'integer_histogram_host_test.cc'), '-o', str(san)], 'sanitizer-compile.log')
        text = run([str(san)], 'sanitizer-tests.log', {**env, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                   'UBSAN_OPTIONS': 'halt_on_error=1'})
        require(records(text) == report['host_tests'], 'sanitized host tests differ'); report['host_sanitizers'] = 'passed'
        native_env = {**env, 'LD_LIBRARY_PATH': ':'.join(str(p) for p in [
            cann / 'lib64', cann / 'lib64/plugin/opskernel', cann / 'lib64/plugin/nnengine',
            cann / 'opp/built-in/op_impl/ai_core/tbe/op_tiling', Path('/usr/local/Ascend/driver/lib64'),
            Path('/usr/local/Ascend/driver/lib64/common'), Path('/usr/local/Ascend/driver/lib64/driver')])}
        native_env.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann),
                          ASCEND_TOOLKIT_LATEST_HOME=str(cann), ASCEND_OPP_PATH=str(cann / 'opp'), ASCEND_AICPU_PATH=str(cann))
        report['device_environment'] = {k: v for k, v in native_env.items() if k.startswith('ASCEND_') or k == 'LD_LIBRARY_PATH'}
        binary = out / 'integer_histogram_test'
        run([*flags, '-O2', '-isystem', str(cann / 'include'), str(source / 'integer_histogram_test.cc'),
             f'-L{root / "lib-npu"}', f'-Wl,-rpath,{root / "lib-npu"}', '-lxgboost_npu_kernels',
             f'-L{cann / "lib64"}', f'-Wl,-rpath,{cann / "lib64"}', '-lopapi', '-lnnopbase', '-lascendcl', '-ldl',
             '-o', str(binary)], 'device-compile.log', native_env)
        linked = run(['ldd', str(binary)], 'device-libraries.txt', native_env)
        require('not found' not in linked and 'libxgboost.so' not in linked and
                str(root / 'lib-npu/libxgboost_npu_kernels.so') in linked, 'invalid standalone linkage')
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith('/usr/local/Ascend/'): provenance(word)
        guard = out / 'no_device_guard.so'
        run([*flags, '-O2', '-fPIC', '-shared', '-isystem', str(cann / 'include'),
             str(source / 'runtime_no_device_guard.cc'), '-o', str(guard)], 'guard-compile.log')
        guarded = {**native_env, 'LD_PRELOAD': str(guard)}
        text = run([str(binary), '--host-api-only'], 'host-api-guard.log', guarded)
        require('INTEGER_HISTOGRAM_API_HOST_PASS' in text and 'FORBIDDEN_ACL_CALL' not in text, 'host API touched ACL')
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
                text = run(command, 'device-tests.log', native_env)
                report['device_tests'] = device_records(text)
                idle('device-after.txt'); report['npu_execution'] = 'passed'
                if args.profile:
                    profile = args.profile_output_dir.resolve() if args.profile_output_dir else out / 'profile'
                    require(not profile.exists(), 'profiling output exists; use fresh output directory')
                    idle('profile-before.txt')
                    text = run([str(cann / 'tools/profiler/bin/msprof'), f'--output={profile}', '--runtime-api=on',
                                '--task-time=on', '--ai-core=on', '--aicpu=on', '--sys-lp=off', *command],
                               'profile-run.log', native_env)
                    require(device_records(text) == report['device_tests'], 'profile replay differs')
                    idle('profile-after.txt')
                    summaries = list(profile.rglob('op_summary_*.csv')); require(len(summaries) == 1, 'operator summary missing or ambiguous')
                    with summaries[0].open() as stream: rows = list(csv.DictReader(stream))
                    validate_profile_rows(rows)
                    names = {r['OP Type'] for r in rows}
                    require(any('Scatter' in x for x in names) and any('Gather' in x for x in names) and
                            'ReduceSum' in names and any('Axpy' in x or 'Sub' in x for x in names),
                            'missing gather/scatter/reduction/subtraction device evidence')
                    report['execution_evidence'] = {'operator_summary': str(summaries[0]), 'task_count': len(rows),
                        'op_counts': dict(collections.Counter(r['OP Type'] for r in rows)),
                        'task_types': dict(collections.Counter(r['Task Type'] for r in rows)), 'performance_acceptance': 'not_run'}
        paths = [root / 'src/common/npu' / name for name in
                 ('runtime.h', 'acl_api.h', 'acl_ops.h', 'histogram_contract.h', 'integer_histogram.h',
                  'integer_histogram.asc', 'integer_histogram_abi.h', 'kernel/production_kernels.asc',
                  'quantizer_scale.h', 'certified_scale.h', 'fixed_quantizer.h', 'dynamic_quantizer.h')]
        paths += [source / name for name in ('quantizer_cases.h', 'histogram_cases.h', 'integer_histogram_host_test.cc',
                                            'integer_histogram_test.cc', 'cuda_contract.json')]
        paths += [Path(__file__), host, binary, root / 'lib-npu/libxgboost_npu_kernels.so']
        report['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['status'] = 'passed'
        print(f"INTEGER_HISTOGRAM_VALIDATION_PASS host_checks=14 npu_execution={report['npu_execution']} training_backend=not_implemented")
        return 0
    except Exception as error:
        report['error'] = str(error); print(f'INTEGER_HISTOGRAM_VALIDATION_FAIL: {error}', file=sys.stderr)
        return 1
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
