#!/usr/bin/env python3
"""Validate device bounds and certified dynamic scales, not tree training.

CUDA runtime comparison is not performed. A scale is accepted only if its
exponent is constant over an enclosure of every RN binary64 addition tree.
Ambiguous certificates are explicit unsupported inputs, never CPU fallbacks.
"""
from __future__ import annotations
import argparse
import collections
import csv
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def records(text: str) -> list[dict]:
    result = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    require(len({r['test'] for r in result}) == len(result), 'duplicate test names')
    require(all(r['status'] == 'passed' for r in result), 'failed test record')
    return result


def exact_intervals(text: str) -> int:
    fixtures = [json.loads(line[9:]) for line in text.splitlines() if line.startswith('INTERVAL ')]
    require(len(fixtures) == 136, 'missing exact interval fixtures')
    for fixture in fixtures:
        bins = fixture['bins']
        require(len(bins) == 512, 'wrong bucket count')
        for offset, name in ((0, 'positive'), (256, 'negative')):
            exact = Fraction(0)
            for e in range(255):
                power = max(e, 1) - 150
                exact += Fraction(bins[offset + e]) * (Fraction(2) ** power)
            lo, hi = [Fraction.from_float(float.fromhex(x)) for x in fixture[name]]
            require(0 <= lo <= exact <= hi, f'exact dyadic sum escaped {name} interval')
    return len(fixtures)


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
    require(not args.profile or args.device_test is not None, 'profiling needs device opt-in')
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), 'unexpected source checkout')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    cann = Path('/usr/local/Ascend/cann-9.1.0').resolve(strict=True)
    report = {'status': 'failed', 'scope': 'device_bounds_certified_dynamic_scale_quantization_and_root',
              'source_root': str(root), 'cann_root': str(cann), 'commands': [],
              'cuda_execution': 'not_run', 'npu_execution': 'not_run', 'training_backend': 'not_implemented',
              'limitation': 'uncertified reduction-order scales rejected, including conservative ambiguities'}
    env = os.environ.copy(); env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', LD_LIBRARY_PATH='')

    def run(command: list[str], name: str, e: dict | None = None, expected: int = 0) -> str:
        report['commands'].append({'argv': command, 'log': name, 'expected_exit': expected})
        with (out / name).open('w') as log:
            result = subprocess.run(command, cwd=root, env=e or env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=240)
        require(result.returncode == expected, f'exit {result.returncode}, expected {expected}; see {out / name}')
        return (out / name).read_text()

    def provenance(path: str) -> None:
        p = Path(path).resolve()
        require(p.is_relative_to(cann) or p.is_relative_to('/usr/local/Ascend/driver'), f'mixed CANN: {p}')

    def idle(name: str) -> None:
        text = run(['npu-smi', 'info'], name, native_env)
        require('No running processes found in NPU 0' in text, 'NPU occupied; test deferred')

    def device_records(text: str) -> list[dict]:
        require('DYNAMIC_QUANTIZER_NPU_PASS checks=24' in text, 'missing NPU completion sentinel')
        require('cleanup failure' not in text and 'cleanup status=' not in text, 'resource cleanup diagnostic')
        result = records(text)
        require(len(result) == 24, 'unexpected device test count')
        require({r['test'][4:] for r in result if 'rows' in r} ==
                {r['device_fixture'] for r in report['host_tests'] if 'device_fixture' in r},
                'host/device fixture mismatch')
        for r in result:
            if 'fit_control_download_bytes' in r:
                expected = 8 if 'infinity' in r['test'] or 'nan' in r['test'] else (
                    4104 if 'reject_' in r['test'] else 4152)
                require(r['fit_control_download_bytes'] == expected, 'unexpected component transfer')
        for line in text.splitlines():
            if line.startswith('mapping '): provenance(line.split()[-1])
        return result

    try:
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip() ==
                'xgb-npu-port', 'unexpected branch')
        cxx = os.environ.get('CXX', 'g++'); source = root / 'tests/npu_port'
        flags = [cxx, '-std=c++17', '-fopenmp', '-Wall', '-Wextra', '-Werror', '-pedantic',
                 '-fno-fast-math', '-ffp-contract=off', f'-I{root / "include"}', f'-I{root / "dmlc-core/include"}']
        host = out / 'dynamic_quantizer_host_test'
        run([*flags, '-O2', str(source / 'dynamic_quantizer_host_test.cc'), '-o', str(host)], 'host-compile.log')
        linked = run(['ldd', str(host)], 'host-libraries.txt')
        require(not any(x in linked.lower() for x in ('ascend', 'libopapi', 'libcuda', 'not found')),
                'host oracle linked device runtime')
        text = run([str(host)], 'host-tests.log')
        require('DYNAMIC_QUANTIZER_HOST_PASS checks=28' in text, 'host sentinel missing')
        report['host_tests'] = records(text)
        require(len(report['host_tests']) == 28, 'unexpected host test count')
        report['exact_dyadic_interval_fixtures'] = exact_intervals(text)
        san = out / 'dynamic_quantizer_host_sanitized'
        run([*flags, '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
             str(source / 'dynamic_quantizer_host_test.cc'), '-o', str(san)], 'sanitizer-compile.log')
        text = run([str(san)], 'sanitizer-tests.log', {**env, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                   'UBSAN_OPTIONS': 'halt_on_error=1'})
        require(records(text) == report['host_tests'], 'sanitized host tests differ')
        exact_intervals(text); report['host_sanitizers'] = 'passed'
        native_env = {**env, 'LD_LIBRARY_PATH': ':'.join(str(p) for p in [
            cann / 'lib64', cann / 'lib64/plugin/opskernel', cann / 'lib64/plugin/nnengine',
            cann / 'opp/built-in/op_impl/ai_core/tbe/op_tiling', Path('/usr/local/Ascend/driver/lib64'),
            Path('/usr/local/Ascend/driver/lib64/common'), Path('/usr/local/Ascend/driver/lib64/driver')])}
        native_env.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann),
                          ASCEND_TOOLKIT_LATEST_HOME=str(cann), ASCEND_OPP_PATH=str(cann / 'opp'), ASCEND_AICPU_PATH=str(cann))
        report['device_environment'] = {k: v for k, v in native_env.items() if k.startswith('ASCEND_') or k == 'LD_LIBRARY_PATH'}
        binary = out / 'dynamic_quantizer_test'
        run([*flags, '-O2', '-isystem', str(cann / 'include'), str(source / 'dynamic_quantizer_test.cc'),
             f'-L{cann / "lib64"}', f'-Wl,-rpath,{cann / "lib64"}', '-lopapi', '-lnnopbase',
             '-lascendcl', '-ldl', '-o', str(binary)], 'device-compile.log', native_env)
        linked = run(['ldd', str(binary)], 'device-libraries.txt', native_env)
        require('not found' not in linked and 'libxgboost' not in linked, 'invalid standalone linkage')
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith('/usr/local/Ascend/'): provenance(word)
        guard = out / 'no_device_guard.so'
        run([*flags, '-O2', '-fPIC', '-shared', '-isystem', str(cann / 'include'),
             str(source / 'runtime_no_device_guard.cc'), '-o', str(guard)], 'guard-compile.log')
        guarded = {**native_env, 'LD_PRELOAD': str(guard)}
        text = run([str(binary), '--host-api-only'], 'host-api-guard.log', guarded)
        require('DYNAMIC_QUANTIZER_API_HOST_PASS' in text and 'FORBIDDEN_ACL_CALL' not in text, 'host API touched ACL')
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
                    profile = out / 'profile'
                    require(not profile.exists(), 'profiling output exists; use fresh output directory')
                    idle('profile-before.txt')
                    text = run([str(cann / 'tools/profiler/bin/msprof'), f'--output={profile}', '--runtime-api=on',
                                '--task-time=on', '--ai-core=on', '--aicpu=on', '--sys-lp=off', *command],
                               'profile-run.log', native_env)
                    require(device_records(text) == report['device_tests'], 'profile replay differs')
                    idle('profile-after.txt')
                    files = list(profile.rglob('op_summary_*.csv'))
                    require(len(files) == 1, 'operator summary missing or ambiguous')
                    with files[0].open() as stream: rows = list(csv.DictReader(stream))
                    require(rows and all(r['Device_id'] == '0' for r in rows), 'missing device-task evidence')
                    require(all(r['Task Type'] in ('MIX_AIV', 'AI_VECTOR_CORE') for r in rows), 'non-vector compute task')
                    names = {r['OP Type'] for r in rows}
                    require(any('Scatter' in x for x in names) and 'RightShift' in names and 'ReduceSum' in names,
                            'missing bounds/quantization/root operator evidence')
                    report['execution_evidence'] = {'operator_summary': str(files[0]), 'task_count': len(rows),
                        'op_counts': dict(collections.Counter(r['OP Type'] for r in rows)),
                        'task_types': dict(collections.Counter(r['Task Type'] for r in rows)), 'performance_acceptance': 'not_run'}
        paths = [root / 'src/common/npu' / name for name in
                 ('runtime.h', 'acl_api.h', 'acl_ops.h', 'quantizer_scale.h', 'certified_scale.h', 'fixed_quantizer.h', 'dynamic_quantizer.h')]
        paths += [source / name for name in ('quantizer_cases.h', 'dynamic_quantizer_cases.h', 'dynamic_quantizer_host_test.cc', 'dynamic_quantizer_test.cc')]
        paths += [Path(__file__), host, binary]
        report['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['status'] = 'passed'
        print(f"DYNAMIC_QUANTIZER_VALIDATION_PASS host_checks=28 exact_intervals=136 npu_execution={report['npu_execution']}")
        return 0
    except Exception as error:
        report['error'] = str(error); print(f'DYNAMIC_QUANTIZER_VALIDATION_FAIL: {error}', file=sys.stderr)
        return 1
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
