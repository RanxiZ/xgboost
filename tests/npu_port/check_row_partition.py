#!/usr/bin/env python3
"""Validate independent NPU row partitioning; device execution requires explicit opt-in."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIN = '6fe8c547bdd21c73e4555d85b087d9260595d30d'
# Fixed official-source artifact identities, checked against both checkout and git.
PINNED_SHA256 = {
    'src/tree/gpu_hist/row_partitioner.cu': '5f220fb58a6fc6bfeefa2a80920250e0e870e1a8244aca474ff01ec1d3b51ddd',
    'src/tree/gpu_hist/row_partitioner.cuh': '8f4b917b98bc9e49132a6e286afb66d088cfe69170ef207b76efcb3e6397ea67',
    'src/tree/gpu_hist/evaluate_splits.cu': '162d6837fa8802a17c1731a6fa09979523e3107f5c79137eba43aebd5b12b2c0',
    'src/tree/updater_gpu_hist.cu': '0624991e85efbaca5cecd3dcac2abc5762f8f0d24f0e0e11e79bd0383d398560',
    'src/data/ellpack_page.cuh': '62beed5517191378c68f392a10885bbf28e8bc8561efbe20fa683b14f72cdd72',
    'src/predictor/predict_fn.h': '8a0d899edc165bc66a017204973d3ab27578ba74fd7e71fa2e885a9492e8d3b4',
}
FROZEN_FIXTURES_SHA256 = {
    'tests/npu_port/row_partition_cases.h': 'a1c267cad89facc0697a0feba2767fcab456db166b12a3645d0e77b142cd6648',
    'tests/npu_port/row_partition_host_test.cc': '11b141eb13a4d2d934a8716e010468894c96b4c23e5edd9d54c300e9f1e2dc8b',
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def records(text: str) -> list[dict]:
    result = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    require(len({r['test'] for r in result}) == len(result), 'duplicate test records')
    require(all(r['status'] == 'passed' for r in result), 'failed test record')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-source-root', type=Path, required=True)
    parser.add_argument('--device-test', type=int)
    parser.add_argument('--confirm-device-idle', action='store_true')
    args = parser.parse_args()
    require((args.device_test is None) == (not args.confirm_device_idle), 'device requires both opt-in arguments')
    require(args.device_test in (None, 0), 'only device ordinal 0 is supported')
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), 'unexpected source root')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    cann = Path('/usr/local/Ascend/cann-9.1.0').resolve(strict=True)
    report = {'status': 'failed', 'scope': 'independent_numerical_row_partition_only',
              'source_root': str(root), 'cann_root': str(cann), 'reference_commit': PIN,
              'npu_execution': 'not_run', 'cuda_execution': 'not_run', 'profiling': 'not_run',
              'training_backend': 'not_implemented', 'commands': [],
              'limitations': ['single numerical node, scalar target, in-memory immutable dataset/runtime',
                              'candidate-local bin index only; no raw threshold or categorical routing',
                              'source cuts/binning provenance is external to the histogram component',
                              'zero-bin features have no candidates and are rejected',
                              'no final leaf ids, updater integration, or performance acceptance',
                              'host reference is source-derived, not CUDA runtime parity']}
    env = os.environ.copy(); env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', LD_LIBRARY_PATH='')
    native = {**env, 'LD_LIBRARY_PATH': ':'.join(str(p) for p in [
        cann/'lib64', cann/'lib64/plugin/opskernel', cann/'lib64/plugin/nnengine',
        cann/'opp/built-in/op_impl/ai_core/tbe/op_tiling', Path('/usr/local/Ascend/driver/lib64'),
        Path('/usr/local/Ascend/driver/lib64/common'), Path('/usr/local/Ascend/driver/lib64/driver')])}
    native.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann),
                  ASCEND_TOOLKIT_LATEST_HOME=str(cann), ASCEND_OPP_PATH=str(cann/'opp'), ASCEND_AICPU_PATH=str(cann))
    report['device_environment'] = {k:v for k,v in native.items() if k.startswith('ASCEND_') or k == 'LD_LIBRARY_PATH'}

    def run(command: list[str], name: str, e: dict | None = None, expected: int = 0) -> str:
        report['commands'].append({'argv': command, 'log': name, 'expected_exit': expected})
        with (out/name).open('w') as log:
            proc = subprocess.run(command, cwd=root, env=e or env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        require(proc.returncode == expected, f'exit {proc.returncode}, expected {expected}; see {out/name}')
        return (out/name).read_text()

    def provenance(path: str) -> None:
        p = Path(path).resolve()
        require(p.is_relative_to(cann) or p.is_relative_to('/usr/local/Ascend/driver'), f'mixed CANN library: {p}')

    def idle(name: str) -> None:
        require('No running processes found in NPU 0' in run(['npu-smi', 'info'], name, native),
                'NPU occupied; device test deferred')

    try:
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip() ==
                'xgb-npu-port', 'unexpected branch')
        require(len(PINNED_SHA256) == 6, 'missing frozen CUDA artifact hashes')
        for path, digest in PINNED_SHA256.items():
            official = subprocess.check_output(['git', 'show', f'{PIN}:{path}'], cwd=root)
            require(hashlib.sha256(official).hexdigest() == digest and
                    hashlib.sha256((root/path).read_bytes()).hexdigest() == digest, f'changed routing source: {path}')
        report['pinned_sources_sha256'] = PINNED_SHA256
        for path, digest in FROZEN_FIXTURES_SHA256.items():
            require(hashlib.sha256((root/path).read_bytes()).hexdigest() == digest, f'changed frozen fixture: {path}')
        report['frozen_fixture_sha256'] = FROZEN_FIXTURES_SHA256
        source = root/'tests/npu_port'; cxx = os.environ.get('CXX', 'g++')
        flags = [cxx, '-std=c++17', '-fopenmp', '-Wall', '-Wextra', '-Werror', '-pedantic',
                 '-fno-fast-math', '-ffp-contract=off', f'-I{root/"include"}', f'-I{root/"dmlc-core/include"}']
        host = out/'row_partition_host_test'
        run([*flags, '-O2', str(source/'row_partition_host_test.cc'), '-o', str(host)], 'host-compile.log')
        linked = run(['ldd', str(host)], 'host-libraries.txt')
        require(not any(x in linked.lower() for x in ('ascend', 'libopapi', 'libcuda', 'not found')), 'host links device runtime')
        text = run([str(host)], 'host-tests.log')
        require('ROW_PARTITION_HOST_PASS checks=27' in text, 'missing host sentinel')
        report['host_tests'] = records(text); require(len(report['host_tests']) == 27, 'host count mismatch')
        sanitized = out/'row_partition_host_sanitized'
        run([*flags, '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
             str(source/'row_partition_host_test.cc'), '-o', str(sanitized)], 'sanitizer-compile.log')
        text = run([str(sanitized)], 'sanitizer-tests.log', {**env, 'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1',
                   'UBSAN_OPTIONS':'halt_on_error=1'})
        require(records(text) == report['host_tests'], 'sanitized results differ'); report['host_sanitizers'] = 'passed'
        binary = out/'row_partition_test'
        library = (root/'lib-npu/libxgboost_npu_kernels.so').resolve(strict=True)
        run([*flags, '-O2', '-isystem', str(cann/'include'), str(source/'row_partition_test.cc'),
             str(library), f'-Wl,-rpath,{library.parent}',
             f'-L{cann/"lib64"}', f'-Wl,-rpath,{cann/"lib64"}', '-lopapi', '-lnnopbase', '-lascendcl', '-ldl',
             '-o', str(binary)], 'device-compile.log', native)
        linked = run(['ldd', str(binary)], 'device-libraries.txt', native)
        require('not found' not in linked and 'libxgboost.so' not in linked and str(library) in linked,
                'invalid native test linkage')
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith('/usr/local/Ascend/'): provenance(word)
        guard = out/'no_device_guard.so'
        run([*flags, '-O2', '-fPIC', '-shared', '-isystem', str(cann/'include'),
             str(source/'runtime_no_device_guard.cc'), '-o', str(guard)], 'guard-compile.log')
        guarded = {**native, 'LD_PRELOAD':str(guard)}
        text = run([str(binary), '--host-api-only'], 'host-api-guard.log', guarded)
        require('ROW_PARTITION_API_HOST_PASS' in text and 'FORBIDDEN_ACL_CALL' not in text, 'host API touched device')
        text = run([str(binary), '--device-test', '0', '--confirm-device-idle'], 'guard-positive.log', guarded, 86)
        require('FORBIDDEN_ACL_CALL aclInit' in text, 'guard positive control failed')
        for i, argv in enumerate([[], ['--device-test','0'], ['--device-test','0','--wrong-ack'],
                                  ['--device-test','1','--confirm-device-idle']]):
            text = run([str(binary), *argv], f'argument-guard-{i}.log', guarded, 2)
            require('FORBIDDEN_ACL_CALL' not in text, 'invalid arguments touched ACL')
        report['device_call_guards'] = 'passed'
        if args.device_test is not None:
            with (out.parent/'npu0-test.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                idle('device-before.txt'); time.sleep(1); idle('device-recheck.txt')
                text = run([str(binary), '--device-test', '0', '--confirm-device-idle'], 'device-tests.log', native)
                require('ROW_PARTITION_NPU_PASS checks=44' in text, 'missing device completion sentinel')
                require('cleanup failure' not in text and 'cleanup status=' not in text, 'resource cleanup failed')
                report['device_tests'] = records(text); require(len(report['device_tests']) == 44, 'device count mismatch')
                require({'npu_buffered_ranges_mixed_depth_and_lifetime',
                         'npu_buffered_consuming_guards_preserve_parent'} <=
                        {r['test'] for r in report['device_tests']}, 'buffered row regressions missing')
                expected = {r['test'].removeprefix('host_') for r in report['host_tests'][:16]}
                require({r['test'].removeprefix('npu_') for r in report['device_tests'][:16]} == expected,
                        'host/device fixtures differ')
                for line in text.splitlines():
                    if line.startswith('mapping '): provenance(line.split()[-1])
                idle('device-after.txt'); report['npu_execution'] = 'passed'
        paths = [root/'src/common/npu'/name for name in
                 ('acl_api.h','acl_ops.h','integer_histogram.h','integer_histogram_abi.h',
                  'integer_histogram.asc','kernel/production_kernels.asc','device_row_partitioner.h',
                  'row_partition_contract.h')]
        paths += [source/name for name in ('row_partition_cases.h','row_partition_host_test.cc','row_partition_test.cc',
                                          'runtime_no_device_guard.cc')]
        paths += [Path(__file__),host,binary,library]
        report['artifact_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['status'] = 'passed'
        print(f"ROW_PARTITION_VALIDATION_PASS host_checks=27 npu_execution={report['npu_execution']} training_backend=not_implemented")
        return 0
    except Exception as error:
        report['error'] = str(error); print(f'ROW_PARTITION_VALIDATION_FAIL: {error}', file=sys.stderr); return 1
    finally:
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())