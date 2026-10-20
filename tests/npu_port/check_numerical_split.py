#!/usr/bin/env python3
"""Pinned CUB source/host checks and opt-in standalone NPU numerical split validation.

No default device initialization or CUDA execution. Independent component success
is not native XGBoost training success. The host compiler never links CANN.
"""
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
import urllib.request
from check_integer_histogram import require, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-source-root', required=True, type=Path)
    parser.add_argument('--cccl-source', type=Path, help='optional already acquired official pinned source tree')
    parser.add_argument('--kernel-build', type=Path, help='optional existing standalone kernel build, rebuilt and hashed')
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
    source = root/'tests/npu_port'
    env = os.environ.copy(); env.pop('LD_PRELOAD', None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', LD_LIBRARY_PATH='')
    report = dict(status='failed', scope='standalone_histogram_statistics_precise_scores_candidate_selection_numerical_evaluator',
        source_root=str(root), cuda_execution='not_run', npu_execution='not_run', native_training='not_run', commands=[],
        limitations=['CCCL 2.8.2 pinned contract only; source evidence is not CUDA device execution',
            'correctness-oriented serial vector-core reduction; performance acceptance not run',
            'caller supplies certified quantization exponents; native dataset scale binding pending',
            'unconstrained numerical single-target single-device component; no native training acceptance',
            'binary64 integer emulation finite supported domain, no native FP64 or measured CUDA FMA/FTZ claim',
            'custom MIX_AIC VF dispatch uses vector arithmetic, not pure AIV-only'])
    artifacts = [Path(__file__)]
    inputs = list((root/'src/common/npu').glob('*.h'))+list((root/'src/common/npu').glob('*.asc'))
    inputs += [p for p in source.glob('*') if p.is_file() and p.suffix in ('.cc','.h','.asc','.py','.json')]
    inputs += [source/'precision_kernel/CMakeLists.txt']
    initial_sha256 = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}

    def run(command, name, e=None, expected=0):
        command = list(map(str, command)); report['commands'].append(dict(argv=command, log=name, expected_exit=expected))
        with (out/name).open('w') as stream:
            result = subprocess.run(command, cwd=root, env=e or env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        require(result.returncode == expected, f'exit {result.returncode}, expected {expected}: {out/name}')
        return (out/name).read_text()

    try:
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip() == 'xgb-npu-port', 'wrong branch')
        contract = json.loads((source/'candidate_selection_contract.json').read_text())
        evidence = out/'cccl-source'; evidence.mkdir()
        for name, meta in contract['files'].items():
            path = evidence/name; path.parent.mkdir(parents=True, exist_ok=True)
            data = (args.cccl_source/name).read_bytes() if args.cccl_source else urllib.request.urlopen(meta['url'], timeout=60).read()
            require(hashlib.sha256(data).hexdigest() == meta['sha256'], f'pinned CCCL hash mismatch: {name}')
            path.write_bytes(data); artifacts.append(path)
        report['cccl_contract'] = contract
        cuda_commit = '6fe8c547bdd21c73e4555d85b087d9260595d30d'
        report['cuda_source_sha256'] = {}
        for name in ('src/tree/gpu_hist/evaluate_splits.cu', 'src/tree/updater_gpu_common.cuh'):
            path = root/name
            pinned = subprocess.check_output(['git', 'show', f'{cuda_commit}:{name}'], cwd=root)
            require(path.read_bytes() == pinned, f'CUDA source changed; contract review required: {name}')
            report['cuda_source_sha256'][name] = hashlib.sha256(pinned).hexdigest()
            artifacts.append(path)
        # Compile the ACTUAL official ArgMax struct, not an invented min-lane
        # reference. Only CUDA declaration attributes and KeyValuePair storage
        # are adapted for ordinary C++; the operator body is unchanged.
        text = (evidence/'cub/cub/thread/thread_operators.cuh').read_text()
        operator = text[text.index('struct ArgMax'):text.index('struct ArgMin')]
        operator = operator[:operator.index('};')+2]
        prefix = '#define _CCCL_HOST_DEVICE\n#define _CCCL_FORCEINLINE inline\ntemplate<class K,class V> struct KeyValuePair {K key; V value;};\n'
        official = out/'official_cub_argmax.cc'
        official.write_text('#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <random>\n#include <iostream>\n#include "'+str(root/'src/common/npu/candidate_selection_abi.h')+'"\n'+prefix+operator+'''
int main() {
  namespace p=xgboost::common::npu::precise;
  std::mt19937 rng{20261017};
  for (unsigned i=0;i<100000;++i) {
    uint32_t a=rng(),b=rng();
    if ((a&0x7fffffffU)>=0x7f800000U || (b&0x7fffffffU)>=0x7f800000U) continue;
    if (i<100) {a=(i&1)?0:0x80000000U;b=(i&2)?0:0x80000000U;}
    float av,bv;std::memcpy(&av,&a,4);std::memcpy(&bv,&b,4);
    uint32_t ak=rng()%32,bk=rng()%32;
    auto expected=ArgMax{}(KeyValuePair<uint32_t,float>{ak,av},KeyValuePair<uint32_t,float>{bk,bv});
    auto x=p::EmptyCandidate(),y=x;x.candidate=ak;y.candidate=bk;x.score.gain=a;y.score.gain=b;
    auto actual=p::LaneArgMax(x,y);uint32_t bits;std::memcpy(&bits,&expected.value,4);
    if (actual.candidate!=expected.key || actual.score.gain!=bits) return 1;
  }
  std::cout<<"OFFICIAL_CUB_ARGMAX_PASS trials=100000 source_only=true\\n";
}
''')
        artifacts.append(official)
        flags = [os.environ.get('CXX', 'g++'), '-std=c++17', '-fopenmp', '-Wall', '-Wextra', '-Werror', '-pedantic',
            '-fno-fast-math', '-ffp-contract=off', f'-I{root/"include"}', f'-I{root/"dmlc-core/include"}']
        run([*flags, '-O2', official, '-o', out/'official_cub_argmax'], 'official-cub-compile.log')
        report['official_argmax'] = run([out/'official_cub_argmax'], 'official-cub-tests.log').strip()
        report['host_tests'] = []
        for component in ('candidate_selection', 'numerical_split'):
            host = out/(component+'_host_test'); artifacts.append(host)
            run([*flags, '-O2', source/(component+'_host_test.cc'), '-o', host], component+'-host-compile.log')
            linked = run(['ldd', host], component+'-host-libraries.txt')
            require(not any(x in linked.lower() for x in ('ascend', 'libcuda', 'libopapi', 'not found')), 'host linked device runtime')
            text = run([host], component+'-host-tests.log'); fixtures = records(text)
            require(len(fixtures) == 4 and '_HOST_PASS checks=4' in text, 'incomplete host fixtures')
            report['host_tests'] += fixtures
            sanitized = out/(component+'_sanitized')
            run([*flags, '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer', source/(component+'_host_test.cc'), '-o', sanitized], component+'-sanitizer-compile.log')
            text = run([sanitized], component+'-sanitizer-tests.log', {**env, 'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS':'halt_on_error=1'})
            require(records(text) == fixtures, 'sanitizer fixtures differ')
        report['host_sanitizers'] = 'passed'
        # Compile the production VF bodies as ordinary C++ to exercise physical
        # block bounds without CANN. Invalid blocks receive null payloads: any
        # metadata or output access before the guard must fail under sanitizers.
        kernel_text = (root/'src/common/npu/numerical_split_evaluator.asc').read_text()
        begin = kernel_text.index('__simt_callee__ inline __gm__ ne::NodeBatchFeature const* BatchFeatures(')
        end = kernel_text.index('extern "C" __global__ __aicore__ void BatchedSplitPrefixKernel(')
        vf_host = out/'numerical_split_block_guard.cc'
        vf_host.write_text('''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <memory>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(n)
#include "src/common/npu/numerical_split_evaluator_abi.h"
#include "src/common/npu/candidate_selection_device.h"
namespace ne=xgboost::common::npu::precise;
struct { unsigned x; } threadIdx;
''' + kernel_text[begin:end] + '''
int main() {
  for (uint32_t blocks : {1U, 16U}) {
    for (uint32_t block : {blocks, blocks+1, blocks*3}) {
      for (unsigned lane=0; lane<128; ++lane) {
        threadIdx.x=lane;
        BatchedSplitPrefixVf(nullptr,nullptr,nullptr,nullptr,block,blocks);
        BatchedSplitScoreVf(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,block,blocks);
        BatchedSplitSelectVf(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,block,blocks);
        BatchedSplitFinishVf(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,block,blocks);
      }
    }
  }
  // Logical blocks with no features must still leave every output untouched.
  auto metadata=std::make_unique<uint64_t[]>((sizeof(ne::NodeBatchHeader)+sizeof(ne::NodeBatchFeature))/8);
  auto* header=reinterpret_cast<ne::NodeBatchHeader*>(metadata.get());
  int64_t pg=123,ph=456;uint32_t ready=789;
  auto selected=ne::EmptyCandidate();
  for (unsigned lane=0;lane<128;++lane) {
    threadIdx.x=lane;
    BatchedSplitPrefixVf(header,&pg,&ph,&ready,0,1);
    BatchedSplitScoreVf(header,nullptr,nullptr,&pg,&ph,&ready,nullptr,&ready,0,1);
    BatchedSplitSelectVf(header,&ready,nullptr,&ready,&selected,&ready,0,1);
  }
  assert(pg==123 && ph==456 && ready==789 && !selected.valid);
}
''')
        artifacts.append(vf_host)
        for label, options in [('host', ['-O2']), ('sanitizer', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'])]:
            binary = out/('numerical_split_block_guard_'+label)
            run([*flags, f'-I{root}', *options, vf_host, '-o', binary], 'block-guard-'+label+'-compile.log')
            run([binary], 'block-guard-'+label+'-tests.log', {**env, 'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS':'halt_on_error=1'})
        report['host_kernel_block_guards'] = 'passed_production_vf_bodies_with_sanitizers'
        if args.device_test is not None:
            cann = Path('/usr/local/Ascend/cann-9.1.0').resolve(strict=True)
            build = args.kernel_build.resolve() if args.kernel_build else out/'kernel-build'
            native = {**env, 'LD_LIBRARY_PATH': ':'.join(map(str, [build, cann/'lib64', cann/'lib64/plugin/opskernel',
                cann/'lib64/plugin/nnengine', cann/'opp/built-in/op_impl/ai_core/tbe/op_tiling',
                Path('/usr/local/Ascend/driver/lib64'),Path('/usr/local/Ascend/driver/lib64/common'),Path('/usr/local/Ascend/driver/lib64/driver')]))}
            native.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann), ASCEND_TOOLKIT_LATEST_HOME=str(cann),
                ASCEND_OPP_PATH=str(cann/'opp'), ASCEND_AICPU_PATH=str(cann), PATH=f'{cann}/tools/bisheng_compiler/bin:'+env['PATH'])
            run(['cmake','-S',source/'precision_kernel','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_ASC_ARCHITECTURES=dav-3510'], 'kernel-configure.log', native)
            run(['cmake','--build',build,'-j2'], 'kernel-build.log', native)
            library = build/'libxgb_npu_precise.so'; artifacts.append(library)
            require('.aicore_binary' in run(['readelf','-S',library], 'kernel-sections.txt'), 'missing device binary')
            binary = out/'numerical_split_test'; artifacts.append(binary)
            run([*flags,'-O2','-isystem',cann/'include',source/'numerical_split_test.cc',f'-L{build}',f'-Wl,-rpath,{build}',
                '-lxgb_npu_precise',f'-L{cann/"lib64"}',f'-Wl,-rpath,{cann/"lib64"}','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',binary], 'device-compile.log', native)
            linked = run(['ldd',binary], 'device-libraries.txt', native)
            require('not found' not in linked and 'libxgboost' not in linked, 'bad standalone linkage')
            guard = out/'no_device_guard.so'
            run([*flags,'-O2','-fPIC','-shared','-isystem',cann/'include',source/'runtime_no_device_guard.cc','-o',guard], 'guard-compile.log')
            guarded = {**native,'LD_PRELOAD':str(guard)}
            text = run([binary,'--host-api-only'], 'host-api-guard.log', guarded)
            require('NUMERICAL_SPLIT_API_HOST_PASS' in text and 'FORBIDDEN_ACL_CALL' not in text, 'host touched device')
            text = run([binary,'--device-test','0','--confirm-device-idle'], 'guard-positive.log', guarded,86)
            require('FORBIDDEN_ACL_CALL aclInit' in text, 'guard positive control')
            for i,argv in enumerate([[],['--device-test','0'],['--device-test','0','--wrong-ack'],['--device-test','1','--confirm-device-idle']]):
                text = run([binary,*argv],f'argument-guard-{i}.log',guarded,2);require('FORBIDDEN_ACL_CALL' not in text,'argument guard touched device')
            report['device_call_guards'] = 'passed'

            def idle(name):
                require('No running processes found in NPU 0' in run(['npu-smi','info'],name,native),'NPU occupied; deferred')

            def device_records(text):
                result=records(text);require('NUMERICAL_SPLIT_NPU_PASS checks=26' in text and len(result)==26,'incomplete NPU fixtures')
                require(any(r['test']=='npu_multi_node_isolation_single_workspace_and_download' for r in result),
                        'missing multi-node isolation and workspace regression')
                require('cleanup failure' not in text and 'cleanup status=' not in text,'cleanup error')
                for line in text.splitlines():
                    if line.startswith('mapping '):
                        p=Path(line.split()[-1]).resolve();require(p.is_relative_to(cann) or p.is_relative_to('/usr/local/Ascend/driver'),'mixed CANN mapping')
                return result

            with (out.parent/'npu0-test.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                idle('device-before.txt');time.sleep(1);idle('device-recheck.txt')
                command=[binary,'--device-test','0','--confirm-device-idle']
                report['device_tests']=device_records(run(command,'device-tests.log',native));idle('device-after.txt');report['npu_execution']='passed'
                if args.profile:
                    profile=out/'profile';idle('profile-before.txt');time.sleep(1);idle('profile-recheck.txt')
                    text=run([cann/'tools/profiler/bin/msprof',f'--output={profile}','--runtime-api=on','--task-time=on','--ai-core=on','--aicpu=on','--sys-lp=off',*command],'profile-run.log',native)
                    require(device_records(text)==report['device_tests'],'profile replay differs');idle('profile-after.txt')
                    summaries=list(profile.rglob('op_summary_*.csv'));require(len(summaries)==1,'missing op summary')
                    with summaries[0].open() as stream: rows=list(csv.DictReader(stream))
                    required={'BatchedSplitPrefixKernel','BatchedSplitScoreKernel','BatchedSplitSelectKernel','BatchedSplitFinishKernel',
                              'IntegerHistogramBuildKernel'}
                    require(rows and {r['OP Type'] for r in rows}>=required,'missing evaluator kernels')
                    for row in rows:
                        require(row['Device_id']=='0' and 'AI_CPU' not in row['Task Type'],'wrong device or AI_CPU')
                        if row['Task Type'] in ('AI_VECTOR_CORE','MIX_AIV'): continue
                        require(row['Task Type']=='MIX_AIC' and row['OP Type'] in required|{'SelectCandidatesKernel','SplitMergeKernel','SplitFinishKernel'},'unexpected compute task')
                        require(float(row['aiv_vec_time(us)'])>0 and float(row['aic_mac_time(us)'])==0,'missing vector evidence')
                    report['execution_evidence']=dict(operator_summary=str(summaries[0]),task_count=len(rows),task_types=dict(collections.Counter(r['Task Type'] for r in rows)),kernel_tasks=dict(collections.Counter(r['OP Type'] for r in rows if r['OP Type'] in required)),performance_acceptance='not_run')
        require(all(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==initial_sha256[str(p)] for p in inputs), 'validation inputs changed during run')
        report['validated_input_sha256'] = initial_sha256
        report['status']='passed'
        print(f"NUMERICAL_SPLIT_VALIDATION_PASS host_groups=8 npu_execution={report['npu_execution']} native_training=not_run")
        return 0
    except Exception as error:
        report['error']=str(error);print(f'NUMERICAL_SPLIT_VALIDATION_FAIL: {error}',file=sys.stderr);return 1
    finally:
        artifacts += list((root/'src/common/npu').glob('*.h'))+list((root/'src/common/npu').glob('*.asc'))
        artifacts += [p for p in source.glob('*') if p.is_file() and (p.name.startswith(('candidate_selection','numerical_split','split_statistics','histogram_cases','precise_scoring','split_finalization')) or p.name in ('precise_test_kernels.asc','runtime_no_device_guard.cc'))]
        artifacts += [source/'precision_kernel/CMakeLists.txt']
        report['artifact_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts if p.exists()}
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    raise SystemExit(main())