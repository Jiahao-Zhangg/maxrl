"""Compute-only, recoverable TACO GRPO -> LCB v6 -> TACO non-SPJ queue."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

from qwen3_experiments import grpo_compute_control as archive
from qwen3_experiments import verify_checkpoint_upload as verifier
from qwen3_experiments.code_grading import final_code, grade
from qwen3_experiments.taco_eval import digest, now, read, write
from qwen3_experiments.taco_resilience import cuda_oom, fallback_concurrency, local_log, log_tail, terminate_failed_step

MODULE = "qwen3_experiments.taco_grpo_pipeline"


def persist(plan, name, value):
    """State failures on one filesystem never kill the controller."""
    value = {**value, "updated_at": now()}
    success = False
    for base in (Path(plan["output_root"]), Path(plan["scratch"]) / "control_mirrors"):
        try:
            write(base / name, value)
            success = True
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT, errno.EIO):
                raise
    return success


def state(plan, name):
    copies = []
    for base in (Path(plan["output_root"]), Path(plan["scratch"]) / "control_mirrors"):
        try:
            copies.append(read(base / name))
        except (OSError, ValueError):
            pass
    return max(copies, key=lambda item: item.get("updated_at", ""), default={})


def safe_print(value):
    try:
        print(value, flush=True)
    except OSError:
        pass


def verify(root):
    plan = read(root / "plan.json")
    if plan["output_root"] != str(root.resolve()):
        raise ValueError("Wrong run root")
    archive.require_compute(plan["job_id"])
    if socket.gethostname().split('.')[0] != plan["node"]:
        raise ValueError("Wrong compute node")
    for name, checksum in plan["frozen_files"].items():
        if digest(root / name) != checksum:
            raise ValueError(f"Frozen input changed: {name}")
    launched = state(plan, "launch.json")
    if launched and launched["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Plan changed after launch")
    return plan


def dependency_ready(plan):
    previous = Path(plan["dependency"]["root"])
    if digest(previous / "plan.json") != plan["dependency"]["plan_sha256"]:
        raise ValueError("Predecessor plan changed")
    if not (previous / "queue_status.json").exists():
        return False
    if read(previous / "queue_status.json").get("state") != "complete":
        return False
    audit = read(previous / "report/audit.json")
    if (not audit.get("complete") or audit.get("points") != 105 or audit.get("new_points") != 90
            or not audit.get("all_rollout_ledgers_verified")
            or digest(previous / "report/metrics.json") != audit["metrics_sha256"]):
        raise ValueError("Predecessor result audit failed")
    return True


def predecessor_cache_paths(plan):
    """Allow only the completed predecessor's private model/download caches."""
    root = Path(plan['dependency']['root'])
    if digest(root / 'plan.json') != plan['dependency']['plan_sha256']:
        raise ValueError('Predecessor plan changed')
    previous = read(root / 'plan.json')
    scratch = Path(previous['scratch'])
    if not scratch.is_absolute() or scratch.resolve() != scratch or scratch == Path('/'):
        raise ValueError('Invalid predecessor scratch path')
    protected = [Path(plan[key]).resolve() for key in ('output_root', 'runtime', 'scratch', 'checkpoint_dir')]
    protected += [Path(plan['base_model']['path']).resolve(), root.resolve()]
    caches, keys = [], set()
    for stage in previous['stages']:
        key = stage['key']
        if not re.fullmatch(r'[a-z0-9_]+', key) or key in keys:
            raise ValueError('Invalid or repeated predecessor model key')
        keys.add(key)
        folder = scratch / 'models' / key
        if Path(stage['owned_model_cache']) != folder:
            raise ValueError('Model cache is outside the predecessor models directory')
        relative = f'prepared_models/{key}.json'
        if digest(root / relative) != previous['frozen_files'][relative]:
            raise ValueError('Predecessor model receipt changed')
        model = read(root / relative)
        if any(model[field] != stage['model'][field] for field in ('repo', 'revision')):
            raise ValueError('Predecessor checkpoint identity changed')
        if Path(model['path']) != folder / 'model':
            raise ValueError('Unexpected prepared model path')
        hashes = {f'model/{name}': checksum for name, checksum in model['merged_files'].items()}
        hashes.update({f'source_model/{name}': value['sha256']
                       for name, value in model.get('source_files', {}).items()})
        source_receipt = folder / 'model_receipt.json'
        if source_receipt.exists():
            if read(source_receipt) not in (model, {k: v for k, v in model.items() if k != 'path'}):
                raise ValueError('Local source-model receipt differs from its frozen copy')
            hashes['model_receipt.json'] = digest(source_receipt)
        hashes['model_prepare.lock'] = hashlib.sha256(b'').hexdigest()
        caches.append({'key': key, 'path': str(folder), 'model': stage['model'], 'expected_hashes': hashes})
    if len(keys) != 3:
        raise ValueError('Expected three predecessor evaluation models')
    models = scratch / 'models'
    if models.exists() and any(path.name not in keys for path in models.iterdir()):
        raise ValueError('Unexpected model directory; retaining it')
    caches += [{'key': name, 'path': str(scratch / name), 'download_cache': True}
               for name in ('hub', 'hf_xet')]
    for cache in caches:
        path = Path(cache['path'])
        if path.is_symlink() or path.resolve() != path:
            raise ValueError('Refusing redirected evaluation cache')
        if any(path == p or p.is_relative_to(path) or path.is_relative_to(p) for p in protected):
            raise ValueError('Refusing to remove training inputs or evaluation results')
    return caches


def cache_users(caches):
    """Detect live commands, open descriptors or mappings using these caches."""
    roots = [item['path'].rstrip('/') for item in caches]
    users = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid() or int(proc.name) == os.getpid():
                continue
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            used = any(arg == root or arg.startswith(root + '/') for arg in args for root in roots)
            if not used:
                for fd in (proc / 'fd').iterdir():
                    try:
                        target = os.readlink(fd)
                    except OSError:
                        continue
                    if any(target == root or target.startswith(root + '/') for root in roots):
                        used = True
                        break
            if not used:
                mappings = (proc / 'maps').read_text()
                used = any(root + '/' in mappings for root in roots)
            if used:
                users.append(int(proc.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return users


def cleanup_predecessor_caches(plan):
    """A durable, retryable cleanup barrier before starting training."""
    if not dependency_ready(plan):
        raise RuntimeError('Predecessor evaluation is incomplete; retaining checkpoints')
    caches = predecessor_cache_paths(plan)
    users = cache_users(caches)
    if users:
        raise RuntimeError(f'Evaluation caches are still in use by processes {users}')
    identity = {'predecessor_plan_sha256': plan['dependency']['plan_sha256'],
                'training_plan_sha256': digest(Path(plan['output_root']) / 'plan.json')}
    name = 'pretraining_cleanup.json'
    old = state(plan, name)
    if old and any(old.get(key) != value for key, value in identity.items()):
        raise ValueError('Cleanup receipt belongs to a different plan')
    if old.get('state') == 'complete' and all(not Path(c['path']).exists() for c in caches):
        return old
    previous_caches = {item['path']: item for item in old.get('caches', [])}
    # The next training has its own immutable base-model copy; check it before
    # discarding the older evaluation copies, including the reference model.
    for relative, checksum in plan['base_model']['files_sha256'].items():
        if digest(Path(plan['base_model']['path']) / relative) != checksum:
            raise ValueError('Training base-model copy is not intact')
    for cache in caches:
        folder = Path(cache['path'])
        previous_cache = previous_caches.get(str(folder), {})
        files = dict(previous_cache.get('files', {}))
        if folder.exists():
            if not folder.is_dir() or folder.stat().st_uid != os.getuid():
                raise ValueError('Invalid evaluation cache ownership or type')
            for path in folder.rglob('*'):
                if path.is_symlink() or path.stat().st_uid != os.getuid():
                    raise ValueError('Refusing redirected or foreign evaluation cache files')
                if not path.is_file():
                    continue
                relative = path.relative_to(folder)
                stat = path.stat()
                checksum = digest(path)
                current = path.stat()
                if (current.st_size, current.st_mtime_ns, current.st_ino) != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
                    raise ValueError('Evaluation cache changed during verification')
                if not cache.get('download_cache') and '.cache' not in relative.parts:
                    if checksum != cache['expected_hashes'].get(relative.as_posix()):
                        raise ValueError(f'Unrecognized or changed evaluation file: {path}')
                files[relative.as_posix()] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                             'inode': stat.st_ino, 'sha256': checksum}
        cache.update(files=files, bytes=sum(item['size'] for item in files.values()),
                     state='pending' if folder.exists() else ('deleted' if files else 'already_absent'))
    receipt = {**identity, 'state': 'cleaning', 'caches': caches, 'started_at': now(),
               'retained_training_base': plan['base_model']['path']}
    if not persist(plan, name, receipt):
        raise OSError(errno.ENOSPC, 'Cannot preserve the pretraining cleanup receipt')
    if cache_users(caches):
        raise RuntimeError('Evaluation cache became active; retaining it')
    for cache in caches:
        folder = Path(cache['path'])
        if folder.exists():
            # A partially deleted cache can be retried. Any new or changed
            # non-cache file must still match the pinned preparation receipt.
            finish_verified_deletion(folder, {'files': cache['files']})
            cache['state'] = 'deleted'
            if not persist(plan, name, receipt):
                raise OSError(errno.ENOSPC, 'Cannot preserve cleanup progress')
        if folder.exists() or folder.is_symlink():
            raise RuntimeError('Evaluation checkpoint cleanup is incomplete')
    receipt.update(state='complete', completed_at=now(),
                   deleted_bytes=sum(item['bytes'] for item in caches if item['state'] == 'deleted'))
    if not persist(plan, name, receipt):
        raise OSError(errno.ENOSPC, 'Cannot preserve the completed cleanup receipt')
    return receipt


def environment(plan):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("RAY_", "VLLM_", "SLURM_", "MAXRL_", "GRPO_", "WANDB_")) and key != "WANDB_API_KEY":
            env.pop(key)
    for key in ("CUDA_VISIBLE_DEVICES", "PYTHONHOME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env.pop(key, None)
    scratch = Path(plan["scratch"])
    env.update(PATH=str(Path(plan["python_bin"]).parent) + os.pathsep + env.get("PATH", ""),
               PYTHONPATH=plan["runtime"], PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               VLLM_ATTENTION_BACKEND="FLASH_ATTN", VLLM_USE_V1="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               NCCL_DEBUG="WARN", TORCH_NCCL_ASYNC_ERROR_HANDLING="1", SEED="79",
               TMPDIR=str(scratch / "tmp"), TRITON_CACHE_DIR=str(scratch / "triton"),
               HF_HOME=str(scratch / "hf_home"), HF_HUB_DISABLE_PROGRESS_BARS="1",
               WANDB_DIR=str(scratch / "wandb"), WANDB_MODE="online",
               WANDB_INIT_TIMEOUT="60", TACO_GRPO_ROOT=plan["output_root"])
    for path in (scratch / "tmp", scratch / "triton", scratch / "hf_home", Path(env["WANDB_DIR"])):
        path.mkdir(parents=True, exist_ok=True)
    return env


def active(plan, command):
    """Adopt this exact run's live processes after a controller restart."""
    result = []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / 'cmdline').read_bytes().decode().split('\0')
            if MODULE in args and command in args and plan['output_root'] in args and int(entry.name) != os.getpid():
                result.append(int(entry.name))
        except (OSError, UnicodeError):
            pass
    return result


def launch_child(plan, command, *extra, slurm=False, gpu=None):
    args = [plan['python_bin'], '-u', '-m', MODULE, command, '--root', plan['output_root'], *extra]
    if slurm:
        args = ['srun', f"--jobid={plan['job_id']}", '--overlap', '--nodes=1', '--ntasks=1',
                f"--nodelist={plan['node']}", '--cpus-per-task=96', '--gres=gpu:8',
                '--kill-on-bad-exit=1', '--job-name=taco-grpo-pipeline', *args]
    env = environment(plan)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    log_path = local_log(plan, 'logs/' + command + ('_' + '_'.join(extra) if extra else '') + '.log')
    with log_path.open('ab', buffering=0) as log:
        return subprocess.Popen(args, cwd=plan['runtime'], env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def valid_receipt(plan, step, receipt):
    required = {'data.pt', 'actor/config.json', 'actor/tokenizer_config.json'}
    required.update(f'actor/{kind}_world_size_8_rank_{rank}.pt'
                    for kind in ('model', 'optim', 'extra_state') for rank in range(8))
    return (receipt.get('state') in ('verified', 'archived_and_deleted')
            and receipt.get('repo_id') == f"{plan['hf_repo_prefix']}-step_{step}"
            and receipt.get('checkpoint_path') == str(Path(plan['checkpoint_dir']) / f'global_step_{step}')
            and bool(re.fullmatch('[0-9a-f]{40}', receipt.get('remote_commit', '')))
            and required.issubset(receipt.get('files', {}))
            and receipt.get('file_count') == len(receipt.get('files', {}))
            and all(v.get('size', 0) > 0 and re.fullmatch('[0-9a-f]{64}', v.get('sha256', ''))
                    for v in receipt.get('files', {}).values()))


def receipt_ready(plan, step):
    receipt = state(plan, f'hf_checkpoint_archive/receipts/global_step_{step}.json')
    return valid_receipt(plan, step, receipt) and receipt['state'] == 'archived_and_deleted'


def finish_verified_deletion(checkpoint, receipt):
    """Resume interrupted deletion, including an already-empty directory."""
    if checkpoint.is_symlink():
        raise ValueError('Refusing redirected checkpoint cleanup')
    if not checkpoint.exists():
        return
    for path in checkpoint.rglob('*'):
        if path.is_symlink():
            raise ValueError('Refusing checkpoint symlink')
        relative = path.relative_to(checkpoint)
        if '.cache' in relative.parts or not path.is_file():
            continue
        stat = path.stat()
        expected = receipt['files'].get(relative.as_posix(), {})
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(
                expected.get(key) for key in ('size', 'mtime_ns', 'inode')):
            raise ValueError('Partially deleted checkpoint changed; retaining it')
    shutil.rmtree(checkpoint)


def monitor(plan):
    from huggingface_hub import HfApi

    root, directory = Path(plan['output_root']), Path(plan['checkpoint_dir'])
    lock_path = root / 'monitor.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api = HfApi()
        while True:
            try:
                for step in plan['checkpoint_steps']:
                    checkpoint = directory / f'global_step_{step}'
                    name = f'hf_checkpoint_archive/receipts/{checkpoint.name}.json'
                    receipt = state(plan, name)
                    if receipt.get('state') == 'verified' and valid_receipt(plan, step, receipt):
                        finish_verified_deletion(checkpoint, receipt)
                        persist(plan, name, {**receipt, 'state': 'archived_and_deleted', 'reconciled': True})
                    if receipt_ready(plan, step) or not archive.checkpoint_complete(checkpoint, directory):
                        continue
                    repo = f"{plan['hf_repo_prefix']}-step_{step}"
                    persist(plan, 'checkpoint_status.json', {'state': 'uploading', 'step': step, 'pid': os.getpid()})
                    archive.ensure_public_repository(api, repo)
                    api.upload_folder(repo_id=repo, folder_path=str(checkpoint), path_in_repo=checkpoint.name,
                                      ignore_patterns=['.cache/**'], commit_message=f'Archive {checkpoint.name}')
                    def durable_receipt(path, value):
                        if not persist(plan, name, value):
                            raise OSError(errno.ENOSPC, 'Both checkpoint receipt destinations are full')
                    verifier.write_receipt = durable_receipt
                    receipt = verifier.verify_checkpoint(checkpoint, repo, root / name, api)
                    expected = {k: {f: v[f] for f in ('size', 'mtime_ns', 'inode')}
                                for k, v in receipt['files'].items()}
                    if verifier.inventory(checkpoint) != expected:
                        raise ValueError('Checkpoint changed before cleanup')
                    shutil.rmtree(checkpoint)
                    persist(plan, name, {**receipt, 'state': 'archived_and_deleted', 'deleted_at': now()})
                completed = [s for s in plan['checkpoint_steps'] if receipt_ready(plan, s)]
                persist(plan, 'checkpoint_status.json', {'state': 'complete' if len(completed) == 10 else 'monitoring',
                                                       'archived_steps': completed, 'pid': os.getpid()})
                if len(completed) == 10:
                    return 0
                time.sleep(2)
            except Exception as exc:
                persist(plan, 'checkpoint_status.json', {'state': 'retrying', 'error': str(exc), 'pid': os.getpid()})
                safe_print(traceback.format_exc())
                time.sleep(30)


def restore_checkpoint(plan, step, *, models_only=False):
    from huggingface_hub import snapshot_download

    receipt = state(plan, f'hf_checkpoint_archive/receipts/global_step_{step}.json')
    root = Path(plan['scratch']) / ('merge_source' if models_only else 'resume_source')
    names = [name for name in receipt['files'] if not models_only or
             (name.startswith('actor/') and not any(x in name for x in ('optim_world', 'extra_state_world')))]
    snapshot_download(receipt['repo_id'], revision=receipt['remote_commit'], local_dir=root,
                      allow_patterns=[f'global_step_{step}/{name}' for name in names], max_workers=4)
    checkpoint = root / f'global_step_{step}'
    for name in names:
        if digest(checkpoint / name) != receipt['files'][name]['sha256']:
            raise ValueError(f'Restored checkpoint hash mismatch: {name}')
    return checkpoint


def train(plan):
    with ExitStack() as stack:
        for path in [Path(plan['output_root']) / 'train.lock', *map(Path, plan['holder_locks'])]:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not dependency_ready(plan):
            raise RuntimeError('Predecessor evaluation is incomplete')
        if gpu_busy():
            return 75
        cleanup_predecessor_caches(plan)
        if shutil.disk_usage(plan['scratch']).free < 100 * 1024**3:
            raise RuntimeError('Need 100 GiB of compute-local free space before training')
        previous = state(plan, 'training_exit.json')
        if previous.get('exit_code') == 0:
            return 0
        attempt = previous.get('attempt', 0) + 1
        command = ['bash', str(Path(plan['runtime']) / 'qwen3_experiments/run_qwen3_1_7b_taco_grpo.sh')]
        concurrency = fallback_concurrency(plan['training']['max_num_seqs'], previous, state(plan, 'training_concurrency.json'))
        if concurrency != plan['training']['max_num_seqs']:
            if not persist(plan, 'training_concurrency.json', {
                'max_num_seqs': concurrency, 'initial_max_num_seqs': plan['training']['max_num_seqs'],
                'reason': 'automatic CUDA OOM fallback', 'failed_attempt': previous.get('attempt'),
            }):
                raise OSError(errno.ENOSPC, 'Cannot retain the OOM fallback decision')
        command += [f'actor_rollout_ref.rollout.max_num_seqs={concurrency}',
                    f'actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs={concurrency}']
        # Wait for complete local saves to finish publication before choosing a
        # resume step; the upload monitor continues independently during retries.
        for step in plan['checkpoint_steps']:
            checkpoint = Path(plan['checkpoint_dir']) / f'global_step_{step}'
            if archive.checkpoint_complete(checkpoint, Path(plan['checkpoint_dir'])) and not receipt_ready(plan, step):
                raise RuntimeError('Waiting for complete checkpoint upload before restarting training')
        archived = [s for s in plan['checkpoint_steps'] if receipt_ready(plan, s)]
        if archived:
            checkpoint = restore_checkpoint(plan, max(archived))
            command += ['trainer.resume_mode=resume_path', f'trainer.resume_from_path={checkpoint}']
        execution = {'pid': os.getpid(), 'attempt': attempt, 'started_at': now(),
                     'resume_step': max(archived, default=0), 'max_num_seqs': concurrency,
                     'rollouts_per_prompt': plan['training']['rollouts_per_prompt'],
                     'slurm_step_id': os.environ.get('SLURM_STEP_ID'),
                     'cgroup': Path('/proc/self/cgroup').read_text()}
        if not persist(plan, 'training_active.json', execution):
            raise OSError(errno.ENOSPC, 'Cannot retain the training attempt identity')
        logfile = local_log(plan, f'train_attempt_{attempt}.log')
        with logfile.open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=plan['runtime'], env=environment(plan),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        failed = {**execution, 'exit_code': code, 'finished_at': now(), 'log': str(logfile),
                  'cuda_oom': code != 0 and cuda_oom(log_tail(logfile))}
        persist(plan, f'training_attempts/attempt_{attempt}.json', failed)
        persist(plan, 'training_exit.json', failed)
        if code != 0:
            terminate_failed_step(plan, failed)
        return code


def gpu_busy():
    value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return bool(value.strip())


def verify_uploaded_folder(api, folder, repo):
    info = api.repo_info(repo_id=repo, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    hashes = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        item = remote[path.name]
        if item.size != path.stat().st_size:
            raise ValueError('Published model size mismatch')
        checksum = digest(path)
        if item.lfs:
            if item.lfs.sha256 != checksum:
                raise ValueError('Published model SHA256 mismatch')
        elif hashlib.sha1(f'blob {item.size}\0'.encode() + path.read_bytes()).hexdigest() != item.blob_id:
            raise ValueError('Published model Git blob mismatch')
        hashes[path.name] = checksum
    return {'repo': repo, 'revision': info.sha, 'path': str(folder), 'files_sha256': hashes}


def final_model(plan):
    from huggingface_hub import HfApi

    saved = state(plan, 'final_model.json')
    if saved:
        for name, checksum in saved['files_sha256'].items():
            if digest(Path(saved['path']) / name) != checksum:
                raise ValueError('Merged final model changed')
        return saved
    checkpoint = restore_checkpoint(plan, 100, models_only=True)
    destination = Path(plan['scratch']) / 'final_model'
    subprocess.run([plan['python_bin'], str(Path(plan['runtime']) / 'scripts/model_merger.py'), 'merge',
                    '--backend', 'fsdp', '--local_dir', str(checkpoint / 'actor'), '--target_dir', str(destination)],
                   env=environment(plan), cwd=plan['runtime'], check=True)
    api = HfApi()
    archive.ensure_public_repository(api, plan['hf_repo_prefix'] + '-final')
    api.upload_folder(repo_id=plan['hf_repo_prefix'] + '-final', folder_path=destination, ignore_patterns=['.cache/**'])
    record = verify_uploaded_folder(api, destination, plan['hf_repo_prefix'] + '-final')
    if not persist(plan, 'final_model.json', record):
        raise OSError('Cannot persist final model identity')
    shutil.rmtree(checkpoint.parent)
    return record


def worker(plan, dataset, rank):
    from vllm import LLM, SamplingParams

    root = Path(plan['output_root'])
    questions = read(root / f'data/{dataset}.json')[rank::8]
    directory = root / 'evaluation' / dataset / 'responses'
    directory.mkdir(parents=True, exist_ok=True)
    model = state(plan, 'final_model.json')
    pending = []
    for question in questions:
        path = directory / f"{question['source_index']}.json"
        if not path.exists():
            pending.append(question)
        else:
            saved = read(path)
            if (saved['id'] != question['id'] or saved['model_revision'] != model['revision']
                    or saved['plan_sha256'] != digest(root / 'plan.json')):
                raise ValueError('Existing evaluation response has a different identity')
    if not pending:
        return 0
    engine = LLM(model=model['path'], tokenizer=model['path'], **plan['evaluation_engine'])
    for offset in range(0, len(pending), 64):
        batch = pending[offset:offset + 64]
        outputs = engine.generate([{'prompt_token_ids': q['prompt_token_ids']} for q in batch],
                                  [SamplingParams(**plan['sampling'], seed=q['source_index']) for q in batch],
                                  use_tqdm=False)
        for question, output in zip(batch, outputs):
            if len(output.outputs) != 1:
                raise ValueError('Expected one sample per evaluation question')
            sample = output.outputs[0]
            write(directory / f"{question['source_index']}.json", {
                'id': question['id'], 'index': question['source_index'], 'response': sample.text,
                'token_ids': list(sample.token_ids), 'finish_reason': sample.finish_reason,
                'model_revision': model['revision'], 'plan_sha256': digest(root / 'plan.json'),
            })
    return 0


def evaluate(plan, dataset):
    root = Path(plan['output_root'])
    folder = root / 'evaluation' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in [folder / 'run.lock', *map(Path, plan['holder_locks'])]:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_busy():
            return 75
        children = [launch_child(plan, 'worker', '--dataset', dataset, '--rank', str(rank), gpu=rank) for rank in range(8)]
        try:
            codes = [child.wait() for child in children]
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
        if any(codes):
            raise RuntimeError(f'Evaluation workers exited: {codes}')
    questions = read(root / f'data/{dataset}.json')
    grading_plan = read(root / 'grading_plan.json')
    def grade_one(question):
        index = question['source_index']
        response_path = folder / 'responses' / f'{index}.json'
        response = read(response_path)
        if response['id'] != question['id'] or response['plan_sha256'] != digest(root / 'plan.json'):
            raise ValueError('Response belongs to another question or plan')
        target = folder / 'grades' / f'{index}.json'
        if target.exists():
            saved = read(target)
            if saved['response_sha256'] == digest(response_path):
                return saved
        code, reason = final_code(response['response'])
        if question['test_case_issue']:
            result = {'score': 0.0, 'error': question['test_case_issue'], 'results': []}
        elif reason != 'ok':
            result = {'score': 0.0, 'error': reason, 'results': []}
        else:
            result = grade(grading_plan, question['input_output'], code, 'lcb' if dataset == 'lcb_v6' else 'taco')
        record = {**result, 'id': question['id'], 'difficulty': question['difficulty'],
                  'tokens': len(response['token_ids']), 'response_sha256': digest(response_path)}
        write(target, record)
        return record
    with ThreadPoolExecutor(max_workers=32) as pool:
        records = list(pool.map(grade_one, questions))
    groups = {'all': records}
    for record in records:
        groups.setdefault(record['difficulty'], []).append(record)
    metrics = {name: {'questions': len(group), 'correct': sum(r['score'] for r in group),
                      'pass_at_1_percent': 100 * sum(r['score'] for r in group) / len(group),
                      'mean_output_tokens': sum(r['tokens'] for r in group) / len(group)}
               for name, group in groups.items()}
    write(folder / 'metrics.json', metrics)
    persist(plan, f'evaluation/{dataset}/audit.json', {
        'complete': True, 'questions': len(questions), 'plan_sha256': digest(root / 'plan.json'),
        'metrics_sha256': digest(folder / 'metrics.json'), 'model': state(plan, 'final_model.json'),
        'response_hashes': {str(q['source_index']): digest(folder / 'responses' / f"{q['source_index']}.json")
                            for q in questions},
    })
    return 0


def evaluation_complete(plan, dataset):
    audit = state(plan, f'evaluation/{dataset}/audit.json')
    if not audit.get('complete'):
        return False
    folder = Path(plan['output_root']) / 'evaluation' / dataset
    expected = 175 if dataset == 'lcb_v6' else 782
    if audit['questions'] != expected or digest(folder / 'metrics.json') != audit['metrics_sha256']:
        raise ValueError('Evaluation audit mismatch')
    if audit['plan_sha256'] != digest(Path(plan['output_root']) / 'plan.json'):
        raise ValueError('Evaluation belongs to another plan')
    return True


def queue(plan):
    root = Path(plan['output_root'])
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        while True:
            children = [child for child in children if child.poll() is None]
            status = {'pid': os.getpid(), 'node': socket.gethostname(),
                      'disks_free_gib': {p: shutil.disk_usage(p).free / 1024**3 for p in (str(root), plan['scratch'], '/project/flame')}}
            try:
                if not dependency_ready(plan):
                    status.update(state='waiting_for_current_evaluation_queue', predecessor=plan['dependency']['root'])
                else:
                    if not all(receipt_ready(plan, s) for s in plan['checkpoint_steps']) and not active(plan, 'monitor'):
                        children.append(launch_child(plan, 'monitor'))
                    result = state(plan, 'training_exit.json')
                    if result.get('exit_code') != 0:
                        status.update(state='training_or_recovering', previous_exit=result, active_pids=active(plan, 'train'))
                        if not status['active_pids'] and result.get('exit_code') is not None:
                            terminate_failed_step(plan, result)
                        if state(plan, 'pretraining_cleanup.json').get('state') != 'complete':
                            status['state'] = 'cleaning_evaluation_checkpoints'
                        if not status['active_pids'] and not gpu_busy():
                            children.append(launch_child(plan, 'train', slurm=True))
                    elif not all(receipt_ready(plan, s) for s in plan['checkpoint_steps']):
                        status.update(state='archiving_checkpoints')
                    else:
                        final_model(plan)
                        for dataset in ('lcb_v6', 'taco_test'):
                            if evaluation_complete(plan, dataset):
                                continue
                            status.update(state='evaluating', dataset=dataset)
                            if not active(plan, 'evaluate') and not active(plan, 'worker') and not gpu_busy():
                                children.append(launch_child(plan, 'evaluate', '--dataset', dataset, slurm=True))
                            break
                        else:
                            lines = ['# TACO GRPO final checkpoint', '',
                                     'Thinking on; 32,768 output tokens; temperature/top-p/top-k 0.6/0.95/20; '
                                     'one sample per question; after-thinking grading; no extra EOS gate.', '',
                                     '| Dataset | Difficulty | Correct | Questions | Pass@1 (%) |',
                                     '|---|---|---:|---:|---:|']
                            for name in ('lcb_v6', 'taco_test'):
                                for difficulty, item in read(root / f'evaluation/{name}/metrics.json').items():
                                    lines.append(f"| {name} | {difficulty} | {item['correct']:.0f} | "
                                                 f"{item['questions']} | {item['pass_at_1_percent']:.2f} |")
                            (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
                            persist(plan, 'queue_status.json', {**status, 'state': 'complete'})
                            return 0
                persist(plan, 'queue_status.json', status)
            except Exception as exc:
                persist(plan, 'queue_status.json', {**status, 'state': 'retrying', 'error': str(exc)})
                safe_print(traceback.format_exc())
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('queue', 'monitor', 'train', 'evaluate', 'worker'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--dataset', choices=('lcb_v6', 'taco_test'))
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    plan = verify(args.root)
    if args.command == 'worker':
        return worker(plan, args.dataset, args.rank)
    if args.command == 'evaluate':
        return evaluate(plan, args.dataset)
    return {'queue': queue, 'monitor': monitor, 'train': train}[args.command](plan)


if __name__ == '__main__':
    raise SystemExit(main())
